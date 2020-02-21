#!/bin/bash
#input parameters
if [ -f $HOSTS ]; then
    echo "HOSTS is not set, use localhost"
    HOSTS=localhost
fi
if [ -f $CFG_FOLDER ]; then
    echo "CFG_FOLDER is not set, use /tmp"
    CFG_FOLDER=/tmp
fi
if [ -f $CARDS_PER_CHASSIS ]; then
    echo "CARDS_PER_CHASSIS is not set, use 8"
    CARDS_PER_CHASSIS=8
fi
if [ -f $RESET_NET ]; then
    echo "RESET_NET is not set, use true"
    RESET_NET=true
fi
if [ -f $EXCLUSIVE_MODE ]; then
    echo "EXCLUSIVE_MODE is not set, use false"
    EXCLUSIVE_MODE=false
fi

#get cpus info
phy_cpu_num=`cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l`
tmp=`cat /proc/cpuinfo| grep "cpu cores"| uniq`
phy_core_num_per_cpu=${tmp#*:}
logic_cpu_num=`cat /proc/cpuinfo| grep "processor"| wc -l`
hyperthread=1
if [ $((phy_cpu_num*phy_core_num_per_cpu)) -eq $logic_cpu_num ]; then
    hyperthread=0
    echo "HyperThread is not enabled"
fi

#count total host
host_array=(${HOSTS//,/ })
host_num=0
for host in ${host_array[@]}
do
    let $((host_num++))
done

#make file names and truncate zero
host_file=$CFG_FOLDER/hostfile_$((host_num*CARDS_PER_CHASSIS)).txt
rank_file_rn50=$CFG_FOLDER/rankfile_rn50_$((host_num*CARDS_PER_CHASSIS)).txt
rank_file_bert=$CFG_FOLDER/rankfile_bert_$((host_num*CARDS_PER_CHASSIS)).txt
aeon_file=$CFG_FOLDER/aeon.txt
rm -rf $host_file $rank_file_rn50 $rank_file_bert $aeon_file
touch $host_file $rank_file_rn50 $rank_file_bert $aeon_file

#for single card hostfile, rankfile, aeon file
if [ $CARDS_PER_CHASSIS -eq 1 ]; then
    #hostfile
    echo "$host slots=1" >> $host_file

    cores_used=$((phy_core_num_per_cpu*phy_cpu_num/8))
    if [ $EXCLUSIVE_MODE = "true" ]; then
	cores_used=$phy_core_num_per_cpu
    fi
    #RN50 rankfile
    echo "rank 0=$host slot=0:0-$((cores_used/2-1))" >> $rank_file_rn50
    #BERT rankfile
    echo "rank 0=$host slot=0:$((cores_used-1))" >> $rank_file_bert
    #AEON file
    offset_s=$((cores_used/2))
    offset_e=$((cores_used-1))
    aeon_threads=$offset_s-$offset_e
    if [ $hyperthread -eq 1 ]; then
        numa=$((phy_core_num_per_cpu*phy_cpu_num))
        aeon_threads=$aeon_threads":$((offset_s+numa))-$((offset_e+numa))"
    fi
    echo $aeon_threads >> $aeon_file
else
    cards_per_socket=$((CARDS_PER_CHASSIS/phy_cpu_num))
    cores_per_card=$((phy_core_num_per_cpu/cards_per_socket))

    #for multi cards hostfile, rankfile, aeon file
    chassis_id=0
    #generate aeon items
    #aeon_threads=0-4,80-84:10-14,90-94:20-24,100-104:30-34,110-114:40-44,120-124:50-54,130-134:60-64,140-144:70-74,150-154
    rn50_pd_cores=cores_per_card/2
    aeon_threads=""

    for i in $(seq 0 $((CARDS_PER_CHASSIS-1)))
    do
        offset_s=$((i*cores_per_card+cores_per_card/2))
        offset_e=$(((i+1)*cores_per_card-1))
        numa=$((phy_core_num_per_cpu*phy_cpu_num))

        aeon_threads=$aeon_threads"$offset_s-$offset_e"
        if [ $hyperthread -eq 1 ]; then
	    aeon_threads=$aeon_threads":$((offset_s+numa))-$((offset_e+numa))"
        fi
        aeon_threads="$aeon_threads,"
    done
    echo $aeon_threads >> $aeon_file

    for host in ${host_array[@]}
    do
        echo "$host slots=$CARDS_PER_CHASSIS" >> $host_file
    
        #generate rankfile items
	for i in $(seq 0 $((CARDS_PER_CHASSIS-1)))
        do
	    slot_id1=$((i/cards_per_socket))
            slot_id2=$((i%cards_per_socket))
            #RN50 rankfile
            echo "rank $((chassis_id*CARDS_PER_CHASSIS+i))=$host slot=$slot_id1:$((slot_id2*cores_per_card))-$((slot_id2*cores_per_card+cores_per_card/2-1))" >> $rank_file_rn50
            #BERT rankfile
            echo "rank $((chassis_id*CARDS_PER_CHASSIS+i))=$host slot=$slot_id1:$((slot_id2*cores_per_card))-$((slot_id2*cores_per_card+cores_per_card-1))" >> $rank_file_bert
        done

        let $((chassis_id++))
    done
fi

echo "Generating necesssary hostfile & rankfile in $CFG_FOLDER done"

if [ ! -f $USE_DEFAULT_MPI ]; then
    echo "Using default MPI installation."
    MPI_RUN=mpirun
else
    MPI_RUN=`find /usr/lib* -name mpirun | grep openmpi4`
    if [ ! -f $MPI_RUN ]; then
        echo "Cannot find MPI4 installed in the system. Fallback to default MPI."
        MPI_RUN=mpirun
    fi
fi

if [ $RESET_NET = "true" ]; then
    echo "Resetting hosts: $HOSTS"
    echo "rmmod intel_nnp"
    sudo $MPI_RUN --allow-run-as-root -n $host_num -host $HOSTS rmmod intel_nnp || exit
    echo "modprobe intel_nnp"
    sudo $MPI_RUN --allow-run-as-root -n $host_num -host $HOSTS modprobe intel_nnp || exit
    echo "nnptool net -s"
    sudo $MPI_RUN --allow-run-as-root -n $host_num -host $HOSTS nnptool net -s
    echo "Resetting hosts done"
fi
