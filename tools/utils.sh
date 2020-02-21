#!/bin/bash

#get cpus info
_PHY_CPU_NUM=`cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l`
tmp=`cat /proc/cpuinfo| grep "cpu cores"| uniq`
_PHY_CORE_NUM_PER_CPU=${tmp#*:}
_LOGIC_CORE_NUM=`cat /proc/cpuinfo| grep "processor"| wc -l`
_NUMA=$((_PHY_CORE_NUM_PER_CPU*_PHY_CPU_NUM))
_HYPER_THREAD=1
if [ $((_PHY_CPU_NUM*_PHY_CORE_NUM_PER_CPU)) -eq $_LOGIC_CORE_NUM ]; then
    _HYPER_THREAD=0
    echo "HyperThread is not enabled"
fi

#input parameters
if [ -f $HOSTS ]; then
    echo "HOSTS is not set, use localhost"
    HOSTS=localhost
fi
if [ -f $CFG_FOLDER ]; then
    echo "CFG_FOLDER is not set, use /tmp"
    CFG_FOLDER=/tmp
fi
if [ -f $EXCLUSIVE_MODE ]; then
    echo "EXCLUSIVE_MODE is not set, use false"
    EXCLUSIVE_MODE=false
fi
CARDS_PER_CHASSIS=8

#count total host
host_array=(${HOSTS//,/ })
_HOST_NUM=0
for host in ${host_array[@]}
do
	((_HOST_NUM++))
done


#make file names and truncate zero
#_HOST_CFGFILE=$CFG_FOLDER/hostfile_$((_HOST_NUM*CARDS_PER_CHASSIS)).txt
#_RANK_CFGFILE_RN50=$CFG_FOLDER/rankfile_rn50_$((_HOST_NUM*CARDS_PER_CHASSIS)).txt
#_RANK_CFGFILE_BERT=$CFG_FOLDER/rankfile_bert_$((_HOST_NUM*CARDS_PER_CHASSIS)).txt
_HOST_CFGFILE=$CFG_FOLDER/hostfile.txt
_RANK_CFGFILE_RN50=$CFG_FOLDER/rankfile_rn50.txt
_RANK_CFGFILE_BERT=$CFG_FOLDER/rankfile_bert.txt
_AEON_CFGFILE=$CFG_FOLDER/aeon.txt

reset_cfgfiles()
{
	rm -rf $_HOST_CFGFILE $_RANK_CFGFILE_RN50 $_RANK_CFGFILE_BERT $_AEON_CFGFILE
	touch $_HOST_CFGFILE $_RANK_CFGFILE_RN50 $_RANK_CFGFILE_BERT $_AEON_CFGFILE
}

generate_aeon()
{
	if [ $EXCLUSIVE_MODE = "true" ]; then
		return
	fi

	cores_per_card=$((_PHY_CORE_NUM_PER_CPU*_PHY_CPU_NUM/CARDS_PER_CHASSIS))
	aeon_threads=""
	card_id=0
	while(( $card_id<$CARDS_PER_CHASSIS ))
	do
		offset_s=$((card_id*cores_per_card+cores_per_card/2))
		offset_e=$(((card_id+1)*cores_per_card-1))
		aeon_threads=$aeon_threads"$offset_s-$offset_e"
		if [ $_HYPER_THREAD -eq 1 ]; then
			aeon_threads=$aeon_threads":$((offset_s+_NUMA))-$((offset_e+_NUMA))"
		fi
		aeon_threads="$aeon_threads,"
		((card_id++))
	done
	echo $aeon_threads >> $_AEON_CFGFILE
}

config_single_card()
{
	reset_cfgfiles

	#hostfile
	echo "$host slots=1" >> $_HOST_CFGFILE

	cores_per_card=$((_PHY_CORE_NUM_PER_CPU*_PHY_CPU_NUM/CARDS_PER_CHASSIS))
	if [ $EXCLUSIVE_MODE = "true" ]; then
		cores_per_card=$_PHY_CORE_NUM_PER_CPU
	fi

	#RN50 rankfile
	echo "rank 0=$host slot=0:0-$((cores_per_card/2-1))" >> $_RANK_CFGFILE_RN50
	#BERT rankfile
	echo "rank 0=$host slot=0:$((cores_per_card-1))" >> $_RANK_CFGFILE_BERT
	#AEON file
	generate_aeon

	echo "Generating necesssary hostfile & rankfile in $CFG_FOLDER done"
}

config_multi_card()
{
	reset_cfgfiles

	cards_per_socket=$((CARDS_PER_CHASSIS/_PHY_CPU_NUM))
	cores_per_card=$((_PHY_CORE_NUM_PER_CPU/cards_per_socket))

	#aeon_threads=0-4,80-84:10-14,90-94:20-24,100-104:30-34,110-114:40-44,120-124:50-54,130-134:60-64,140-144:70-74,150-154
	#generate aeon cfg file
	generate_aeon 

        host_array=(${HOSTS//,/ })
	host_id=0
	for host in ${host_array[@]}
	do
		echo "$host slots=$CARDS_PER_CHASSIS" >> $_HOST_CFGFILE

		card_id=0
		while(( $card_id<$CARDS_PER_CHASSIS ))
		do
			id1=$((card_id/cards_per_socket))
			id2=$((card_id%cards_per_socket))
			rank_id=$((host_id*CARDS_PER_CHASSIS+card_id))
			offset_s=$((id2*cores_per_card))
			#RN50 rankfile
			echo "rank $rank_id=$host slot=$id1:$offset_s-$((offset_s+cores_per_card/2-1))" >> $_RANK_CFGFILE_RN50
			#BERT rankfile
			echo "rank $rank_id=$host slot=$id1:$offset_s-$((offset_s+cores_per_card-1))" >> $_RANK_CFGFILE_BERT
			((card_id++))
		done
		((host_id++))
	done

	echo "Generating necesssary hostfile & rankfile in $CFG_FOLDER done"
}

reset_net()
{
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
	
	if [ 1 -eq $_HOST_NUM ]; then
		CMD_PREFIX="sudo "
		echo "Resetting localhost"
	else
		CMD_PREFIX="sudo $MPI_RUN --allow-run-as-root -n $_HOST_NUM -host $HOSTS "
		echo "Resetting hosts: $HOSTS"
	fi

	echo "rmmod intel_nnp"
	$CMD_PREFIX rmmod intel_nnp || exit
	echo "modprobe intel_nnp"
	$CMD_PREFIX modprobe intel_nnp || exit
	echo "nnptool net -s"
	$CMD_PREFIX nnptool net -s
	echo "Resetting hosts done"
}

scp_bc()
{
        host_array=(${HOSTS//,/ })
	for host in ${host_array[@]}
	do
		scp -r $1 $host:$2
	done
}

scp_gather()
{
        host_array=(${HOSTS//,/ })
	for host in ${host_array[@]}
	do
		scp -r $host:$1 $2
	done
}
