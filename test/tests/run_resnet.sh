#!/bin/bash

if [ $# -lt 4 ]; then
  echo "Usage: "
  echo "  $0 MPI_NUM_PROCS BS EPOCH MPI_LOG"
  exit
fi
if [ -z $PROJECT_DIR ]; then
  echo "PROJECT_DIR is not set"
  exit
fi
if [ -z $DATA_DIR ]; then
  echo "DATA_DIR is not set, use default /dev/shm/ILSVRC2012/"
  export DATA_DIR=/dev/shm/ILSVRC2012/
fi
if [ -z $CFG_FOLDER ]; then
  echo "CFG_FOLDER is not set, use default /tmp"
  CFG_FOLDER=/tmp
fi

MODEL_TRAIN=rn50

#training specific env variable
export FLAGS_nnp_dtype_cfg=../datatype.config   #relative folder
export MPI_NUM_PROCS=$1
export BATCH_SIZE=$2
export EPOCH_NUM=$3
MPI_LOG=$4/${MODEL_TRAIN}_${MPI_NUM_PROCS}cards_bs${BATCH_SIZE}_epoch${EPOCH_NUM}

#export AEON_THREADS=0-4:10-14:20-24:30-34:40-44:50-54:60-64:70-74
#export AEON_THREADS=0-4,80-84:10-14,90-94:20-24,100-104:30-34,110-114:40-44,120-124:50-54,130-134:60-64,140-144:70-74,150-154
RANKFILE=$CFG_FOLDER/rankfile_rn50.txt
HOSTFILE=$CFG_FOLDER/hostfile.txt
export AEON_THREADS=`cat $CFG_FOLDER/aeon.txt`

RANK_NUM=`cat ${RANKFILE} | sed '/^\s*$/d' | wc -l`
if [ "$MPI_NUM_PROCS" -ne "$RANK_NUM" ]; then
  echo "MPI_NUM_PROCS doesn't match with rankfile"
  exit
fi

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

starttime=`date +'%Y-%m-%d %H:%M:%S'`
echo "Start RN50 Training on ${MPI_NUM_PROCS} card, BS: ${BATCH_SIZE}, epochs: ${EPOCH_NUM}, log: ${MPI_LOG} at $starttime"

$MPI_RUN --allow-run-as-root --output-filename $MPI_LOG \
	--mca oob_tcp_if_exclude docker0,lo --mca btl_tcp_if_exclude docker0,lo \
	--hostfile $HOSTFILE \
	-rf $RANKFILE \
	-x MPI_NUM_PROCS \
	-x AEON_THREADS \
	-x BATCH_SIZE \
	-x EPOCH_NUM \
	-x PROJECT_DIR \
	-x FLAGS_nnp_dtype_cfg \
	-x DATA_DIR $PROJECT_DIR/tests/dist_train_nnp_rn50.sh

endtime=`date +'%Y-%m-%d %H:%M:%S'`

start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "Finish RN50 training task at $endtime, cost $((end_seconds-start_seconds)) sec"
