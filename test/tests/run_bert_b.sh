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
if [ -z $CFG_FOLDER ]; then
  echo "CFG_FOLDER is not set, use default /tmp"
  CFG_FOLDER=/tmp
fi

MODEL_TRAIN=bert_b

#training specific env variable 
export FLAGS_nnp_dtype_cfg=datatype.config  #relative folder, set to use BF16
#export FLAGS_nnp_dtype_cfg=         #unset to use FP32
export NUM_BUCKET=1
export MPI_NUM_PROCS=$1
export BATCH_SIZE=$2
export EPOCH_NUM=$3
MPI_LOG=$4/${MODEL_TRAIN}_${MPI_NUM_PROCS}cards_bs${BATCH_SIZE}_epoch${EPOCH_NUM}

RANKFILE=$CFG_FOLDER/rankfile_bert.txt
HOSTFILE=$CFG_FOLDER/hostfile.txt

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
echo "Start BERT_B Training on ${MPI_NUM_PROCS} card, BS: ${BATCH_SIZE}, epochs: ${EPOCH_NUM}, log: ${MPI_LOG} at $starttime"

$MPI_RUN --allow-run-as-root --output-filename $MPI_LOG  \
        --mca oob_tcp_if_exclude docker0,lo --mca btl_tcp_if_exclude docker0,lo \
	--hostfile $HOSTFILE \
        -rf $RANKFILE \
        -x BATCH_SIZE \
        -x EPOCH_NUM \
        -x PROJECT_DIR \
	-x FLAGS_nnp_dtype_cfg \
	-x NUM_BUCKET \
        -x MPI_NUM_PROCS $PROJECT_DIR/tests/dist_train_nnp_bert_b.sh

endtime=`date +'%Y-%m-%d %H:%M:%S'`

start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "Finish BERT_B training task at $endtime, cost $((end_seconds-start_seconds)) sec"

