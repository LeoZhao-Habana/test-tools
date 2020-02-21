#!/bin/bash
cd $PROJECT_DIR/ngraph-models/paddle_scripts/BERT

DISTRO=`grep DISTRIB_ID /etc/*-release | awk -F '=' '{print $2}'`
if [ "$DISTRO" == "Ubuntu" ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libargon_api.so
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi4/lib:${LD_LIBRARY_PATH}
else
  export LD_PRELOAD=/usr/lib64/libargon_api.so
  export LD_LIBRARY_PATH=/usr/lib64/openmpi4/lib64:${LD_LIBRARY_PATH}
fi

centosDistroFile=/etc/centos-release
if [ -f "$centosDistroFile" ]; then
	centosDistro=`cat /etc/centos-release`
	if [[ "$centosDistro" == *"CentOS release 6.3"* ]]; then 
		source /opt/rh/python27/enable
        fi
fi
export SAVE_STEPS=1000000
export SKIP_STEPS=1 #How many steps to print on
export BATCH_SIZE=1
#export LR_RATE=1e-4
export WEIGHT_DECAY=0.01
export MAX_LEN=384
#export TRAIN_DATA_DIR=data/train
#export VALIDATION_DATA_DIR=data/validation
#export CONFIG_PATH=data/demo_config/bert_config_large.json
export CONFIG_PATH=./uncased_L-24_H-1024_A-16/bert_config.json
#export VOCAB_PATH=data/demo_config/vocab.txt
export FLAGS_ngraph_backend=NNP
export NGRAPH_PROVENANCE_ENABLE=true
export FLAGS_use_ngraph=true
export FLAGS_with_distributed=true
export NNP_ENABLE_BATCHED_ALLREDUCE=1
export CPU_NUM=$MPI_NUM_PROCS
#export VALIDATION_STEP=$((960/$MPI_NUM_PROCS))
export FLAGS_nnp_dtype_cfg=datatype.config

BERT_LARGE_PATH="./uncased_L-24_H-1024_A-16"
CHECKPOINT_PATH=./output_large
SQUAD_PATH=./squad_v11

python -u ./run_squad.py \
    --use_cuda false \
    --batch_size $BATCH_SIZE \
    --in_tokens false \
    --init_pretraining_params ${BERT_LARGE_PATH}/params \
    --checkpoints ${CHECKPOINT_PATH} \
    --vocab_path ${BERT_LARGE_PATH}/vocab.txt \
    --bert_config_path $CONFIG_PATH \
    --do_train true \
    --do_predict true \
    --save_steps $SAVE_STEPS \
    --warmup_proportion 0.1 \
    --weight_decay $WEIGHT_DECAY \
    --epoch $EPOCH_NUM \
    --max_seq_len $MAX_LEN \
    --predict_file ${SQUAD_PATH}/dev-v1.1.json \
    --do_lower_case true \
    --doc_stride 128 \
    --train_file ${SQUAD_PATH}/train-v1.1.json \
    --learning_rate 3e-5 \
    --lr_scheduler linear_warmup_decay \
    --skip_steps $SKIP_STEPS \
    --num_buckets 1 \
    --epoch $EPOCH_NUM
