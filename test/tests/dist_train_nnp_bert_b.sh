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
#export BATCH_SIZE=32
export LR_RATE=1e-4
export WEIGHT_DECAY=0.01
export MAX_LEN=128
export TRAIN_DATA_DIR=data/train
export VALIDATION_DATA_DIR=data/validation
export CONFIG_PATH=data/demo_config/bert_config.json
export VOCAB_PATH=data/demo_config/vocab.txt
export FLAGS_ngraph_backend=NNP
export NGRAPH_PROVENANCE_ENABLE=true
export FLAGS_use_ngraph=true
export FLAGS_with_distributed=true
export NNP_ENABLE_BATCHED_ALLREDUCE=1
export CPU_NUM=$MPI_NUM_PROCS
export VALIDATION_STEP=$((960/$MPI_NUM_PROCS))
#export NUM_BUCKET=1
#export FLAGS_nnp_dtype_cfg=datatype.config

python ./train.py \
    --is_distributed false \
    --use_cuda false \
    --weight_sharing true \
    --batch_size $BATCH_SIZE \
    --data_dir $TRAIN_DATA_DIR \
    --validation_set_dir $VALIDATION_DATA_DIR \
    --bert_config_path $CONFIG_PATH \
    --vocab_path $VOCAB_PATH \
    --generate_neg_sample true \
    --checkpoints ./output \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_len $MAX_LEN \
    --skip_steps $SKIP_STEPS \
    --validation_steps $VALIDATION_STEP \
    --num_iteration_per_drop_scope 10 \
    --use_fp16 false \
    --loss_scaling 8.0 \
    --in_tokens false \
    --num_buckets $NUM_BUCKET \
    --do_profile false \
    --drop_last_batch true \
    --ngraph_distributed true \
    --strong_scaling false \
    --epoch $EPOCH_NUM 
