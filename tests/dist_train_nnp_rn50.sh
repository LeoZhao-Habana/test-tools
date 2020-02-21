#!/bin/bash
cd $PROJECT_DIR/ngraph-models/paddle_scripts/resnet50/dist_train/

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

export NGRAPH_PROVENANCE_ENABLE=true
export FLAGS_use_ngraph=true
export FLAGS_with_distributed=true
export FLAGS_ngraph_backend=NNP
export NRV_AR_FORCE_HWCN_TENSOR_LAYOUT=1
export NNP_FORCE_LAYOUT=HWCxN
#export FLAGS_nnp_dtype_cfg=../datatype.config
export NNP_ENABLE_BATCHED_ALLREDUCE=1
export CPU_NUM=$MPI_NUM_PROCS

python ./dist_train.py \
 --data_dir=$DATA_DIR \
 --with_mem_opt=True \
 --batch_size=$BATCH_SIZE \
 --num_epochs=$EPOCH_NUM \
 --start_test_pass=88 \
 --log_level=3 \
 --print_every=30 \
 --report_training_topk=False \
 --use_aeon_reader=True \
 --aeon_random_seed=1 \
 --start_checkpoint_pass=90 \
 --aeon_cpu_list=$AEON_THREADS \
 --use_gpu=False
