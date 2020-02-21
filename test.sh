export PROJECT_DIR=/home/scr-test/TE-test
export DATA_DIR=/dev/shm/ILSVRC2012
export CFG_FOLDER=/tmp

LOG_FOLDER=/home/scr-tset/TE-test/logs

CARD=1
./tests/run_bert_b.sh $CARD  32 5 $LOG_FOLDER
./tests/run_resnet.sh $CARD 128 3 $LOG_FOLDER
./tests/run_resnet.sh $CARD 256 3 $LOG_FOLDER

#CARD=16
#./tests/run_bert_b.sh $CARD  32 100 $LOG_FOLDER
#./tests/run_resnet.sh $CARD 128  90 $LOG_FOLDER
#./tests/run_resnet.sh $CARD 256  90 $LOG_FOLDER
