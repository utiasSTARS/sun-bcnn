TOOLS=/caffe-sl/build/tools
PRETRAIN_WEIGHTS_MODEL=/places_googlenet.caffemodel
LOG_FILENAME=sunbcnn_train.log

$TOOLS/caffe train --solver=solver_sunbcnn.prototxt --weights=$PRETRAIN_WEIGHTS_MODEL 2>&1 | tee $LOG_FILENAME
