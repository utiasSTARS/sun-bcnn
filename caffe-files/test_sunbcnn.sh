#!/usr/bin/env sh
WEIGHTS=/weights
DATA=/dataset
MEANFILE=/meanfile

python ../scripts/test_sun_bcnn.py --model=../caffe-models/test_sunbcnn.prototxt --weights=$WEIGHTS --dataset=$DATA --meanfile=$MEANFILE
