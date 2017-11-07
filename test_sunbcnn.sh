#!/usr/bin/env sh
TRAINED_MODEL=/home/valentinp/Research/stars-sun-cnn/KITTI/train/UTC+2-Shuffle/kitti_sunbcnn_0_snapshot_iter_30000.caffemodel
DATA=/media/raid5-array/experiments/Sun-BCNN/KITTI-UTC+2-Shuffle/kitti_sun_utc+2_test_00_lmdb
MEANFILE=/media/raid5-array/experiments/Sun-BCNN/KITTI-UTC+2-Shuffle/kitti_sun_utc+2_train_00_lmdb_mean.binaryproto

python3 scripts/test_sun_bcnn.py --model=caffe-files/test_sunbcnn.prototxt --weights=$TRAINED_MODEL --dataset=$DATA --meanfile=$MEANFILE
