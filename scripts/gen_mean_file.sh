#!/usr/bin/env sh
DATA=/media/stars/SunBPN2
TOOLS=~/Research/caffe-posenet/build/tools

$TOOLS/compute_image_mean $DATA/kitti_sun_train_07_lmdb $DATA/kitti_sun_train_07_lmdb_mean.binaryproto
echo "Done 07."
$TOOLS/compute_image_mean $DATA/kitti_sun_train_07_lmdb $DATA/kitti_sun_train_08_lmdb_mean.binaryproto
echo "Done 08."
$TOOLS/compute_image_mean $DATA/kitti_sun_train_07_lmdb $DATA/kitti_sun_train_09_lmdb_mean.binaryproto
echo "Done 09."
$TOOLS/compute_image_mean $DATA/kitti_sun_train_07_lmdb $DATA/kitti_sun_train_10_lmdb_mean.binaryproto
echo "Done 10."
