#!/usr/bin/env sh
DATA=/sunbcnn-data
TOOLS=~/caffe-sl/build/tools

$TOOLS/compute_image_mean $DATA/train_lmdb_file $DATA/mean_file.binaryproto
echo "Done."
