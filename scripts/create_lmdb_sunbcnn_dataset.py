caffe_root = '/home/valentinp/Research/caffe-sl/'
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import lmdb
import caffe
import random
import cv2
import time
import pylab as plt

directory = '/home/valentinp/Desktop/stars-sun-cnn/kitti_extract/'
datasets = [
    'kitti_sun_test_00.txt',
    'kitti_sun_test_01.txt',
    'kitti_sun_test_02.txt',
    'kitti_sun_test_04.txt',
    'kitti_sun_test_05.txt',
    'kitti_sun_test_06.txt',
    'kitti_sun_test_07.txt',
    'kitti_sun_test_08.txt',
    'kitti_sun_test_09.txt',
    'kitti_sun_test_10.txt',
    'kitti_sun_train_00.txt',
    'kitti_sun_train_01.txt',
    'kitti_sun_train_02.txt',
    'kitti_sun_train_04.txt',
    'kitti_sun_train_05.txt',
    'kitti_sun_train_06.txt',
    'kitti_sun_train_07.txt',
    'kitti_sun_train_08.txt',
    'kitti_sun_train_09.txt',
    'kitti_sun_train_10.txt']

exportDirectory = '/media/stars/SunBPN3/'
for dataset in datasets:
    poses = []
    images = []

    with open(directory + dataset) as f:
        # next(f)  # skip the 3 header lines
        # next(f)
        # next(f)
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p0, p1, p2, p3, p4, p5, p6))
            images.append(fname)

    r = range(len(images))
    #random.shuffle(r)
    fileParts = dataset.split(".")
    lmdbFile = exportDirectory + fileParts[0] + "_lmdbTESTDONOTUSE"
    print "Creating PoseNet Sun CNN Dataset: %s." % (fileParts[0] + "_lmdb")
    env = lmdb.open(lmdbFile, map_size=int(1e12))

    count = 0
    start = time.clock()
    print('Processing %d images...' % len(images))
    


    with env.begin(write=True) as txn:
        for i in r:
            if (count + 1) % 500 == 0:
                print 'Saving image: ', count + 1
            im = 0*np.ones([224,224,3], dtype=np.uint8)
            im_orig = cv2.imread(images[i])
            im_resize = cv2.resize(im_orig, (224, 68))           
            im[78:146:,:,:] = im_resize
            X = np.transpose(im, (2, 0, 1))

            im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
            im_dat.float_data.extend(poses[i])
            str_id = '{:0>10d}'.format(count)
            txn.put(str_id, im_dat.SerializeToString())
            count = count + 1
    env.close()
    end = time.clock()
    print 'Done. Elapsed time: %f sec.' % (end - start)
