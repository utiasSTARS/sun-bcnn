#Pertinent directories
caffe_root = '~/caffe-sl/' #Caffe-Sl directory
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import lmdb
import caffe
import random
import cv2
import time
import math
import pylab as plt
import random
import scipy.io as io

#Dataset groundtruth files to be compiled into single LMDB files
datasetFilenames = [
    'kitti_sun_test_00.csv',
    'kitti_sun_test_01.csv',
    'kitti_sun_test_02.csv',
    'kitti_sun_test_04.csv',
    'kitti_sun_test_05.csv',
    'kitti_sun_test_06.csv',
    'kitti_sun_test_07.csv',
    'kitti_sun_test_08.csv',
    'kitti_sun_test_09.csv',
    'kitti_sun_test_10.csv',
    'kitti_sun_train_00.csv',
    'kitti_sun_train_01.csv',
    'kitti_sun_train_02.csv',
    'kitti_sun_train_04.csv',
    'kitti_sun_train_05.csv',
    'kitti_sun_train_06.csv',
    'kitti_sun_train_07.csv',
    'kitti_sun_train_08.csv',
    'kitti_sun_train_09.csv',
    'kitti_sun_train_10.csv'
]
datasetFilenamesDir = './' 
kittiDataDir = '/media/raid5-array/datasets/KITTI/raw/' #High level KITTI directory
exportDirectory = './' #Where you want your lmdb files to end up in

kittiImageSize = [1241, 376]

preserveAspectRatio = False #If true, 0s will be padded at the top and bottom of the image to make the final igure 224x224
centreCrop = False #If true, this will centre crop the KITTI image to 376x376
azZenTarget = False #If true, target will be azimuth/zenith instead of unit vectors




#Read the ground truth
def readGroundTruth(datasetTxtFilepath):
    sunDirList = []
    imageFileNames = []
    with open(datasetTxtFilepath) as f:
        for line in f:
            lineItems = line.split(',')
            fname = lineItems[0]

            sunDir = lineItems[1:4]
            sunDir = [float(i) for i in sunDir]

            if azZenTarget:
            	sunAzZen = [0, 0]
            	sunAzZen[0] = math.degrees(math.atan2(sunDir[0], sunDir[2]))
            	sunAzZen[1] = math.degrees(math.acos(-sunDir[1]))
            	sunDirList.append(sunAzZen)
            else:
                sunDirList.append(sunDir)
            imageFileNames.append(fname)

    return sunDirList, imageFileNames



def createLMDB(lmdbFileName, kittiDataDir, imageFileNames, sunDirList, shuffle):
    env = lmdb.open(lmdbFileName, map_size=int(1e12))
    count = 0

    with env.begin(write=True) as txn:
        ids = list(range(len(imageFileNames)))
        if shuffle:
            random.shuffle(ids)

        for i in ids:
            if (count + 1) % 500 == 0:
                print('Saving image: %d (ID: %d)' % (count + 1, i))

            sunDir = sunDirList[int(i)]
            fileName_l = kittiDataDir + imageFileNames[int(i)]
            
            #Read in and ensure BGR
            im_orig_l = cv2.imread(fileName_l)

            #Resize
            if preserveAspectRatio:
                im_final = np.ones([224,224,3], dtype=np.uint8)
                im_resize = cv2.resize(im_orig_l, (224, 68))
                im_final[78:146:,:,:] = im_resize
            elif centreCrop:
                im_crop = im_orig_l[:,434:434+376,:] #Centre crop to 376x376
                im_final = cv2.resize(im_crop, (224, 224))
            else:
                im_final = cv2.resize(im_orig_l, (224, 224))



            #Put the channels first
            X = np.transpose(im_final, (2, 0, 1))

            #Convert to caffe LMDB
            im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
            im_dat.float_data.extend(sunDir)
            str_id = '{:0>10d}'.format(count)
            txn.encode(str_id, im_dat.SerializeToString())
            count = count + 1

    env.close()
    if shuffle:
        io.savemat(lmdbFileName + '_indices.mat', mdict={'shuffled_ids': ids})



for dat_i in range(len(datasetFilenames)):
    #Export train and test data
    fileNameParts = datasetFilenames[dat_i].split(".")
    lmdbFileName = exportDirectory + fileNameParts[0] + "_lmdb"
    sunDirList, imageFileNames = readGroundTruth(datasetFilenamesDir + datasetFilenames[dat_i])

    print("Creating KITTI Sun-BCNN Dataset: %s" % (lmdbFileName))
    print("Total images to process: %d" % (len(imageFileNames)))

    fileNameParts = datasetFilenames[dat_i].split("_")
    trialType = fileNameParts[-2]
    if trialType == 'test':
        shuffle = False
        print('Detected TEST file. Not shuffling images.')
    else:
        shuffle = True
        print('Detected TRAIN file. Shuffling images.')

    start = time.clock()
    createLMDB(lmdbFileName, kittiDataDir, imageFileNames, sunDirList, shuffle)
    end = time.clock()
    print("Done. Elapsed time: %f sec." % (end - start))
