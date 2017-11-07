import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import os.path
import json
import scipy
import argparse
import math
import pylab
import lmdb
import random
import sys
import caffe
from timeit import default_timer as timer


caffe_root = '~/Research/caffe-sl/'  # Change to your directory to caffe-sl
sys.path.insert(0, caffe_root + 'python') # Ensure pycaffe is on the path

from caffe.io import datum_to_array

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--meanfile', type=str, required=True)
args = parser.parse_args()
caffe.set_mode_gpu()

# Load the net
net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

# Extract the image dimensions and the amount of stochastic passes
sample_size = net.blobs['data'].data.shape[0]
sample_w = net.blobs['data'].data.shape[2]
sample_h = net.blobs['data'].data.shape[3]

lmdb_env = lmdb.open(args.dataset)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

# Number of samples in the test dataset
test_samples = lmdb_env.stat()['entries']

count = 0
blob_meanfile = caffe.proto.caffe_pb2.BlobProto()
data_meanfile = open(args.meanfile , 'rb' ).read()
blob_meanfile.ParseFromString(data_meanfile)
meanfile = np.squeeze(np.array( caffe.io.blobproto_to_array(blob_meanfile)))


def computeAzZen(u_est):
    azzen = np.empty(2)
    #azimuth error
    azzen[0] = math.atan2(u_est[0], u_est[2])
    #zenith error
    azzen[1] = math.acos(-u_est[1])

    return azzen


# def meanOfAngles(zenaz_mat):
#     """This is a non-arithmetic angle mean useful for large angle diffs"""
#     zenaz = np.empty(2)

# 	ssinz = np.sum(np.sin(zenaz_mat[0,:]))
# 	scosz = np.sum(np.cos(zenaz_mat[0,:]))
# 	zenaz[0] = math.atan2(ssinz, scosz)

# 	ssina = np.sum(np.sin(zenaz_mat[1,:]))
# 	scosa = np.sum(np.cos(zenaz_mat[1,:]))
# 	zenaz[1] = math.atan2(ssina, scosa)

# 	return zenaz

def convertVecsToAzZen(u_mat):
    if u_mat.ndim > 1:
        numSamples = u_mat.shape[1]
        azzen_mat = np.empty([2, numSamples])
        for i in range(numSamples):
            u_est = u_mat[:,i]/np.linalg.norm(u_mat[:,i])
            azzen_mat[0, i] = math.atan2(u_est[0], u_est[2])
            azzen_mat[1, i] = math.acos(-u_est[1])
        return azzen_mat
    else:
        return computeAzZen(u_mat)


def convertAzZenToVec(azzen):
    az = azzen[0]
    zen = azzen[1]
    v_xz = math.sin(zen)
    y = -math.cos(zen)
    x = v_xz*math.sin(az)
    z = v_xz*math.cos(az)
    return np.array([x,y,z])


def convertToCosineDist(u_mat, u_mat_true):
    numSamples = u_mat.shape[1]

    cosdist_mat = np.empty(numSamples)
    for i in range(numSamples):
        cosdist_mat[i] = 1.0 - u_mat[:,i].dot(u_mat_true[:,i])/np.linalg.norm(u_mat[:,i])

    return cosdist_mat

def convertToVecError(u_mat, u_true):
    if u_mat.ndim > 1:
        numSamples = u_mat.shape[1]
        for i in range(numSamples):
            vecerror_mat[i] = np.arccos((u_mat[:,i].dot(u_true))/(np.linalg.norm(u_mat[:,i])*np.linalg.norm(u_true[:,i])))

        return vecerror_mat
    else:
        vec_error = np.arccos((u_mat.dot(u_true))/(np.linalg.norm(u_mat)*np.linalg.norm(u_true)))
        return vec_error

predicted_vec_hist = np.empty([3, test_samples, 3])
vec_error_hist = np.empty([3, test_samples])
true_vec_hist = np.empty([3, test_samples])
predicted_azzen_hist = np.empty([2, test_samples,3])
azzen_error_hist = np.empty([2, test_samples,3])
covariance_azzen_hist = np.empty([2,2,test_samples,3])

for key, value in lmdb_cursor:
    t0 = timer()
    datum.ParseFromString(value)

    label = np.array(datum.float_data)
    data = caffe.io.datum_to_array(datum)

    #Subtract mean from image
    data = (data-meanfile)


    w = data.shape[1]
    h = data.shape[2]

    input_image = data

    batch = np.repeat([input_image],sample_size,axis=0)
    net.forward_all(data = batch)


    predicted_vec_mat_fc1 = net.blobs['cls1_fc_sundir_norm'].data
    predicted_vec_mat_fc1 = np.squeeze(predicted_vec_mat_fc1).T #3xN

    predicted_vec_mat_fc2 = net.blobs['cls2_fc_sundir_norm'].data
    predicted_vec_mat_fc2 = np.squeeze(predicted_vec_mat_fc2).T #3xN

    predicted_vec_mat_fc3 = net.blobs['cls3_fc_sundir_norm'].data
    predicted_vec_mat_fc3 = np.squeeze(predicted_vec_mat_fc3).T #3xN



    # #Could also use meanOfAngles() for a non arithmetic mean
    # predicted_azzen_fc1 = np.mean(azzen_mat_fc1, axis=1)
    # predicted_azzen_fc2 = np.mean(azzen_mat_fc2, axis=1)
    # predicted_azzen_fc3 = np.mean(azzen_mat_fc3, axis=1)

    # predicted_vec_fc1 = convertAzZenToVec(predicted_azzen_fc1)
    # predicted_vec_fc2 = convertAzZenToVec(predicted_azzen_fc2)
    # predicted_vec_fc3 = convertAzZenToVec(predicted_azzen_fc3)

    predicted_vec_fc1 = np.mean(predicted_vec_mat_fc1, axis=1)
    predicted_vec_fc1 /= np.linalg.norm(predicted_vec_fc1)

    predicted_vec_fc2 = np.mean(predicted_vec_mat_fc2, axis=1)
    predicted_vec_fc2 /= np.linalg.norm(predicted_vec_fc2)

    predicted_vec_fc3 = np.mean(predicted_vec_mat_fc3, axis=1)
    predicted_vec_fc3 /= np.linalg.norm(predicted_vec_fc3)

    predicted_azzen_fc1 = convertVecsToAzZen(predicted_vec_fc1)
    predicted_azzen_fc2 = convertVecsToAzZen(predicted_vec_fc2)
    predicted_azzen_fc3 = convertVecsToAzZen(predicted_vec_fc3)


    true_vec = np.array(label[0:3])/np.linalg.norm(label[0:3])
    true_azzen = convertVecsToAzZen(true_vec)

    azzen_error_fc1 = predicted_azzen_fc1 - true_azzen
    azzen_error_fc2 = predicted_azzen_fc2 - true_azzen
    azzen_error_fc3 = predicted_azzen_fc3 - true_azzen

    azzen_mat_fc1 = convertVecsToAzZen(predicted_vec_mat_fc1)
    azzen_mat_fc2 = convertVecsToAzZen(predicted_vec_mat_fc2)
    azzen_mat_fc3 = convertVecsToAzZen(predicted_vec_mat_fc3)

    covariance_azzen_fc1 = np.cov(azzen_mat_fc1,rowvar=1)
    covariance_azzen_fc2 = np.cov(azzen_mat_fc2,rowvar=1)
    covariance_azzen_fc3 = np.cov(azzen_mat_fc3,rowvar=1)

    vec_error_fc1 = convertToVecError(predicted_vec_fc1, true_vec)
    vec_error_fc2 = convertToVecError(predicted_vec_fc2, true_vec)
    vec_error_fc3 = convertToVecError(predicted_vec_fc3, true_vec)


    predicted_vec_hist[:,count, 0] = predicted_vec_fc1
    predicted_vec_hist[:,count, 1] = predicted_vec_fc2
    predicted_vec_hist[:,count, 2] = predicted_vec_fc3

    true_vec_hist[:,count] = true_vec
    vec_error_hist[:,count] = np.array([vec_error_fc1, vec_error_fc2, vec_error_fc3])

    predicted_azzen_hist[:,count, 0] = predicted_azzen_fc1
    predicted_azzen_hist[:,count, 1] = predicted_azzen_fc2
    predicted_azzen_hist[:,count, 2] = predicted_azzen_fc3


    azzen_error_hist[:,count,0] = azzen_error_fc1
    azzen_error_hist[:,count,1] = azzen_error_fc2
    azzen_error_hist[:,count,2] = azzen_error_fc3

    covariance_azzen_hist[:,:,count,0] = covariance_azzen_fc1
    covariance_azzen_hist[:,:,count,1] = covariance_azzen_fc2
    covariance_azzen_hist[:,:,count,2] = covariance_azzen_fc3

    count += 1

    print ('Image:  %d/%d'% (count,test_samples))
    print ('Vec Error (deg): %.3f / %.3f / %.3f ' % (tuple(np.array([vec_error_fc1, vec_error_fc2, vec_error_fc3])*180/np.pi)))


stats = np.array([ np.median(vec_error_hist[0,:]),
                np.median(vec_error_hist[1,:]),
                np.median(vec_error_hist[2,:]),
                np.mean(vec_error_hist[0,:]),
                np.mean(vec_error_hist[1,:]),
                np.mean(vec_error_hist[2,:])])*180/np.pi

print('Median error angles: %f | %f | %f,  Mean error angles: %f | %f | %f' % tuple(stats))

empirical_azzen_covar = np.empty([2,2,3])
empirical_azzen_covar[:,:,0] = np.cov(predicted_azzen_hist[:,:,0],rowvar=1)
empirical_azzen_covar[:,:,1] = np.cov(predicted_azzen_hist[:,:,1],rowvar=1)
empirical_azzen_covar[:,:,2] = np.cov(predicted_azzen_hist[:,:,2],rowvar=1)

weightsFileName = args.weights.split("/")[-1]
weightsFilePrefix = weightsFileName.split(".")[0]

datasetName = args.dataset.split("/")[-1]
fileName = weightsFilePrefix + "_" + datasetName +"_test_stats.mat"


print('Saving file: %s.'%fileName)
io.savemat(fileName, mdict={'predicted_vec_hist': predicted_vec_hist,
                            'true_vec_hist': true_vec_hist,
                            'vec_error_hist': vec_error_hist,
                            'predicted_azzen_hist': predicted_azzen_hist,
                            'covariance_azzen_hist': covariance_azzen_hist,
                            'azzen_error_hist': azzen_error_hist,
                            'empirical_azzen_covar': empirical_azzen_covar
                            })
print('Done.')
