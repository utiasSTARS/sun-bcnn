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

caffe_root = '~/caffe-sl/'  # Change to your directory to caffe-sl
sys.path.insert(0, caffe_root + 'python') # Ensure pycaffe is on the path

from caffe.io import datum_to_array

estimation_type = 0 #1 or 0. If 1, the covariances will be estimated by propagating the covariance of unit norm vectors to a 2x2 azimuth, zenith covariances
                    # If 0, all unit norm vectors will be first converted to azimuth, zenith angles, then averaged and covarianced.


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

#Compute the jacobian of cosdist(u_est,v_true) wrt [x,y,z] of u_est
def computeCosDistJacob(u_est, v_true):
    G = np.empty([1,3])
    nu = np.linalg.norm(u_est)
    nv = np.linalg.norm(v_true)
    a = np.dot(u_est,v_true)/(nu*nv)

    invsqr = -1.0/math.sqrt(1-math.pow(a,2))
    df_dx = invsqr*(v_true[0]*(nu*nv) - u_est[0]*(nv/nu)*np.dot(u_est,v_true))/(math.pow(nu*nv,2))
    df_dy = invsqr*(v_true[1]*(nu*nv) - u_est[1]*(nv/nu)*np.dot(u_est,v_true))/(math.pow(nu*nv,2))
    df_dz = invsqr*(v_true[2]*(nu*nv) - u_est[2]*(nv/nu)*np.dot(u_est,v_true))/(math.pow(nu*nv,2))

    G[0,0] = df_dx
    G[0,1] = df_dy
    G[0,2] = df_dz

    return G

#Compute the jacobian of zenaz(u_est,v_true) wrt [x,y,z] of u_est
def computeZenAzJacob(u_est, v_true):
    G = np.empty([2,3])

    dthz_dx = 0
    dthz_dy = 1.0/math.sqrt(1-math.pow(u_est[1],2))
    dthz_dz = 0

    dtha_dx = (1.0/u_est[2])*(1/(1+math.pow(u_est[0]/u_est[2],2)))
    dtha_dy = 0
    dtha_dz = (-1*u_est[0]/math.pow(u_est[2],2))*(1/(1+math.pow(u_est[0]/u_est[2],2)))

    G[0,0] = dthz_dx
    G[0,1] = dthz_dy
    G[0,2] = dthz_dz

    G[1,0] = dtha_dx
    G[1,1] = dtha_dy
    G[1,2] = dtha_dz

    return G

def computeZenAz(u_est):
    zenaz = np.empty(2)
    #zenith error
    zenaz[0] = math.acos(-u_est[1])
    #azimuth error
    zenaz[1] = math.atan2(u_est[0], u_est[2])
    return zenaz

def convertToZenAz(u_mat):
    numSamples = u_mat.shape[1]
    zenaz_mat = np.empty([2, numSamples])
    for i in range(numSamples):
        u_est = u_mat[:,i]
        #zenith error
        zenaz_mat[0,i] = math.acos(-u_est[1])
        #azimuth error
        zenaz_mat[1, i] = math.atan2(u_est[0], u_est[2])
    return zenaz_mat


def convertZenAzToVec(zen, az):
    y = -math.cos(zen)
    x = math.sin(az)*math.cos(zen)
    z = math.sqrt(1 - x*x - y*y)
    return np.array([x,y,z])


def convertToCosineDist(u_mat, u_mat_true):
    numSamples = u_mat.shape[1]

    cosdist_mat = np.empty(numSamples)
    for i in range(numSamples):
        cosdist_mat[i] = 1.0 - u_mat[:,i].dot(u_mat_true[:,i])/np.linalg.norm(u_mat[:,i])
    return cosdist_mat


predicted_vec_hist = np.empty([3, test_samples])
variance_angle_hist = np.empty([1, test_samples])
error_angle_hist = np.empty([1, test_samples])
true_vec_hist = np.empty([3, test_samples])
zenaz_hist = np.empty([2, test_samples])
zenaz_error_hist = np.empty([2, test_samples])
covariance_zenaz_hist = np.empty([2,2,test_samples])

for key, value in lmdb_cursor:

    datum.ParseFromString(value)

    label = np.array(datum.float_data)
    data = caffe.io.datum_to_array(datum)

    #Subtract mean from image
    data = data-meanfile

    w = data.shape[1]
    h = data.shape[2]

    input_image = data
    batch = np.repeat([input_image],sample_size,axis=0)
    net.forward_all(data = batch)

    #Extract the predicted direction vectors (where N=sample_size)
    predicted_vec_mat = net.blobs['cls3_fc_xyz_norm'].data
    predicted_vec_mat = np.squeeze(predicted_vec_mat).T #3xN

    true_vec = label[0:3]/np.linalg.norm(label[0:3])
    true_vec_mat = np.repeat([true_vec],sample_size, axis=0)
    true_vec_mat = true_vec_mat.T #3xN

    #Rotate to avoid signularities at zen = 0
    rotMat = np.array([[1.,0.,0.], [0.,0.,1.], [0.,-1.,0.]])


    #Option 1: Convert to az,zen first, then covariance
    if estimation_type == 1:
        zenaz_mat = convertToZenAz(rotMat.dot(predicted_vec_mat))
        zenaz = np.mean(zenaz_mat, axis=1)

        cosinedist_mat = convertToCosineDist(predicted_vec_mat, true_vec_mat)
        error_angle_mat = np.arccos(1.0 - cosinedist_mat)*180/math.pi

        zenaz_mat_true = convertToZenAz(rotMat.dot(true_vec_mat))
        cosinedist_mean = np.mean(cosinedist_mat)
        error_angle = np.mean(error_angle_mat)
        zenaz_error = np.mean(zenaz_mat-zenaz_mat_true, axis=1)* 180/math.pi

        #Variance
        covariance_zenaz = math.pow(180/math.pi, 2)*np.cov(zenaz_mat,rowvar=1)
        variance_angle = np.var(error_angle_mat)

        predicted_vec = convertZenAzToVec(zenaz[0], zenaz[1])
        predicted_vec = rotMat.T.dot(predicted_vec)

        zenaz = zenaz*180/math.pi

    #Option 0: Stay in unit norms, and propagate covariance through Jacobians
    else:
        predicted_vec_mean = np.mean(predicted_vec_mat, axis=0)
        predicted_vec = predicted_vec_mean / np.linalg.norm(predicted_vec_mean)

        error_angle = math.acos(np.dot(true_vec,predicted_vec)/(np.linalg.norm(true_vec)*np.linalg.norm(predicted_vec)))
        error_angle = error_angle * 180/math.pi

        #Rotate to avoid signularities at zen = 0
        rotMat = np.array([[1.,0.,0.], [0.,0.,1.], [0.,-1.,0.]])

        zenaz = computeZenAz(rotMat.dot(predicted_vec)) * 180/math.pi
        zenaz_error = zenaz - computeZenAz(rotMat.dot(true_vec)) * 180/math.pi

        #Variance
        covariance_vec = np.cov(predicted_vec_mat,rowvar=0)

        #Jacobian into zenith and azimuth angles
        G_zenaz = computeZenAzJacob(rotMat.dot(predicted_vec), rotMat.dot(true_vec))
        covariance_zenaz = math.pow(180/math.pi, 2)*np.dot(np.dot(G_zenaz, covariance_vec), G_zenaz.T)


        #Jacobian into error (cosine distance) angle
        G_cos = computeCosDistJacob(predicted_vec, true_vec)
        variance_angle = math.pow(180/math.pi, 2)*np.dot(np.dot(G_cos, covariance_vec), G_cos.T)



    predicted_vec_hist[:,count] = predicted_vec
    true_vec_hist[:,count] = true_vec
    variance_angle_hist[0,count] = variance_angle
    error_angle_hist[0,count] = error_angle
    zenaz_hist[:,count] = zenaz
    zenaz_error_hist[:,count] = zenaz_error
    covariance_zenaz_hist[:,:,count] = covariance_zenaz
    count += 1

    print 'Iteration:  ', count
    print 'Vector Error (deg):  ', error_angle
    print ('Zenith error (deg): %f, Azimuth error (deg): %f' % (zenaz_error[0],zenaz_error[1]))
    print ('Zenith (deg): %f, Azimuth (deg): %f' % (zenaz[0],zenaz[1]))
    if zenaz[0] < 5:
        print 'WARNING!! Zenith angle less than 5 degrees!'
    print 'Uncertainty (deg):   ', math.sqrt(variance_angle)


median_err = np.median(error_angle_hist)
median_std = np.median(np.sqrt(variance_angle_hist))

print('Median error angle: %f | Median error sigma: %f'%(median_err,median_std))

weightsFileName = args.weights.split("/")[-1]
weightsFilePrefix = weightsFileName.split(".")[0]
fileName = weightsFilePrefix + "_testStats_zenazmean.mat"
print('Saving file: %s.'%fileName)
io.savemat(fileName, mdict={'predicted_vec_hist': predicted_vec_hist,
                            'true_vec_hist': true_vec_hist,
                            'error_angle_hist': error_angle_hist,
                            'variance_angle_hist': variance_angle_hist,
                            'zenaz_hist': zenaz_hist,
                            'covariance_zenaz_hist': covariance_zenaz_hist,
                            'zenaz_error_hist': zenaz_error_hist
                            })
print('Done.')
