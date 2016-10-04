# Sun-BCNN
Bayesian Convolutional Neural Network to infer Sun Direction from a single RGB image, trained on the KITTI dataset [1].

![](http://i.giphy.com/qB4pb5T94axLa.gif)

## Installation & Pre-Requisites

1. Download and compile [Caffe-Sl](https://github.com/wanji/caffe-sl) (we use their L2Norm layer).

2. Ensure that the **lmdb** and **cv2** python packages are installed (e.g. through pip).

3. Clone sun-bcnn:
```
git clone https://github.com/utiasSTARS/sun-bcnn-vo.git
```

## Testing with pre-trained model
1. Visit ftp://128.100.201.179/2016-sun_bcnn and download a pre-trained model, test LMDB file and appropriate mean file.

2. Edit *caffe-files/test_sunbcnn.sh* to match appropriate mean file, weights file and testing file.  Edit *scripts/test_sunbcnn.py* with appropriate directories.

3. Run *scripts/test_sunbcnn.sh*:
```
bash scripts/test_sunbcnn.sh
```

## Training
### Using KITTI data
1. Visit ftp://128.100.201.179/2016-sun_bcnn and download the training LMDB file. Visit http://vision.princeton.edu/pvt/GoogLeNet/Places/ and download the pre-trained GoogLeNet from Princeton (trained on MIT Places data).

2. Edit *caffe-files/train_sunbcnn.prototxt* with the appropriate file names (search 'CHANGEME')

3. Edit *caffe-files/train_sunbcnn.sh* with the appropriate folder and file names.

4. Run *scripts/train_sunbcnn.sh*:
```
bash scripts/train_sunbcnn.sh
```

Note: the LMDB files contain images that have been re-sized and padded with zeros along with target Sun directions (extracted through ephemeris tables and the ground truth provided by KITTI GPS/INS). A human readable table of image filenames and Sun directions can be found in the kitti-groundtruth-data folder (consult our paper for camera frame orientation).

### Using your own data
See *scripts/create_lmdb_sunbcnn_dataset.py* for a wireframe of how to create your own training LMDB files.

##  Citation
V. Peretroukhin, L. Clement, J. Kelly.
*Reducing Drift in Visual Odometry by Inferring Sun Direction using a Bayesian Convolutional Neural Network*

Submitted to ICRA 2016. Pre-print available: [arXiv:1609.05993](http://arxiv.org/abs/1609.05993).

![SUN-BCNN](sun-bcnn.png)

##  References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)

[2] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in Proceedings of The 33rd International Conference on Machine Learning, 2016, pp. 1050–1059.

[3] A. Kendall, and R. Cipolla, "Modelling Uncertainty in Deep Learning for Camera Relocalization." The International Conference on Robotics and Automation, 2015.
