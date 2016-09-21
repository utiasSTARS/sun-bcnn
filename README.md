# Sun-BCNN
Bayesian Convolutional Neural Network to infer Sun Direction from a single RGB image, trained on the KITTI dataset [1].

## Installation & Pre-Requisites

1. Download and compile [Caffe-Sl](https://github.com/wanji/caffe-sl) (we use their L2Norm layer).

2. Ensure that the **lmdb** and **cv2** python packages are installed (e.g. through pip).

3. Clone sun-bcnn:
```
git clone https://github.com/utiasSTARS/sun-bcnn-vo.git
```

## Testing with pre-trained model
1. Visit ftp://128.100.201.179/2016-sun_bcnn to download the pre-trained models from the models folder along with a test LMDB file.

2. Edit *caffe-files/test_sunbcnn.sh* to match appropriate mean file, weights file and testing file.  Edit *scripts/test_sunbcnn.py* with appropriate directories.

3. Run *scripts/test_sunbcnn.sh*:
```
bash scripts/test_sunbcnn.sh
```

## Training
### Using KITTI data
Coming soon...
### Using your own data
You're on your own!


##  Citation
V. Peretroukhin, L. Clement, J. Kelly.
Reducing Drift in Visual Odometry by Inferring Sun Direction using a Bayesian Convolutional Neural Network
Submitted to ICRA 2016.

[arXiv:1609.05993](http://arxiv.org/abs/1609.05993)

![SUN-BCNN](sun-bcnn.png)

##  References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)

[2] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approxi- mation: Representing model uncertainty in deep learning,” in Proceedings of The 33rd International Conference on Machine Learning, 2016, pp. 1050–1059.
