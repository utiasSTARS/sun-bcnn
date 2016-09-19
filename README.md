# Sun-BCNN
Bayesian Convolutional Neural Network to infer Sun Direction

## Installation & Pre-Requisites

1. Download and compile [Caffe-Sl](https://github.com/wanji/caffe-sl) (we use their L2Norm layer).

2. Ensure that the **lmdb** and **cv2** python packages are installed (e.g. through pip).

3. Clone sun-bcnn:
```
git clone https://github.com/utiasSTARS/sun-bcnn-vo.git
```

## Testing with pre-trained model
1. Visit ftp://128.100.201.179/2016-sun_bcnn to download the pre-trained models from the models folder along with a test LMDB file.

2. Edit **caffe-files/test_sunbcnn.sh** to match appropriate mean file, weights file and testing file.  Edit **scripts/test_sunbcnn.py** with appropriate directories.

3. Run **scripts/test_sunbcnn.sh**:
```
bash scripts/test_sunbcnn.sh
```


## Testing with pre-trained model
Coming soon...

##  Citation
Submitted to ICRA 2016. arXiv publication coming soon...

![SUN-BCNN](sun-bcnn.png)

##  References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
