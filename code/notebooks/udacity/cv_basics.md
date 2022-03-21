# Feature Vectors (Feature extraction techniques)
Convolutional filters and ORB and HOG descriptors all rely on patterns of intensity to identify different shapes (like edges) and eventually whole objects (with feature vectors).

## ORB
- oriented fast and
- roated
- brief

## Keypoint
keypoint: small region in an image that is particularly distinctive. e.g corners where the pixel values sharply change from light to dark

ORB generated keypoints
![Tux, the Linux mascot](./images/orb_kp.png)

once keypoints in an image have been located, ORB calculate correponding feature vector for each keypoint, only containing binary values.
The sequence of ones and zeros vary accordig to what a specific keypoint and its surrounding pixel area look like. The vector represnets the pattern of intensity around a key point.
So multiple feature vectors can be used to identiy a large area, and even a specific object in an image.
### FAST (Features from accelerated segments Test)
find keypoint in an image: [youtube](https://www.youtube.com/watch?v=DCHAc6fjcVM)
### BRIEF (Binary robost independent elementary features)
creat feature vectors from keypoint: [youtube](https://www.youtube.com/watch?v=EKIPEPpRciw)
- to handle object of diffrent scale and object orientation: [youtube](https://www.youtube.com/watch?v=2k3T6rfjvx0)
### Feature Matching
https://www.youtube.com/watch?v=RH05Wnl1-2A
- [limitation](https://www.youtube.com/watch?v=Vzs6B1dFQC0)

## HOG (Histograms of Oriented Gradients )
[youtube](https://www.youtube.com/watch?v=dqe9zGtxoNM)

# CNN
## CNN Layers
[youtube](https://www.youtube.com/watch?v=LX-yVob3c28)
## Pooling and global average pooling layers
for recuding number of parameters used, and reducing dimension of each plane. Too many parameters can cause overfitting
[youtube](https://www.youtube.com/watch?v=OkkIZNs7Cyc)
![Tux, the Linux mascot](./images/vgg-16.png)
## Dropout Layers
https://pytorch.org/docs/stable/nn.html#dropout-layers
Dropout layers essentially turn off certain nodes in a layer with some probability, p. This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.
