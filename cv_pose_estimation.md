## Pose estimation from multi-view input images
- Early attempts tackled pose-estimation from multiview
inputs by optimizing simple parametric models of the
human body to match hand-crafted image features in each
view, achieving limited success outside of the controlled
settings.
    - [Markerless motion capture of interacting characters using multi-view image segmentation](https://ieeexplore.ieee.org/document/5995424)
    - [Optimization and filtering for human motion
capture](https://www.tnt.uni-hannover.de/papers/data/611/611_1.pdf)
    - [3d pictorial structures for multiple view articulated pose estimation](https://ieeexplore.ieee.org/document/6619308)
    - [3d pictorial structures for multiple human pose estimation.](https://ieeexplore.ieee.org/document/6909612)
- With the advent of deep learning, the dominant
paradigm has shifted towards estimating 2D poses from
each view separately, through exploiting efficient monocular
pose estimation architectures, and then
recovering the 3D pose from single view detections.
    - [Stacked hourglass networks for human pose estimation](https://arxiv.org/abs/1603.06937)
    - [Efficient object localization using
convolutional networks](https://arxiv.org/abs/1411.4280)
    - [Convolutional pose machines](https://arxiv.org/pdf/1602.00134.pdf)
    - [Deep high-resolution representation learning for human pose estimation](https://arxiv.org/pdf/1902.09212.pdf)

- Few models have focused on developing lightweight
solutions to reason about multi-view inputs
    - [A generalizable approach for multi-view 3d human pose regression](https://arxiv.org/abs/1804.10462) proposes to concatenate together pre-computed 2D
detections and pass them as input to a fully connected
network to predict global 3D joint coordinates
    - [Cross view fusion for 3d human pose estimation](https://arxiv.org/abs/1909.01203) refines 2D heatmap detections jointly by using a fully
connected layer before aggregating them on 3D volumes.
- these methods
fuse information from different views without using
volumetric grids, they do not leverage camera information
and thus overfit to a specific camera setting. We will show
that our approach can handle different cameras flexibly and
even generalize to unseen ones.
### Multi-view and competing techniques
Multi-view pose estimation is arguably the best way to obtain ground truth
for monocular 3D pose estimation [5, 22] in-the-wild

- [5] [Panoptic studio: A massively
multiview system for social interaction capture]()
- [22] [Humbi 1.0: Human multiview behavioral imaging dataset]()

Competing techniques have certain limitations such as inability to capture rich pose
representations (e.g. to estimate hands pose and face pose
alongside limb pose) as well as various clothing limitations.:
- marker-based motion capture: [A survey of
advances in vision-based human motion capture and analysis.
Computer vision and image understanding]()
- visual-inertial methods: [Human
pose estimation from video and imus.]()

The downside is, previous works that used multi-view triangulation for constructing datasets relied on excessive, almost impractical number of views to get the 3D ground truth of sufficient quality [5, 22]. This makes the collection of
new in-the-wild datasets for 3D pose estimation very challenging and calls for the reduction of the number of views needed for accurate triangulation.

improving the accuracy
of multi-view pose estimation from few views is an
important challenge with direct practical applications.

Studies of multiview 3D human pose estimation are generally aimed at getting
the ground-truth annotations for the monocular 3D human
pose estimation: [Learning Monocular 3D Human Pose Estimation from Multi-view Images](https://arxiv.org/pdf/1803.04775.pdf)

The work [A generalizable approach for multi-view 3D human pose regression](https://arxiv.org/abs/1804.10462) proposed concatenating joints’ 2D coordinates from all views into a single
batch as an input to a fully connected network that is
trained to predict the global 3D joint coordinates. This approach
can efficiently use the information from different
views and can be trained on motion capture data. However,
the method is by design unable to transfer the trained
models to new camera setups, while the authors show that
the approach is prone to strong over-fitting.

### Monocular(Single view) 3D pose estimation
1. using high quality 2D pose estimation engines with subsequent
separate lifting of the 2D coordinates to 3D via deep neural
networks (either fully-connected, convolutional or recurrent): [A simple
yet effective baseline for 3d human pose estimation](https://arxiv.org/abs/1705.03098). It offers several advantages:
it is simple, fast, can be trained on motion capture
data (with skeleton/view augmentations) and allows switching
2D backbones after training. Despite known ambiguities
inherent to this family of methods (i.e. orientation of
arms’ joints in current skeleton models), this paradigm is
adopted in the current multi-frame state of the art [3d
human pose estimation in video with temporal convolutions
and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf)
2. infer the 3D coordinates directly
from the images using convolutional neural networks. The
present best solutions use volumetric representations of the
pose, with current single-frame state-of-the-art results on
Human3.6M, namely [Integral human pose regression](https://arxiv.org/pdf/1711.08229.pdf).
### the idea of learnable triangulation
We propose and investigate two simple
and related methods for multi-view human pose estimation. Behind both of them lies the idea of learnable trangulation, which allows us to dramatically reduce the number of views needed for accurate estimation of 3D pose. During learning, we either use marker based motion capture ground truth or “meta”-ground truth obtained from the excessive number of views

The methods themselves are as follows:
- (1) a simpler approach based on algebraic triangulation with learnable camera-joint confidence weights, and
- (2) a more complex volumetric triangulation approach based on dense geometric aggregation of information from different views that allows modelling a human pose prior.<br> Crucially, both of the proposed solutions are fully differentiable, which permits
end-to-end training.
### Triangulating 2D detections
Computing the position of a point in 3D-space given its images in n views
and the camera matrices of those views is one of the most studied computer vision problems.
- refer to book [Multiple view geometry in computer vision](https://drive.google.com/file/d/1IcfX9ODlmKlHOxta8de2tsLwfNT3x7tR/view?usp=sharing)


In our work,
we use the Direct Linear Triangulation (DLT) method
because it is simple and differentiable. We propose a novel
GPU-friendly implementation of this method, which is up
to two orders of magnitude faster than existing ones that are
based on SVD factorization. We provide a more detailed
overview about this algorithm in Section 7.2
- Several methods lift 2D detections efficiently to 3D by
means of triangulation
    - [Multi-view pictorial structures for 3d human
pose estimation.]
    - [3d pose detection of closely interactive humans using multiview
cameras]
    - [Deepfly3d: A deep
learning-based approach for 3d limb and appendage tracking
in tethered, adult drosophila]
    - [Realtime multi-person 2d pose estimation using part affinity
fields]
- More closely related to
our work, [Learnable triangulation of human pose](https://arxiv.org/abs/1905.05754) proposes to back-propagate through an SVDbased
differentiable triangulation layer by lifting 2D detections
to 3D keypoints. Unlike our approach, these methods
do not perform any explicit reasoning about multi-view inputs
and therefore struggle with large self-occlusions.
# Concepts
1. camera projection matri?
2. absolute world coordinates

# Evaluation
- [Simple baselines for human pose estimation and tracking](https://arxiv.org/pdf/1804.06208.pdf)

# Open Techniques
- [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html)

# Dataset
- Human3.6M: [Human3.6m:
Large scale datasets and predictive methods for 3d
human sensing in natural environments]
- CMU Panoptic : [Panoptic studio: A massively
multiview system for social interaction capture]
