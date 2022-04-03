- [Udacity Project Github](https://github.com/udacity/P1_Facial_Keypoints/tree/master/data)
- [Kaggle facial Keypoint Detection](https://www.kaggle.com/c/facial-keypoints-detection)
## Project feedback:
Here are some of my suggestion which can help in improving accuracy for keypoint detection:

- You can use [transfer learnin](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) to improve the accuracy further.
- If interested, you can try out some more augmentation techniques. You can find out the official documentation for data transform in pytorch on TORCHVISION.TRANSFORMS.
- You can use some deeper network architecture such as inception network.
- Also the choice of 'adam' as [optimizer](https://pytorch.org/docs/stable/optim.html) is great. Adam uses decay learning
rate implicitly and that makes it a better optimizer for most of the convolutional neural network problems.
    - [PyTorch Loss Functions: The Ultimate Guide](https://neptune.ai/blog/pytorch-loss-functions)
-  It's always good to visualize the training, model and data. You can use tensorboard to visualize these. Here is the link for tensorboard in pytorch: [VISUALIZING MODELS, DATA, AND TRAINING WITH TENSORBOARD](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#visualizing-models-data-and-training-with-tensorboard)
- Hyperparameter selection: there are some algorithms and tools available that can help you out with this
    - [Neural Architecture Search (NAS)- The Future of Deep Learning](https://theaiacademy.blogspot.com/2020/05/neural-architecture-search-nas-future.html)
    - [AWS How Hyperparameter Tuning Works](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)
- Sometimes you might not be sure what the kernel is doing. In that case Just compare the original image with the filtered image and see for yourself how the output varies. Different filters will lead to different effects on filtered images. For instance, some filters detects horizontal edges, some might blur the image, or some invert the image, etc.
- Use a Haar cascade face detector to detect faces in a given image. You may also want to try some alternate face detectors present [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).
