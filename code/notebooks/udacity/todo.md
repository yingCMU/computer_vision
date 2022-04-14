# Part 1:
L8-18 Network structure
https://classroom.udacity.com/nanodegrees/nd891/parts/ab20cb33-12e4-4681-80b3-6c7bac36ba0f/modules/e5cce3ef-16b8-4b33-9cf9-508238f35b74/lessons/03554ff0-b8c1-4c05-84ce-d4bbcd20630e/concepts/d6576e8f-f14f-43dd-a0bb-0a456dc893f5
# Part 2:
1. Review Fast RCNN, YOLO
1. https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9
1. Lession3 RNN and L4. LSTM modules
1. watch transformer talk: https://www.youtube.com/watch?v=rBCqOTEfxvg
1. read submission review
- project2 followup:
    - learn how resent/imagenet preproessing is done
        - [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
    - this project used composed transform that rescales, randomly crop, normalizes and turns the images into torch Tensors. torchvision.transform, how it works?
        - Why Pytorch officially use mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] to normalize images? [stackoverflow](https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2): Using the mean and std of Imagenet is a common practice. They are calculated based on millions of images. If you want to train from scratch on your own dataset, you can calculate the new mean and std. Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.
            - [Resnet50 image preprocessing](https://stackoverflow.com/questions/56685995/resnet50-image-preprocessing)
        - [TRANSFORMING AND AUGMENTING IMAGES](https://pytorch.org/vision/stable/transforms.html)

Share

    - [MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/stable/models.html)
# All
- read refereced papers and articles in the notes & project.readme
