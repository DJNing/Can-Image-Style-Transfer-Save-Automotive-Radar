# Can-Image-Style-Transfer-Save-Automotive-Radar

## Tested Models

### Image Style Transfer Using Convolutional Neural Networks

**Cited from: https://github.com/ali-gtw/ImageStyleTransfer-CNN**

This is a PyTorch implementation of [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html), inspired by [authors of paper repo](https://github.com/leongatys/PytorchNeuralStyleTransfer).


Coarse-to-fine high-resolution is also added, from paper [Controlling Perceptual Factors in Neural Style Transfer.](https://arxiv.org/abs/1611.07865)


#### Usage
Simply run `python3 main.py`. 

You may want to change `DATA.STYLE_IMG_PATH`, `DATA.CONTENT_IMG_PATH` in config file, for transferring 
style of your desired style image to your content image.


### Pix2PixHD

**Cited from [https://github.com/NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)**

A few changes were made to dataloader and configs for this experiments.

#### Usage

Change the options in ```./p2pHD/options``` to cope with your task. Please refer to the cited repo for more training/testing details.

### CycleGAN

implementation based on [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

#### Usage

```python ./CycleGAN/train.py --options```

## Docker env

docker file is provided in ```./docker```, which provides env to run all the tested models in this repo. For further information regrading docker installation, please refer to [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Demo video

Please click [here](https://drive.google.com/file/d/1pFEmGHi-RSa7AhYXotBmfA1eeZyb2jB9/view?usp=sharing) to watch the demo video for this project. You can also watch it on YouTube [here](https://www.youtube.com/watch?v=2Ziwj8B2X1U)
