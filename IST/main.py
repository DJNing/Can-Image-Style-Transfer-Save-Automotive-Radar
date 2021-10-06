import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import build_model
from model.engine import do_transfer_style
from config import get_cfg_defaults
from model.engine.hr_transfer_style import do_hr_transfer_style
from model.meta_arch import GramMSELoss, StyleTransfer
from util.logger import setup_logger
from util.prepare_vgg import prepare_vgg_weights
import numpy as np
import torch.nn as nn
import time
import torchvision.transforms.functional as TF
import glob
import cv2
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

def get_model(cfg):
    # build vgg_model
    vgg_model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    vgg_model.to(device)
    # load model weights
    prepare_vgg_weights(cfg)
    vgg_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    for param in vgg_model.parameters():
        param.requires_grad = False

    # define layers, loss functions
    loss_layers = cfg.LOSS.STYLE_LAYERS + cfg.LOSS.CONTENT_LAYERS
    loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS) + \
                     [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
    loss_functions = [loss_function.to(device) for loss_function in loss_functions]

    # loss weights settings
    loss_weights = cfg.LOSS.STYLE_WEIGHTS + cfg.LOSS.CONTENT_WEIGHTS

    model = StyleTransfer(vgg_model, loss_layers, loss_functions, loss_weights)
    return model, device


def transfer_style(cfg, high_resolution=False):
    # build model
    model, device = get_model(cfg)

    # load images
    content_image = Image.open(cfg.DATA.CONTENT_IMG_PATH)
    style_image = Image.open(cfg.DATA.STYLE_IMG_PATH)

    # content_image.save(cfg.OUTPUT.DIR + "content.png")
    # style_image.save(cfg.OUTPUT.DIR + "style.png")

    # start transferring the style
    out_image = do_transfer_style(
        cfg,
        model,
        content_image,
        style_image,
        device
    )

    if high_resolution:
        do_hr_transfer_style(
            cfg,
            model,
            content_image,
            style_image,
            out_image,
            device
        )

def pil_polar_transform(pil_img, reverse=False):

    
    # pil_img = pil_img.convert('RGB')
    np_image = np.array(pil_img)
    # np_image[np_image > 0] = 255
    value = np.sqrt(((np_image.shape[0]/2.0)**2.0)+((np_image.shape[1]/2.0)**2.0))
    if reverse:
        # flag = cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP
        polar = cv2.linearPolar(np_image, (255,255), value, cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS)
        # return Image.fromarray(np_image)
    else:
        # flag = cv2.WARP_FILL_OUTLIERS
        polar = cv2.linearPolar(np_image, (255,255), value, cv2.WARP_FILL_OUTLIERS)
    
    # polar = cv2.linearPolar(test, (255,255), value, cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS)
    
    # polar_image[polar_image>0] = 255
    
    result = Image.fromarray(polar)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Style Transfer Using Convolutional Neural Networks.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_root = "/home/dj/Downloads/lidar/shanghai/sq_img/filtered_ist/"
    save_dir = save_root

    # save_dir = os.path.join(save_root, ref)
    # if is_polar:
    #     save_dir = save_dir + '_polar'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # setup the logger
    # if not os.path.isdir(cfg.OUTPUT.DIR):
    #     os.mkdir(cfg.OUTPUT.DIR)
    logger = setup_logger("style-transfer", save_dir, 'log')
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    # Transfer style to content
    # transfer_style(cfg)
    model, device = get_model(cfg)

    # content_dir = "/workspace/data/save_new/radar/*.png"
    # style_dir = "/workspace/data/save_new/lidar/"
    # content_dir = "/home/dj/Downloads/lidar/nuscene/save_new/data/all_img_40/radar/"
    content_dir = "/home/dj/Downloads/lidar/shanghai/sq_img/filtered_img/"
    style_dir = "/home/dj/Downloads/lidar/nuscene/save_new/lidar/"

    content_list = sorted(glob.glob(content_dir+'*.png'))
    # style_list = sorted(glob.glob(style_dir+'*.jpg'))

    # test = int(len(content_list)*0.9)

    # content_list = content_list[test:]
    # style_list = style_list[test:]

    # content_list = content_list[-100:]
    cnt = 0

    # polar = cfg.DATA.POLAR


    # content_list = os.listdir(content_dir)
    # style_list = os.listdir(style_dir)
    # rotate = transforms.RandomRotation([0,180])
    # toImage = transforms.ToPILImage()
    # load images
    

    is_polar = False

    current_list = os.listdir(save_dir)
    cnt = 0
    # print('cnt = ', str(cnt))
    # print('total_len = ', str(len(content_list)))
    start = time.time()
    # test_list = ["02243.png", "03298.png", "03663.png", "04287.png","04357.png","05805.png","07957.png","08775.png","08823.png","08901.png","03335.png"]

    # test_list = content_list

    rotate = transforms.functional.rotate
    toImage = transforms.ToPILImage()
    toTensor = transforms.ToTensor()

    # style list: 09567.png, 01601.png, 09364.png, 09033.png


    style_image = Image.open(os.path.join(style_dir, '09033.png'))
    style_image = style_image.convert('RGB')
    for i in range(len(content_list)):
        # idx = np.random.randint(len(content_list))

        # content_image = Image.open(os.path.join(content_dir, content_list[i]))
        fname = content_list[i].split('/')[-1]
        ts = fname.split('.')[0]
        print(i)
        # save_dir = os.path.join(save_root, ref)
        # if is_polar:
        #     save_dir = save_dir + '_polar'
        # if not os.path.isdir(save_dir):
        #     os.mkdir(save_dir)

        # style_image = Image.open(os.path.join(style_dir, style_list[i]))
        content_image = Image.open(content_list[i])
        # style_image = Image.open(style_list[i])
        # print(i)
        # break
        start = time.time()
        # content_image = Image.open(test_list[i])
        content_image = content_image.convert('RGB')
        # content_image = toTensor(content_image)
        # content_image = rotate(content_image, 35)
        # content_image = toImage(content_image)
        # style_image = Image.open(test_list[i])
        # style_image = Image.open(ref_folder + ref + ".png")
        
        # content_image.save(os.path.join(save_dir, 'content_' + str(i) + '.png'))
        # style_image = toTensor(style_image)
        # style_image = rotate(style_image, 75)
        # style_image = toImage(style_image)

        # style_image.save(os.path.join(save_dir, 'style_' + str(i) + '.png'))

        if is_polar:
            content_image = pil_polar_transform(content_image)
            content_image.save(os.path.join(save_dir, 'content_polar_' + str(i) + '.png'))
            style_image = pil_polar_transform(style_image)
            style_image.save(os.path.join(save_dir, 'style_polar_' + str(i) + '.png'))

        out_image = do_transfer_style(
            cfg,
            model,
            content_image,
            style_image,
            device
        )

        if is_polar:
            out_image.save(os.path.join(save_dir, ts + '_polar' + '.png'))    
            out_image = pil_polar_transform(out_image, reverse=True)
        
        out_image.save(os.path.join(save_dir, ts + '.png'))
        
        # out_array = np.array(out_image)
        cnt += 1
        # if cnt > 30:
        #     break
        end_time = time.time()
        print("transferring images at %d out of %d, second per frame: %f" % (cnt, len(content_list), end_time-start), end=' ')


    print("avg time per frame: ", str((time.time()-start)/len(content_list)))

if __name__ == "__main__":
    main()
