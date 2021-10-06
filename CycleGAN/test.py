#!/usr/bin/python3

import argparse
import sys
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np

from models import *
from datasets import ImageDataset
from PIL import Image, ImageFont, ImageDraw


def image_stitching(ip, t):
    # ip: list of tensor
    # shape in N x C x H x W
    batch = len(ip)
    size = ip[0].shape
    # batch = int(size[0])
    height = int(size[2])
    width = int(size[3])

    result = Image.new('RGB', (batch * (height+5), width))
    sep = torch.ones((3,width,5))
    sep[0,:,:] = 255

    toImage = transforms.ToPILImage()

    sep = toImage(sep)
    for i in range(batch):
        temp = toImage(ip[i].view(1,512,512))
        # temp = toImage(temp)
        draw = ImageDraw.Draw(temp)
        
        # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 16)
        # font = ImageFont.load_default()
        # draw.text((0, 0), t[i], fill=(255), font=font)
        result.paste(im=temp, box=(i*(height+5), 0))
        draw = ImageDraw.Draw(sep)
        result.paste(im=sep, box=(i*(height+5)+height, 0))
    return result
    


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/dj/Downloads/lidar/nuscene/save_new', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--gen_type', type=str, default="p2p-content", help='different type of generator')
parser.add_argument('--output_dir', type=str, default="./thesis/", help='dir to save model weights')
parser.add_argument('--content_loss', type=bool, default=True, help='dir to save model weights')
parser.add_argument('--generator_A2B', type=str, default='/home/dj/Downloads/lidar/nuscene/save_new/cyclegan/output_dilate_encoder_dense_atrous_unet/3_netG_A2B_dense.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/home/dj/Downloads/lidar/nuscene/save_new/cyclegan/output_dilate_encoder_dense_atrous_unet/3_netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--dense_decoder', type=bool, default=True, help='dir to save model weights')
parser.add_argument('--model_dir', type=str, default='/home/dj/Downloads/lidar/nuscene/save_new/cyclegan/thesis/_p2p', help='dir to model weights')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
if opt.gen_type[:3] == 'p2p':
    netG_A2B = Generator(opt.input_nc, opt.output_nc, in_features=16)
    netG_B2A = Generator(opt.output_nc, opt.input_nc, in_features=16)
elif opt.gen_type[:6] == 'atrous':
    if opt.dense_decoder:
        netG_A2B = GeneratorMultiscaleDenseDecoder(opt.input_nc, opt.output_nc, in_features=16)
        netG_B2A = GeneratorMultiscaleDenseDecoder(opt.input_nc, opt.output_nc, in_features=16)
    else:
        netG_A2B = GeneratorMultiscale(opt.input_nc, opt.output_nc, in_features=16)
        netG_B2A = GeneratorMultiscale(opt.output_nc, opt.input_nc, in_features=16)
elif opt.gen_type[:4] == 'unet':
    netG_A2B = GeneratorUnet(opt.input_nc, opt.output_nc, in_features=16)
    netG_B2A = GeneratorUnet(opt.output_nc, opt.input_nc, in_features=16)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
weights_A2B = os.path.join(opt.model_dir, 'netG_A2B.pth')
weights_B2A = os.path.join(opt.model_dir, 'netG_B2A.pth')


netG_A2B.load_state_dict(torch.load(weights_A2B))
netG_B2A.load_state_dict(torch.load(weights_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5) ]

fake_normalize = [transforms.Normalize(0.5, 0.5)]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################
post_process = [transforms.ToPILImage()]
post_process = transforms.Compose(post_process)
###### Testing######

save_dir = os.path.join(opt.model_dir, 'img_gen_test_rec')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_legend = ["fake_lidar", "real_radar", "real_lidar", "fake_radar", "recover_lidar"]

cnt = 0

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    fake_norm = transforms.Compose(fake_normalize)
    recover_B = netG_A2B(fake_norm(fake_A))

    images_A = image_stitching((fake_B, batch['A'], batch['B'], fake_A, recover_B), image_legend)

    file_name = batch['name'][0]
    
    save_image(recover_B.view(1,512,512), os.path.join(save_dir, file_name))
    sys.stdout.write('\rGenerated images %05d of %05d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
