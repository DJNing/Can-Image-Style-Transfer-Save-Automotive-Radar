
#!/usr/bin/python3


import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from models import *
# from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/dj/Downloads/lidar/nuscene/save_new', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=9, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gen_type', type=str, default="bilinear_content", help='different type of generator')
parser.add_argument('--output_dir', type=str, default="./thesis/", help='dir to save model weights')
parser.add_argument('--content_loss', type=bool, default=False, help='dir to save model weights')
parser.add_argument('--dense_decoder', type=bool, default=True, help='dir to save model weights')
parser.add_argument('--resume', type=bool, default=False, help="weather to resume trainning from the latest ckpt file")

opt = parser.parse_args()
print(opt)

def energy_reg(fake, real):
    post_fake = (fake + 1)/2
    energy_fake = torch.sum(post_fake)
    energy_real = torch.sum(real*0.5 + 0.5)
    energy = torch.nn.L1Loss()(energy_fake, energy_real)
    return energy

def count_points(tensor):
    image = tensor*0.5 + 0.5
    image[image>0.5] = 1
    image[image<0.5] = 0
    b, c, _, _ = image.shape
    points = torch.sum(image)
    points = points/(b*c)
    return points



if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# create output dir
output_dir = opt.output_dir + '_' + opt.gen_type
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


###### Definition of variables ######
# Networks
if opt.gen_type[:3] == 'p2p':
    netG_A2B = Generator(opt.input_nc, opt.output_nc, in_features=16)
    netG_B2A = Generator(opt.output_nc, opt.input_nc, in_features=16)
elif opt.gen_type[:8] == 'bilinear':
    netG_A2B = GeneratorMultiscaleBilinear(opt.input_nc, opt.output_nc, in_features=16)
    netG_B2A = GeneratorMultiscaleBilinear(opt.output_nc, opt.input_nc, in_features=16)
elif opt.gen_type[:6] == 'atrous':
    if opt.dense_decoder:
        netG_A2B = GeneratorMultiscaleDenseDecoder(opt.input_nc, opt.output_nc, in_features=16)
        netG_B2A = GeneratorMultiscaleDenseDecoder(opt.output_nc, opt.input_nc, in_features=16)
    else:
        netG_A2B = GeneratorMultiscale(opt.input_nc, opt.output_nc, in_features=16)
        netG_B2A = GeneratorMultiscale(opt.output_nc, opt.input_nc, in_features=16)
elif opt.gen_type[:4] == 'unet':
    netG_A2B = GeneratorUnet(opt.input_nc, opt.output_nc, in_features=16)
    netG_B2A = GeneratorUnet(opt.output_nc, opt.input_nc, in_features=16)

netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# print(netG_A2B)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

if opt.resume:
    print('loading ckpt...')
    netG_A2B.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netG_A2B.pth')))
    netG_B2A.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netG_B2A.pth')))
    netD_A.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netD_A.pth')))
    netD_B.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netD_B.pth')))
else:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()

if opt.content_loss:
    criterion_cycle = contentLoss()
    criterion_identity = contentLoss()
else:
    print('use L1 loss')
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
Device = "cuda" if opt.cuda else "cpu"
# input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
# transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
#                 transforms.Grayscale(),
#                 #transforms.RandomCrop(opt.size), 
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

transforms_ = [ transforms.Grayscale(),
                #transforms.RandomCrop(opt.size), 
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]


dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), output_dir, opt.epoch)
# logger = Logger(opt.n_epochs, 1)
###################################

###### Training ######

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(Device)
        real_B = batch['B'].to(Device)

        bs = batch['A'].shape[0]
        target_real = Variable(Tensor(bs).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(bs).fill_(0.0), requires_grad=False)

        if real_A.shape[0] != opt.batchSize:
            print(real_A.shape)

        points = count_points(real_A)
        if points < 300:
            continue
        
        # print(points)
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) 
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) 

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake.view(-1), target_real) * 10

        energy_reg_B = energy_reg(fake_B, real_B) * 1e-3

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.view(-1), target_real) * 10
        energy_reg_A = energy_reg(fake_A, real_A) * 1e-3

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 2

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 2

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB #+ energy_reg_B + energy_reg_A
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real.view(-1), target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        if loss_D_A > 0.1:
            # print('loss_D_A: ', loss_D_A)
            loss_D_A.backward()

            optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real.view(-1), target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        if loss_D_B > 0.1:
            # print('loss_D_B', loss_D_B)
            loss_D_B.backward()

            optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B), "loss_D_A":loss_D_A, "loss_D_B":loss_D_B}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B, 'recovered_B':recovered_B})

        torch.cuda.empty_cache()

        # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    # if epoch % 2 == 0:
        # Save models checkpoints
    print("saving models")
    torch.save(netG_A2B.state_dict(), os.path.join(output_dir, str(epoch) + '_netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(output_dir, str(epoch) + '_netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(output_dir, str(epoch) + '_netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(output_dir, str(epoch) + '_netD_B.pth'))
    
    # latest ckpt
    torch.save(netG_A2B.state_dict(), os.path.join(output_dir, 'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(output_dir, 'netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(output_dir, 'netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(output_dir, 'netD_B.pth'))
###################################
