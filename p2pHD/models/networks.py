import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import os
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'multiscale':
        netG = MultiscaleGlobalGenerator(input_nc, output_nc, ngf)
        print('Using multiscale generator')
    elif netG == 'autoencoder':
        netG = AutoEncoder(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
        print('Using auto encoder')
    elif netG == "UNet":
        netG = UNetGenerator(input_nc, output_nc, n_blocks_global, ngf)
        print('Using UNet Generator')
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   

    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    # print(netD)
    print(input_nc, ndf, n_layers_D, num_D, getIntermFeat, norm, use_sigmoid)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        
        
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
# =============================================
# a more convient version for GAN Inversion

class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(AutoEncoder, self).__init__()        
        activation = nn.ReLU(True)        
        
        init_layer = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.init_layer = nn.Sequential(*init_layer)
        # model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        encoder = []
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        self.encoder = nn.Sequential(*encoder)
        ### resnet blocks

        resblock = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            resblock += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.resblock = nn.Sequential(*resblock)
        ### upsample         

        decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]

        self.decoder = nn.Sequential(*decoder)

        # output layer 
        output_layer = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.output_layer = nn.Sequential(*output_layer)
            
    def forward(self, x):
        x = self.init_layer(x)
        x = self.encoder(x)
        x = self.resblock(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x


# =============================================
class MultiscaleGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MultiscaleGlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        n_downsampling = 3

        b1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        b1 += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), norm_layer(ngf), activation]

        self.maxpool = nn.MaxPool2d(3, padding=1, stride=2)

        featureEncoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.b2Feat = nn.Sequential(*featureEncoder)
        self.b3Feat = nn.Sequential(*featureEncoder)

        connectB12 = [  nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1), 
                        norm_layer(ngf*4), activation]  
        connectB23 = [  nn.Conv2d(ngf*5, ngf*8, kernel_size=3, stride=2, padding=1), 
                        norm_layer(ngf*8), activation
                        ]  


        self.b1 = nn.Sequential(*b1)
        # self.b2 = nn.Sequential(*b2)
        # self.b3 = nn.Sequential(*b3)
        self.connectB12 = nn.Sequential(*connectB12)
        self.connectB23 = nn.Sequential(*connectB23)
        
        model = []
        ### downsample
        # for i in range(n_downsampling):
        #     mult = 2**i
        #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #               norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [ nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        b1Feat = self.b1(x)
        b2 = self.maxpool(x)
        b3 = self.maxpool(b2)
        b2Feat = self.b2Feat(b2)
        b3Feat = self.b3Feat(b3)
        b12 = torch.cat((b1Feat, b2Feat), dim=1)
        b12Feat = self.connectB12(b12)
        b123 = torch.cat((b12Feat, b3Feat), dim=1)
        finalFeat = self.connectB23(b123)
        out = self.model(finalFeat)
        # print(out.shape)
        return out
        
# =============================================

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]
                              
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class FeatureEncoder(nn.Module):
    def __init__(self, input_nc, ngf=32, n_downsampling=4, n_scale=3, 
                norm_layer=nn.BatchNorm2d, multi_scale=True):
        super(FeatureEncoder, self).__init__()
        norm_layer = get_norm_layer(norm_type='instance')
        activation = nn.ReLU(True)        
        self.multi_scale = multi_scale
        self.op_nc = (2 ** n_downsampling) * ngf
        self.device = 'cpu'
        if n_scale > n_downsampling:
            assert("n_scale should not be larger than n_downsampling")
        # n_downsampling = 3
        first_layer = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # conv_layer = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), norm_layer(ngf), activation]
        if self.multi_scale:
            self.branch = []
            self.down_conv = []
            for i in range(n_scale):
                if i == 0:
                    temp = nn.Sequential(*first_layer)
                    self.branch.append(temp)
                else:
                    temp = [nn.MaxPool2d(3, padding=1, stride=2)]*i
                    temp += first_layer
                    self.branch.append(nn.Sequential(*temp))
            
            for i in range(n_scale):
                mult = 2**i
                if i == 0:
                    add = 0
                else:
                    add = 1

                temp = [nn.Conv2d(ngf * (mult + add), ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

                self.down_conv.append(nn.Sequential(*temp)) 

            for i in range(n_downsampling-n_scale):
                mult = 2**(n_scale+i)
                temp = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
                self.down_conv.append(nn.Sequential(*temp))
                
        else:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
            ### downsample  
            for i in range(n_downsampling):
                mult = 2**i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                        norm_layer(ngf * mult * 2), activation]
            
            self.branch = nn.Sequential(*model)

    def setDevice(self, device):
        self.device = device
        if type(self.branch) == list:
            for i in self.branch:
                i.to(device)
        else:
            self.branch.to(device)

        if self.down_conv is not None:
            for i in self.down_conv:
                i.to(device)

    def saveNetwork(self, path, network_label, epoch_label):
        save_dict = OrderedDict()
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        if self.multi_scale:
            layer_cnt = 0
            for i in self.branch:
                save_dict[str(layer_cnt)] = i.state_dict()
                layer_cnt += 1
            for i in self.down_conv:
                save_dict[str(layer_cnt)] = i.state_dict()
                layer_cnt += 1
        else:
            save_dict = self.branch.state_dict()

        torch.save(save_dict, save_path)


    def loadNetwork(self, network_label, epoch_label, save_dir=''):   
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        # if not save_dir:
        #     save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            raise('%s not exists yet!' % save_path)
        else:
            if self.multi_scale:
                pretrain_dict = torch.load(save_path)
                layer_cnt = 0
                for i in self.branch:
                    i.load_state_dict(pretrain_dict[str(layer_cnt)])
                    layer_cnt += 1
                for i in self.down_conv:
                    i.load_state_dict(pretrain_dict[str(layer_cnt)])
                    layer_cnt += 1
            else:
                self.branch.load_state_dict(torch.load(save_path))
        
    def getParams(self):
        params = []
        if self.multi_scale:
            for i in self.branch:
                params += i.parameters()
            for i in self.down_conv:
                params += i.parameters()
        else:
            params += self.branch.parameters()
        return params

    def forward(self, ip):
        if self.multi_scale:
            # print(self.down_conv)
            branch_result = []
            for i in self.branch:
                branch_result.append(i(ip))
            for i in range(len(branch_result)):
                if i == 0:
                    # branch_cat = torch.cat((branch_result[i], branch_result[i+1]))
                    result = self.down_conv[i](branch_result[i])
                else:
                    branch_cat = torch.cat((result, branch_result[i]), dim=1)
                    result = self.down_conv[i](branch_cat)
                    # print(result.shape)
            # if len(branch_result) < len(self.down_conv):
            len_branch = len(branch_result)
            len_down_conv = len(self.down_conv)
            if len_branch < len_down_conv:
                for i in range(len_down_conv-len_branch):
                    down_conv = self.down_conv[len_branch + i]
                    result = down_conv(result)
        else:
            result = self.branch(ip)
        
        return result


class TransferGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks, ngf=32, n_upsampling=4, norm_layer=nn.BatchNorm2d, 
                    padding_type='reflect'):
        super(TransferGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type='instance')
        model = []
        activation = nn.ReLU(True) 
        # resblock
        mult = 2**n_upsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            model += [ nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)

    def forward(self, ip):
        return self.model(ip)


# code cited from https://github.com/Lornatang/WassersteinGAN_GP-PyTorch

def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


class wDiscriminator(nn.Module):

    def __init__(self, input_nc, ngf=16, n_layer=5, activate=False, flatten=True):
        super(wDiscriminator, self).__init__()

        def CLLblock(input_nc, output_nc, kw=4, padding=2, stride=1, bias=False, thres=0.2):
            block = []
            block.append(nn.Conv2d(input_nc, output_nc, kw, padding, stride, bias=bias))
            # use layer norm according to the paper
            block.append(nn.InstanceNorm2d(output_nc, affine=True))
            block.append(nn.LeakyReLU(0.2, True))
            return block

        nf = input_nc
        model = []
        self.flatten = flatten
        for i in range(n_layer-1):
            nf_prev = nf
            if i == 0:
                nf = ngf
            else:
                nf = min(nf * 2, 512)
            model += CLLblock(nf_prev, nf)
        
        # last layer without activation
        nf_prev = nf
        model.append(nn.Conv2d(nf_prev, 1, 4, 2, 1, bias=False))
        # model.append(nn.InstanceNorm2d(1, affine=True))
        if activate:
            model.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*model)

    def forward(self, ip):
        
        out = self.model(ip)
        if self.flatten:
            out = torch.mean(torch.flatten(out))
        return out

# ============================================

class UDAEncoder(nn.Module):
    def __init__(self, ipc, size, down_conv=3, ngf=16, resblock=3, linear=False, 
                    norm_layer=nn.BatchNorm2d, max_ch=512, multi_scale=False):
        super(UDAEncoder, self).__init__()
        self.max_ch = max_ch
        self.multi_scale = multi_scale
        self.size = size
        norm_layer = get_norm_layer(norm_type='instance')
        def conv_block(ipc, opc, k, s, p, inplace=True):
            block_list = [
                nn.Conv2d(ipc, opc, k, s, p),
                nn.BatchNorm2d(opc),
                nn.ReLU(inplace)
            ]
            return block_list
        activation = nn.ReLU(True)
        if not self.multi_scale:
            layer_list = []
            first_layer = [nn.ReflectionPad2d(3), nn.Conv2d(ipc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
            layer_list += first_layer
            
            # try to maintain the architecture of pix2pixHD
            prev_nf = ngf
            nf = min(prev_nf*2, self.max_ch)
            for i in range(down_conv):
                layer_list += conv_block(prev_nf, nf, 3, 2, 1)
                prev_nf = nf
                nf = min(prev_nf*2, self.max_ch)
            
            # resblock
            temp_size = self.size / (2 ** down_conv)
            for i in range(resblock):
                layer_list += [ResnetBlock(prev_nf, padding_type='reflect', norm_layer=norm_layer)]
                
            self.op_size = temp_size
            self.op_nc = nf
            if linear:
                num_ip = (temp_size ** 2) * prev_nf
                layer_list += [nn.Linear(num_ip, max_ch)]
                

            self.model = nn.Sequential(*layer_list)

        else:
            # will implement later
            assert('encoder for multiscale is not yet implemented')
            pass
        

    def forward(self, x):
        
        if not self.multi_scale:
            return self.model(x)
        else:
            assert('encoder for multiscale is not yet implemented')
            return None
        
# ============================================


class UDADecoder(nn.Module):
    def __init__(self, ipc, opc, size, f_size, down_conv=3, ngf=16, resblock=3, linear=False, 
                    norm_layer=nn.BatchNorm2d, max_ch=512, upsample=True):
        super(UDADecoder, self).__init__()
        self.max_ch = max_ch
        
        self.size = size
        norm_layer = get_norm_layer(norm_type='instance')
        def upconv_block(ipc, opc, k, s, p, inplace=True):
            block_list = [
                nn.ConvTranspose2d(ipc, opc, k, s, p),
                nn.BatchNorm2d(opc),
                nn.ReLU(inplace)
            ]
            return block_list


        if linear:
            # the input will be 1D vector
            self.rev_linear = nn.Linear(max_ch, 4*max_ch)
            temp_size = 4
            nc = max_ch
        else:
            self.rev_linear = None
            temp_size = f_size
            nc = ipc
        layer_list = []

        for i in range(resblock):
            layer_list += [ResnetBlock(nc, 'reflect', norm_layer), 
                        norm_layer(nc),
                        nn.ReLU(inplace=True)]
        
        if linear:
            up_cnt = int(size/temp_size)
            for i in range(up_cnt):
                next_nc = max(nc // 2, 4)
                layer_list += upconv_block(nc, next_nc, 4, 2, 1)
                nc = next_nc
        else:
            for i in range(down_conv):
                next_nc = max(nc // 2, 4)
                layer_list += upconv_block(nc, next_nc, 4, 2, 1)
                nc = next_nc

        # last layer
        layer_list += [nn.ReflectionPad2d(3),
                        nn.Conv2d(nc, opc, kernel_size=7, padding=0),
                        nn.Tanh()]

        self.model = nn.Sequential(*layer_list)


    def forward(self, x):
        
        return self.model(x)

# ============================================

# use a domain descriminator to replace w_distance measurement


class DomainFeatureDescriminator(nn.Module):
    def __init__(self, ipc, n_layer=5, linear=False, min_nf=8):
        super(DomainFeatureDescriminator, self).__init__()
        layer = []
        if linear:
            assert('linear modle not implemented')
            pass
        else:
            # 5 resblock 
            prev_nf = ipc
            nf = max(ipc//2, min_nf)
            for i in range(4):
              layer += [nn.Conv2d(prev_nf, nf, 3, 1, 1), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2)]  
              prev_nf = nf
              nf = max(prev_nf//2, min_nf)
            layer += [nn.Conv2d(prev_nf, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()]
            
        self.model = nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
#                  padding_type='reflect'):

class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, in_features=64):
        super(UNetGenerator, self).__init__()

        # temp = in_features

        # Initial convolution block       
        
        self.init_block = nn.Sequential(*[   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ])

        # Downsampling
        # in_features = 64
        def down_block(ipc, opc):
            result = [  nn.Conv2d(ipc, opc, 7, stride=2, padding=3),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        encoder = []
        out_features = in_features*2
        for _ in range(3):
            encoder += [down_block(in_features, out_features)]
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.ModuleList(encoder)

        # Residual blocks
        res_block = []
        for _ in range(n_residual_blocks):
            res_block += [MSRB(in_features, 1)]

        self.res_block = nn.Sequential(*res_block)

        # Upsampling
        def up_block(ipc, opc):
            result = [  nn.ConvTranspose2d(ipc, opc, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(opc),
                        nn.ReLU(inplace=True) ]
            return nn.Sequential(*result)

        decoder = []
        # for concatenation
        # in_features = in_features * 2
        out_features = in_features//2
        # print(in_features)
        for _ in range(3):
            # print(in_features)
            decoder += [up_block(in_features*2, out_features)]
            in_features = out_features
            out_features = in_features//2
        self.decoder = nn.ModuleList(decoder)

        # Output layer
        self.output_layer = nn.Sequential(*[  nn.ReflectionPad2d(3),
                    nn.Conv2d(out_features*2, output_nc, 7),
                    nn.Tanh() ])

    def forward(self, x):
        # return self.model(x)
        concat = None
        temp = self.init_block(x)
        for layer in self.encoder:
            if concat is None:
                concat = [layer(temp)]
            else:
                concat.append(layer(concat[-1]))
        temp = self.res_block(concat[-1])

        for cnt in range(len(self.decoder)):
            sub = cnt + 1
            idx = len(self.decoder) - sub
            ip_temp = torch.cat((temp, concat[idx]), dim=1)
            temp = self.decoder[cnt](ip_temp)
        
        temp = self.output_layer(temp)
        return temp

class MSRB(nn.Module):
    def __init__(self, ipc, stride) -> None:
        super(MSRB, self).__init__()
        self.b00 = self.__computing_node(ipc, ipc, 3, stride, 1)
        self.b01 = self.__computing_node(ipc, ipc, 5, stride, 2)
        self.b10 = self.__computing_node(ipc*2, ipc, 3, stride, 1)
        self.b11 = self.__computing_node(ipc*2, ipc, 5, stride, 2)
        self.out = nn.Conv2d(ipc*2, ipc, 1, stride)

        
    @staticmethod
    def __computing_node(ipc, opc, kernel_size, stride, padding):
        conv = nn.Conv2d(ipc, opc, kernel_size, stride=stride, padding=padding)
        relu = nn.ReLU(inplace=True)
        node = nn.Sequential(*[conv, relu])
        return node


    def forward(self, ip):
        ip00 = self.b00(ip)
        ip01 = self.b01(ip)
        
        ip1 = torch.cat((ip00, ip01), dim=1)
        ip10 = self.b10(ip1)
        ip11 = self.b11(ip1)
        ip_out = torch.cat((ip10, ip11), dim=1)
        out = self.out(ip_out)
        return out