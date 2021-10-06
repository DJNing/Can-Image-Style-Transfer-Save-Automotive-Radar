import numpy as np
import torch
import os
from torch.autograd import Variable
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# class Pix2PixHDModel(BaseModel):
#     def name(self):
#         return 'Pix2PixHDModel'
    
#     def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
#         flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
#         def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
#             return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
#         return loss_filter
    
#     def initialize(self, opt):
        
#         BaseModel.initialize(self, opt)

#         if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
#             torch.backends.cudnn.benchmark = True

#         self.isTrain = opt.isTrain
#         self.use_features = opt.instance_feat or opt.label_feat
#         self.gen_features = self.use_features and not self.opt.load_features
#         input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

#         ##### define networks        
#         # Generator network
#         netG_input_nc = input_nc        
#         if not opt.no_instance:
#             netG_input_nc += 1
#         if self.use_features:
#             netG_input_nc += opt.feat_num                  
#         self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
#                                       opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
#                                       opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

#         # Discriminator network
#         if self.isTrain:
#             use_sigmoid = opt.no_lsgan
#             netD_input_nc = input_nc + opt.output_nc
#             if not opt.no_instance:
#                 netD_input_nc += 1
#             self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
#                                           opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

#         ### Encoder network
#         if self.gen_features:
#             self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
#                                           opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  

#         if self.opt.verbose:
#                 print('---------- Networks initialized -------------')

#         # load networks
#         if not self.isTrain or opt.continue_train or opt.load_pretrain:
#             pretrained_path = '' if not self.isTrain else opt.load_pretrain
#             self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
#             if self.isTrain:
#                 self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
#             if self.gen_features:
#                 self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

#         # set loss functions and optimizers
#         if self.isTrain:
#             if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
#                 raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
#             self.fake_pool = ImagePool(opt.pool_size)
#             self.old_lr = opt.lr

#             # define loss functions
#             self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
#             self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
#             self.criterionFeat = torch.nn.L1Loss()

#             if not opt.no_vgg_loss:             
#                 self.criterionVGG = networks.VGGLoss(self.gpu_ids)

#             # Names so we can breakout loss
#             self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

#             # initialize optimizers
#             # optimizer G
#             if opt.niter_fix_global > 0:                
#                 import sys
#                 if sys.version_info >= (3,0):
#                     finetune_list = set()
#                 else:
#                     from sets import Set
#                     finetune_list = Set()

#                 params_dict = dict(self.netG.named_parameters())
#                 params = []
#                 for key, value in params_dict.items():       
#                     if key.startswith('model' + str(opt.n_local_enhancers)):                    
#                         params += [value]
#                         finetune_list.add(key.split('.')[0])  
#                 print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
#                 print('The layers that are finetuned are ', sorted(finetune_list))                         
#             else:
#                 params = list(self.netG.parameters())
#             if self.gen_features:              
#                 params += list(self.netE.parameters())         
#             self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

#             # optimizer D                        
#             params = list(self.netD.parameters())    
#             self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

#     def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
#         if (self.opt.label_nc == 0) | (self.opt.r2l):
#             input_label = label_map.data.cuda()
#         else:
#             # create one-hot vector for label map 
#             size = label_map.size()
#             oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
#             input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
#             input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
#             if self.opt.data_type == 16:
#                 input_label = input_label.half()

#         # get edges from instance map
#         if not self.opt.no_instance:
#             inst_map = inst_map.data.cuda()
#             edge_map = self.get_edges(inst_map)
#             input_label = torch.cat((input_label, edge_map), dim=1)         
#         input_label = Variable(input_label, volatile=infer)

#         # real images for training
#         if real_image is not None:
#             real_image = Variable(real_image.data.cuda())

#         # instance map for feature encoding
#         if self.use_features:
#             # get precomputed feature maps
#             if self.opt.load_features:
#                 feat_map = Variable(feat_map.data.cuda())
#             if self.opt.label_feat:
#                 inst_map = label_map.cuda()

#         return input_label, inst_map, real_image, feat_map

#     def discriminate(self, input_label, test_image, use_pool=False):
#         input_concat = torch.cat((input_label, test_image.detach()), dim=1)
#         if use_pool:            
#             fake_query = self.fake_pool.query(input_concat)
#             return self.netD.forward(fake_query)
#         else:
#             return self.netD.forward(input_concat)

#     def forward(self, label, inst, image, feat, infer=False):
#         # Encode Inputs
#         input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)  

#         # Fake Generation
#         if self.use_features:
#             if not self.opt.load_features:
#                 feat_map = self.netE.forward(real_image, inst_map)                     
#             input_concat = torch.cat((input_label, feat_map), dim=1)                        
#         else:
#             input_concat = input_label
#         fake_image = self.netG.forward(input_concat)

#         # Fake Detection and Loss
#         pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
#         loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

#         # Real Detection and Loss        
#         pred_real = self.discriminate(input_label, real_image)
#         loss_D_real = self.criterionGAN(pred_real, True)

#         # GAN loss (Fake Passability Loss)        
#         pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
#         loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
#         # GAN feature matching loss
#         loss_G_GAN_Feat = 0
#         if not self.opt.no_ganFeat_loss:
#             feat_weights = 4.0 / (self.opt.n_layers_D + 1)
#             D_weights = 1.0 / self.opt.num_D
#             for i in range(self.opt.num_D):
#                 for j in range(len(pred_fake[i])-1):
#                     loss_G_GAN_Feat += D_weights * feat_weights * \
#                         self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
#         # VGG feature matching loss
#         loss_G_VGG = 0
#         if not self.opt.no_vgg_loss:
#             loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
#         # Only return the fake_B image if necessary to save BW
#         return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ), None if not infer else fake_image ]

#     def inference(self, label, inst, image=None):
#         # Encode Inputs        
#         image = Variable(image) if image is not None else None
#         input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

#         # Fake Generation
#         if self.use_features:
#             if self.opt.use_encoded_image:
#                 # encode the real image to get feature map
#                 feat_map = self.netE.forward(real_image, inst_map)
#             else:
#                 # sample clusters from precomputed features             
#                 feat_map = self.sample_features(inst_map)
#             input_concat = torch.cat((input_label, feat_map), dim=1)                        
#         else:
#             input_concat = input_label        
           
#         if torch.__version__.startswith('0.4'):
#             with torch.no_grad():
#                 fake_image = self.netG.forward(input_concat)
#         else:
#             fake_image = self.netG.forward(input_concat)
#         return fake_image

#     def sample_features(self, inst): 
#         # read precomputed feature clusters 
#         cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
#         features_clustered = np.load(cluster_path, encoding='latin1').item()

#         # randomly sample from the feature clusters
#         inst_np = inst.cpu().numpy().astype(int)                                      
#         feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
#         for i in np.unique(inst_np):    
#             label = i if i < 1000 else i//1000
#             if label in features_clustered:
#                 feat = features_clustered[label]
#                 cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
#                 idx = (inst == int(i)).nonzero()
#                 for k in range(self.opt.feat_num):                                    
#                     feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
#         if self.opt.data_type==16:
#             feat_map = feat_map.half()
#         return feat_map

#     def encode_features(self, image, inst):
#         image = Variable(image.cuda(), volatile=True)
#         feat_num = self.opt.feat_num
#         h, w = inst.size()[2], inst.size()[3]
#         block_num = 32
#         feat_map = self.netE.forward(image, inst.cuda())
#         inst_np = inst.cpu().numpy().astype(int)
#         feature = {}
#         for i in range(self.opt.label_nc):
#             feature[i] = np.zeros((0, feat_num+1))
#         for i in np.unique(inst_np):
#             label = i if i < 1000 else i//1000
#             idx = (inst == int(i)).nonzero()
#             num = idx.size()[0]
#             idx = idx[num//2,:]
#             val = np.zeros((1, feat_num+1))                        
#             for k in range(feat_num):
#                 val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
#             val[0, feat_num] = float(num) / (h * w // block_num)
#             feature[label] = np.append(feature[label], val, axis=0)
#         return feature

#     def get_edges(self, t):
#         edge = torch.cuda.ByteTensor(t.size()).zero_()
#         edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
#         edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
#         edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
#         edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
#         if self.opt.data_type==16:
#             return edge.half()
#         else:
#             return edge.float()

#     def save(self, which_epoch):
#         self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
#         self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
#         if self.gen_features:
#             self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

#     def update_fixed_params(self):
#         # after fixing the global generator for a number of iterations, also start finetuning it
#         params = list(self.netG.parameters())
#         if self.gen_features:
#             params += list(self.netE.parameters())           
#         self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
#         if self.opt.verbose:
#             print('------------ Now also finetuning global generator -----------')

#     def update_learning_rate(self):
#         lrd = self.opt.lr / self.opt.niter_decay
#         lr = self.old_lr - lrd        
#         for param_group in self.optimizer_D.param_groups:
#             param_group['lr'] = lr
#         for param_group in self.optimizer_G.param_groups:
#             param_group['lr'] = lr
#         if self.opt.verbose:
#             print('update learning rate: %f -> %f' % (self.old_lr, lr))
#         self.old_lr = lr

# class InferenceModel(Pix2PixHDModel):
#     def forward(self, inp):
#         label, inst = inp
#         return self.inference(label, inst)

# ==================================================
class R2LImageDiscriminator(BaseModel):

    def init_loss_filter(self):
        flags = (True, True, True, True)
        def loss_filter(w_distance, lidar_F, radar_F, gp):
            return [l for (l,f) in zip([w_distance, lidar_F, radar_F, gp],flags) if f]
        return loss_filter

    def initialize(self, opt):
        
        BaseModel.initialize(self, opt)

        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True

        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        # set device
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            device_name = "cuda:" + str(self.gpu_ids[0])
            self.device = torch.device(device_name)

        # Discriminator network
        if self.isTrain:
            netD_input_nc = input_nc
            self.netD = networks.wDiscriminator(netD_input_nc)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netD, 'img_D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            
            self.old_lr = opt.lr

            self.loss_filter = self.init_loss_filter()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('w_distance', 'lidar_F', 'radar_F', 'gp')

            # initialize optimizers
            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(0.5, 0.9), weight_decay=1e-4)
    
    def forward(self, lidar, radar):
        lidar_feat = self.netD(lidar)
        radar_feat = self.netD(radar)

        gp = networks.calculate_gradient_penalty(self.netD, lidar, radar, device=self.device)

        distance = torch.mean(lidar_feat) - torch.mean(radar_feat) + self.opt.w_lambda * gp
        # distance = torch.mean(lidar_feat)
        return self.loss_filter(distance, torch.mean(lidar_feat), torch.mean(radar_feat), gp)

    def save(self, which_epoch):
        self.save_network(self.netD, 'img_D', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    
# ================================================================================================================

class R2LAE(BaseModel):

    def init_loss_filter(self):
        flags = (True, True, True, True, True, True, True)
        def loss_filter(gan_radar, gan_lidar, MSE_radar, MSE_lidar, w_distance_f, d_radar, d_lidar):
            return [l for (l,f) in zip((gan_radar, gan_lidar, MSE_radar, MSE_lidar, w_distance_f, d_radar, d_lidar),flags) if f]
        return loss_filter

    def initialize(self, opt):
        
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc
        
        # One shared encoder
        self.netE = networks.UDAEncoder(input_nc, opt.r2l_res, down_conv=opt.n_downsample_global, ngf=opt.ngf, resblock=opt.encoder_resblock, max_ch=opt.max_ch)

        self.radarG = networks.UDADecoder(self.netE.op_nc, 1, opt.r2l_res, self.netE.op_size, down_conv=opt.n_downsample_global, resblock=opt.encoder_resblock, max_ch=opt.max_ch)
        
        self.lidarG = networks.UDADecoder(self.netE.op_nc, 1, opt.r2l_res, self.netE.op_size, down_conv=opt.n_downsample_global, resblock=opt.encoder_resblock, max_ch=opt.max_ch)
                                                
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            device_name = "cuda:" + str(self.gpu_ids[0])
            self.device = torch.device(device_name)
            self.lidarG.to(self.device)
            self.radarG.to(self.device)
            self.netE.to(self.device) 


        # 3 discriminator for this training period
        if self.isTrain:
            
            if opt.wgan:# wgan
                self.netDF = networks.wDiscriminator(self.netE.op_nc, activate=False, flatten=False)
            else: 
                self.netDF = networks.DomainFeatureDescriminator(self.netE.op_nc)
            # lsgan
            use_sigmoid = opt.no_lsgan
            
            netD_input_nc = input_nc
            self.netDR = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netDL = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
                
            
        self.loss_filter = self.init_loss_filter()
        self.loss_names = self.loss_filter('gan_radar','gan_lidar','MSE_radar', 'MSE_lidar', 'w_distance_F', 'd_radar', 'd_lidar')
        
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            
        # Names so we can breakout loss
        self.loss_filter = self.init_loss_filter()
        
        # define optimizer
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        self.optimizer_DR = torch.optim.Adam(self.netDR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        self.optimizer_DL= torch.optim.Adam(self.netDL.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        self.optimizer_DF = torch.optim.Adam(self.netDF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        self.optimizer_lidarG = torch.optim.Adam(self.lidarG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        self.optimizer_radarG = torch.optim.Adam(self.radarG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
        
        self.adversarial_loss = torch.nn.MSELoss()
        self.encoder_loss = torch.nn.BCELoss()
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   


    def forward(self, radar, lidar, update_encoder=False, infer=False, export_feature=False):

        # 1. concat the radar and lidar images
        batchsize = radar.shape[0]
        ip = torch.cat((radar, lidar), dim=0)

        feat = self.netE(ip)

        # 2. recover the batch for w_distance calculation
        split = torch.split(feat, batchsize, dim=0)

        radar_feat = split[0]
        lidar_feat = split[1]

        # w_distance
        # w_distance = self.wganGP_loss(radar_feat, lidar_feat)

        # feature D loss
        # assign radar featue to be 1, and lidar feature to be 0
        # pred_radar_F = self.netDF(radar_feat)
        # pred_lidar_F = self.netDF(lidar_feat)
        pred_F = self.netDF(feat)
        shape = list(pred_F.shape)
        shape[0] = batchsize
        target_F = torch.cat((torch.ones(shape), torch.zeros(shape)))
        target_F = target_F.to(pred_F.device)
        loss_D_encoder = self.encoder_loss(pred_F, target_F)

        fake_F = torch.cat((torch.zeros(shape), torch.ones(shape)))
        fake_F = fake_F.to(pred_F.device)
        loss_encoder = self.encoder_loss(pred_F, fake_F)

        # update critic
        # if not update_encoder:
            # self.netDF.zero_grad()
            # w_distance.backward(retain_graph=True)
            # self.optimize_DF.step()

        if update_encoder:

            # 3. feed them to the generator network
            lidar_gen = self.lidarG(lidar_feat)
            radar_gen = self.radarG(radar_feat)

            # 4. calculate the gan loss
            MSE_lidar = self.adversarial_loss(lidar, lidar_gen)
            MSE_radar = self.adversarial_loss(radar, radar_gen)
            
            pred_fake_lidar = self.netDL(lidar_gen)
            pred_real_lidar = self.netDL(lidar)

            loss_G_Gan_lidar = self.criterionGAN(pred_fake_lidar, True)
            loss_D_Real_lidar = self.criterionGAN(pred_real_lidar, True)
            loss_D_Fake_lidar = self.criterionGAN(pred_fake_lidar, False)

            pred_fake_radar = self.netDR(radar_gen)
            pred_real_radar = self.netDR(radar)

            loss_G_Gan_radar = self.criterionGAN(pred_fake_radar, True)
            loss_D_Real_radar = self.criterionGAN(pred_real_radar, True)
            loss_D_Fake_radar = self.criterionGAN(pred_fake_radar, False)

            # the w_distance is negative, use minus here
            loss_gan_lidar = MSE_lidar + loss_G_Gan_lidar 
            loss_gan_radar = MSE_radar + loss_G_Gan_radar 

            loss_D_lidar = loss_D_Real_lidar + loss_D_Fake_lidar
            loss_D_radar = loss_D_Real_radar + loss_D_Fake_radar

            # 5. update weights
            self.netE.zero_grad()
            self.lidarG.zero_grad()
            self.radarG.zero_grad()
            self.netDL.zero_grad()
            self.netDR.zero_grad()
            self.netDF.zero_grad()

            loss_gan_lidar.backward(retain_graph=True)
            loss_gan_radar.backward(retain_graph=True)

            loss_D_lidar.backward(retain_graph=True)
            loss_D_radar.backward(retain_graph=True)

            loss_D_encoder.backward(retain_graph=True)
            loss_encoder.backward(retain_graph=True)

            self.optimizer_E.step()
            self.optimizer_DF.step()
            self.optimizer_DR.step()
            self.optimizer_DL.step()
            self.optimizer_lidarG.step()
            self.optimizer_radarG.step()
            
            fake_image = {'lidar_gen':lidar_gen, 'radar_gen':radar_gen}

            return [ self.loss_filter( loss_gan_radar, loss_gan_lidar, MSE_radar, MSE_lidar, loss_D_encoder, loss_D_radar, loss_D_lidar ), None if not infer else fake_image ]
        
        else:

            return [None, None]


    def inference(self, radar, lidar):

        batchsize = radar.shape[0]
        ip = torch.cat((radar, lidar), dim=0)

        feat = self.netE(ip)

        # 2. recover the batch for w_distance calculation
        split = torch.split(feat, batchsize, dim=0)

        radar_feat = split[0]
        lidar_feat = split[1]

        lidar_gen = self.lidarG(lidar_feat)
        radar_gen = self.radarG(radar_feat)
        return {"lidar_gen":lidar_gen, "radar_gen":radar_gen}


    def save(self, which_epoch):
        print(self.save_dir)

        self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        self.save_network(self.netDF, 'DF', which_epoch, self.gpu_ids)
        self.save_network(self.netDR, 'DR', which_epoch, self.gpu_ids)
        self.save_network(self.netDL, 'DL', which_epoch, self.gpu_ids)
        self.save_network(self.lidarG, 'GL', which_epoch, self.gpu_ids)
        self.save_network(self.radarG, 'GR', which_epoch, self.gpu_ids)


# code cite from https://github.com/Lornatang/WassersteinGAN_GP-PyTorch
    def wganGP_loss(self, real_f, gen_f):
        
        real_op = self.netDF(real_f)
        fake_op = self.netDF(gen_f)
        errD_real = torch.mean(real_op)

        errD_fake = torch.mean(fake_op)
        
        gradient_penalty = networks.calculate_gradient_penalty(self.netDF,
                                                    real_f.data, gen_f.data,
                                                    self.device)

        errD = -errD_real + errD_fake + gradient_penalty * 10
        
        return errD


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_lidar_E.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_radar_E.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
# ================================================================================================================

