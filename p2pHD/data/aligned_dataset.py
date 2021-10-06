import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import cv2
from scipy import ndimage
import torchvision.transforms.functional as transFunc
import torchvision.transforms as transforms
from glob import glob

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # slef.A_paths: list of file name for the label images

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

class Radar2LidarDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        dataroot = opt.dataroot    
        self.type = opt.inputType
        self.occupancy_dir = os.path.join(dataroot, "occupancy")
        self.radar_dir = os.path.join(dataroot, "radar")
        self.lidar_dir = os.path.join(dataroot, "lidar")
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5), 
                                            ])
        self.radar_files = sorted(glob(os.path.join(self.radar_dir, '*.png')))
        self.lidar_files = sorted(glob(os.path.join(self.lidar_dir, '*.png')))
        # file_name = os.path.join(dataroot, "timestamp.txt")

        # file_list = np.loadtxt(file_name, delimiter=' ', usecols=[0], dtype=str)
        # file_list = os.listdir(self.radar_dir)

        # import ipdb
        # ipdb.set_trace()
        split = int(len(self.radar_files) * 0.7)

        if opt.isTrain:
            self.file_list = self.radar_files[0:split]
        else:
            self.file_list = self.radar_files[split:]
        self.res = opt.r2l_res


        self.dataset_size = len(self.file_list) 
      
    def __getitem__(self, index):           

        ts = self.file_list[index].split('/')[-1].split('.')[0]
        # occupancy_path = os.path.join(self.occupancy_dir, ts+'.npy')
        radar_path = os.path.join(self.radar_dir, ts + '.png')
        lidar_path = os.path.join(self.lidar_dir, ts + '.png')
        # print(radar_path)
        if self.type == 'npy':
        
            # occupancy = np.load(occupancy_path)[:,:,0:2]
            radar = np.load(radar_path)
            lidar = np.load(lidar_path)[:,:,0:1]
        else:
            radar = cv2.imread(radar_path)[:,:,0]
            lidar = cv2.imread(lidar_path)[:,:,0]
            
        # radar = ndimage.rotate(radar, angle)
        # lidar = ndimage.rotate(lidar, angle)

        # occupancy = cv2.resize(occupancy,(224,224))
        lidar = cv2.resize(lidar,(512,512))
        radar = cv2.resize(radar,(512,512))
        # radar = (radar - radar.min())/(radar.max() - radar.min())

        # lidar[lidar!=0] = 1

        # # occupancy = np.transpose(occupancy, [2,0,1])
        # radar = np.reshape(radar, (1, self.res, self.res))
        # lidar = np.reshape(lidar, (1, self.res, self.res))

        radar = Image.fromarray(radar)
        lidar = Image.fromarray(lidar)

        # rotation
        angle = np.random.uniform() * 360
        radar = transFunc.rotate(radar, angle)
        lidar = transFunc.rotate(lidar, angle)

        # radar = np.array(radar)
        # lidar = np.array(lidar)


        # offset = 50
        # lidar_sample = lidar[offset:-offset, offset:-offset]
        # # crop image into size of 64 x 64/
        
        # x, y = np.nonzero(lidar_sample)
    
        # idx = np.random.randint(0, len(x))
        # x_pos = x[idx] + offset
        # y_pos = y[idx] + offset
        # lidar = lidar[x_pos - 32:x_pos + 32, y_pos - 32: y_pos + 32]
        # radar = radar[x_pos - 32:x_pos + 32, y_pos - 32: y_pos + 32]

        # radar = Image.fromarray(radar)
        # lidar = Image.fromarray(lidar)
        # radar = np.transpose(radar, [2,0,1])
        # lidar = np.transpose(lidar, [2,0,1])

        # occupancy = torch.from_numpy(occupancy).type(torch.FloatTensor)
        # radar = torch.from_numpy(radar).type(torch.FloatTensor)
        # lidar = torch.from_numpy(lidar).type(torch.FloatTensor)
        radar = self.transform(radar)
        lidar = self.transform(lidar)
        # print(radar.max(), radar.min())
        # print(lidar.max(), lidar.min())
        # print(lidar_patch.sum())
        # while True:
        #     idx = np.random.randint(0, high=upper_bound)
        #     ub = idx + 128
        #     lb = idx
        #     radar_patch = radar[:, lb:ub, lb:ub]
        #     lidar_patch = lidar[:, lb:ub, lb:ub] 
        #     if lidar_patch.sum() > 500:
        #         break
        # print(lidar_patch.sum())
        # print("finish")


        # print(idx)
        # print(lidar.shape)

        inst_tensor = feat_tensor = 0

        # input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
        #               'feat': feat_tensor, 'path': A_path}

        input_dict = {'label': radar, 'inst': inst_tensor, 'image': lidar, 
                      'feat': feat_tensor, 'path': radar_path}
        # print('debugging')
        # print(input_dict)
        return input_dict

    def __len__(self):
        return len(self.file_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Radar2LidarDataset'



class UDADataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        dataroot = opt.dataroot    
        self.type = opt.inputType
        
        self.radar_dir = os.path.join(dataroot, "radar")
        self.lidar_dir = os.path.join(dataroot, "lidar")
        self.transform = transforms.Compose([transforms.ToTensor()])

        file_name = os.path.join(dataroot, "timestamp.txt")

        file_list = np.loadtxt(file_name, delimiter=' ', usecols=[0], dtype=str)

        split = int(len(file_list) * 0.3)

        if opt.isTrain:
            self.file_list = file_list[0:split]
        else:
            self.file_list = file_list[split:]
        self.res = opt.r2l_res


        self.dataset_size = len(self.file_list) 
      
    def __getitem__(self, index):           

        ts = self.file_list[index]
        # occupancy_path = os.path.join(self.occupancy_dir, ts+'.npy')
        radar_path = os.path.join(self.radar_dir, ts+'.'+self.type)
        lidar_path = os.path.join(self.lidar_dir, ts+'.'+self.type)

        if self.type == 'npy':
        
            # occupancy = np.load(occupancy_path)[:,:,0:2]
            radar = np.load(radar_path)
            lidar = np.load(lidar_path)[:,:,0:1]
        else:
            radar = cv2.imread(radar_path)[:,:,0]
            lidar = cv2.imread(lidar_path)[:,:,0]
            
        lidar = cv2.resize(lidar,(512,512))
        radar = cv2.resize(radar,(512,512))

        radar = Image.fromarray(radar)
        lidar = Image.fromarray(lidar)

        # rotation
        angle = np.random.uniform() * 360
        radar = transFunc.rotate(radar, angle)
        lidar = transFunc.rotate(lidar, angle)

        radar = self.transform(radar)
        lidar = self.transform(lidar)

        input_dict = {'lidar': radar, 'radar': lidar}

        return input_dict

    def __len__(self):
        return len(self.file_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Radar2LidarDataset'