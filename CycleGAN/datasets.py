import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, 'radar') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'lidar') + '/*.png'))

        # print(len(self.files_A))
        # print(len(self.files_B))

        split = int(len(self.files_A) * 0.5)
        test = int(len(self.files_A)*0.9)

        self.mask_transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5)
        ])

        if mode == 'train':
            self.files_A = self.files_A[:split]
            self.files_B = self.files_B[:split]
        else:
            self.files_A = self.files_A[test:]
            self.files_B = self.files_B[test:]
            pass
        self.mode = mode

    def __getitem__(self, index):
        # t1 = self.files_A[index % len(self.files_A)]
        # t2 = self.files_B[index % len(self.files_B)]
        item_A = self.transform(Image.open(self.files_A[index]))

        fileA = self.files_A[index % len(self.files_A)]
        name_A = fileA.split('/')[-1]

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index]))
        
        if self.mode == 'train':
            angle = random.randint(-45, 45)
            # random_degree = random.rand()
            item_A = TF.rotate(item_A, angle)
            item_B = TF.rotate(item_B, angle)

        item_A = self.mask_transform(item_A)
        item_B = self.mask_transform(item_B)


        return {'A': item_A, 'B': item_B, 'name':name_A}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))