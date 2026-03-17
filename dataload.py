# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:34:14 2024

@author: S4300F
"""
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
# import numpy as np
# import torch
from monai.transforms import Compose,LoadImaged, AddChanneld, ToTensord,CropForegroundd,RandSpatialCropd,RandSpatialCropSamplesd,Resized
from monai.transforms import  Orientationd,RandGaussianNoised,RandScaleIntensityd,RandShiftIntensityd,ResizeWithPadOrCropd, AsDiscrete,CenterSpatialCropd
from monai.transforms import RandGaussianSharpend,ScaleIntensityRanged,AsDiscreted,ToNumpyd
import sys
sys.path.append("..") 

from tqdm import tqdm
import nibabel as nib

import numpy as np
from monai.transforms import MapTransform
import torch


dataset_dir = '../autodl-tmp/'
data_txt_path = './dataset_list'
import random
random.seed(2025)  
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def train_dataload(batch_size=1):
    train_img = []
    train_lbl = []

    for line in open(data_txt_path + '/train12.txt'):
        train_img.append(dataset_dir + line.strip().split()[0])#img_cut_fat
        train_lbl.append(dataset_dir + line.strip().split()[1]) #血管
       
    data_dicts_train = [{'image': image, 'label': label}
                for image, label in zip(train_img, train_lbl)]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            # RandSpatialCropSamplesd(
            #     keys=["image", "label"],
            #     roi_size=[-1, -1, 1],  # dynamic spatial_size for the first two dimensions
            #     num_samples=8,
            #     random_size=False,
            #     ),
            # CenterSpatialCropd(keys=["image", "label"],roi_size =(300,300,-1)),
            CropForegroundd(keys=["image", "label"],source_key='label'),
            
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-20,  # CT值截断范围
                a_max=4200,
                b_min=0.0, 
                b_max=1.0,         
            ),
            RandGaussianSharpend(
                keys=["image"],
                sigma1_x=(0.6, 0.7),  # 增强骨骼边缘
                sigma1_y=(0.6, 0.7),
                sigma1_z=(0.6, 0.7),  # 对3D数据重要
                approx="scalespace",  # 更精确的卷积计算
                prob=1.0
            ),
            # AsDiscreted(keys="label", to_onehot=True, n_classes=4),
            Resized(keys = ["image", "label"], spatial_size=(96,96,192),mode='nearest-exact'),
            # ToNumpyd(['image']),  # 保证是 numpy.ndarray
            # CLAHE3Dd(['image'], clip_limit=0.02),
            ToTensord(keys=["image", "label"]),
            
        ]
    )
    train_dataset = Dataset(data=data_dicts_train,transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
def val_dataload(batch_size=1):
    val_img = []
    val_lbl = []

    for line in open(data_txt_path + '/val12.txt'):
        val_img.append(dataset_dir + line.strip().split()[0])
        val_lbl.append(dataset_dir + line.strip().split()[1])               

    data_dicts_val = [{'image': image, 'label': label}
                for image, label in zip(val_img, val_lbl)]
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            CropForegroundd(keys=["image", "label"],source_key='label'),
            # CenterSpatialCropd(keys=["image", "label"],roi_size =(300,300,-1)),
            
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-20,  # CT值截断范围
                a_max=4200,
                b_min=0.0, 
                b_max=1.0,         
            ),
            RandGaussianSharpend(
                keys=["image"],
                sigma1_x=(0.6, 0.7),  # 增强骨骼边缘
                sigma1_y=(0.6, 0.7),
                sigma1_z=(0.6, 0.7),  # 对3D数据重要
                approx="scalespace",  # 更精确的卷积计算
                prob=1.0
            ),
            # AsDiscreted(keys=["label"],to_onehot=4),
            Resized(keys = ["image", "label"], spatial_size=(96,96,192),mode='nearest-exact'),
            # ToNumpyd(['image']),  # 保证是 numpy.ndarray
            # CLAHE3Dd(['image'], clip_limit=0.02),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_dataset = Dataset(data=data_dicts_val,transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    return val_loader

def test_dataload(batch_size=1):
    test_img = []
    test_lbl = []
    test_name = []
    affine = []
    for line in open(data_txt_path + '/test12.txt'):
        test_img.append(dataset_dir + line.strip().split()[0])
        test_lbl.append(dataset_dir + line.strip().split()[1])
        test_name.append(line.strip().split()[0].split('.')[0])
    for i in test_lbl:
        img_y = nib.load(i)
        affine.append(img_y.affine)
        
    data_dicts_test = [{'image': image, 'label': label, 'name':name, 'affine':affine}
                for image, label, name, affine in zip(test_img, test_lbl, test_name, affine)]
    test_transforms = Compose(       
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            CropForegroundd(keys=["image", "label"],source_key='label'),
            # CenterSpatialCropd(keys=["image", "label"],roi_size =(300,300,-1)),
            
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-20,  # CT值截断范围
                a_max=4200,
                b_min=0.0, 
                b_max=1.0,         
            ),
            RandGaussianSharpend(
                keys=["image"],
                sigma1_x=(0.6, 0.7),  # 增强骨骼边缘
                sigma1_y=(0.6, 0.7),
                sigma1_z=(0.6, 0.7),  # 对3D数据重要
                approx="scalespace",  # 更精确的卷积计算
                prob=1.0
            ),
            # AsDiscreted(keys=["label"],to_onehot=4),
            Resized(keys = ["image", "label"], spatial_size=(96,96,192),mode='nearest-exact'),
            # ToNumpyd(['image']),  # 保证是 numpy.ndarray
            # CLAHE3Dd(['image'], clip_limit=0.02),
            ToTensord(keys=["image", "label"]),
        ]
    )   
    test_dataset = Dataset(data=data_dicts_test,transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    return test_loader  

def test_dataload_nn(batch_size=1):
    test_img = []
    test_lbl = []
    test_name= []
  
    affine = []
    for line in open(data_txt_path + '/nnunet.txt'):
        test_img.append(dataset_dir + line.strip().split()[0])#nnunet
        test_lbl.append(dataset_dir + line.strip().split()[1])#label
        test_name.append(line.strip().split()[1])       
        
    for i in test_img:
        img_y = nib.load(i)
        affine.append(img_y.affine)
        
    data_dicts_test = [{'image': image, 'label': label, 'name':name, 'affine':affine}
                for image, label, name, affine in zip(test_img, test_lbl, test_name,affine)]
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            ToTensord(keys=["image", "label"]),
        ]
    )   
    test_dataset = Dataset(data=data_dicts_test,transform=test_transforms)
    test_loader_nn = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader_nn        
if __name__ == '__main__':
    test_loader = test_dataload(1)
    tmp=[]
    iterator = tqdm( #进度条
                          test_loader, desc="", dynamic_ncols=True
                          )
    for index, batch in enumerate(iterator):
        x, y= batch["image"], batch["label"]
        x, y,name,affine= batch["image"].float(), batch["label"].float(),batch["name"][0],batch["affine"][0]
        print(x.dtype)
        save_path_img = name.strip().split('/')[-1][0] + '_image.nii.gz'
        save_path_y = name.strip().split('/')[-1][0] + '_label.nii.gz'
        outx= x.numpy().squeeze(0).squeeze(0)
        outy= y.numpy().squeeze(0).squeeze(0)

        out_x = nib.Nifti1Image(outx,affine)  
        nib.save(out_x, save_path_img)
        out_y = nib.Nifti1Image(outy,affine)  
        nib.save(out_y, save_path_y)
        print(y.shape)
        
        
        
        # y = np.argmax(y, axis=1)   
        # y = y.astype(np.int32)
        # out= y.squeeze(0)
        # print(f"原始数据范围: {np.min(out)} ~ {np.max(out)}")
        # out = nib.Nifti1Image(out,affine)
        # nib.save(out, save_path)
        # print(y.shape)
        # tmp.append(y.shape)
        
        # y = y.squeeze(4)
        # # pic = np.array(y[:,0,:,:])
        # tmp = y.transpose(0,1)
        
        # onehot = AsDiscrete(to_onehot=4)
        # y = onehot(tmp)
        # y =  y.transpose(0,1)
        # print(x.shape,y.shape)
        # y = y.squeeze(4)
        # pic1 = np.array(y[:,0,:,:])
        # pic2 = np.array(y[:,1,:,:])
        # pic3 = np.array(y[:,2,:,:])
        # pic4 = np.array(y[:,3,:,:])

