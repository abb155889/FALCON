###########################################################################################################################
# 이미지에 noise를 주입한 버전을 시각화하는 코드입니다. 
# <hyperparams>
# output_dir: 이미지 저장 경로  
# data_limit: 몇개를 저장할지 
# visualize_noise_image: 저장 옵션 (True로 설정해야 저장됨)
# 노이즈는 매번 랜덤한 위치에 저장됩니다. 한 이미지에 대해 다양한 노이즈 주입 버전을 보고 싶다면, epochs을 늘려보세요.
###########################################################################################################################

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
from utils.data import ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader, set_global_seed, ViSADataset, get_visa_mask_path_from_image_path, MVTEC_AD_CATEGORIES, MPDD_CATEGORIES, VISA_CATEGORIES
from utils.falcon_arch import ResNet18Teacher
from utils.noise_injection import GradCAM, adaptive_gradcam_noise

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mpdd', 'visa'])
    parser.add_argument('-s', '--subdataset', default='all',
                        help='One of 15 sub-datasets of Mvtec AD or "all" for all categories')
    parser.add_argument('-o', '--output_dir', default='./noise_injection')
    parser.add_argument('-a', '--mvtec_ad_path', default='./MVTEC')
    parser.add_argument('-c', '--mpdd_path', default='./MPDD')
    parser.add_argument('-e', '--visa_path', default='.//ViSA')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

    parser.add_argument('--data_limit', type=int, default=5, help='Number of noise injected image')
    parser.add_argument('--visualize_noise_image', type=bool, default=True, help='Make noise injection image')
    
    return parser.parse_args()

seed = 616
on_gpu = torch.cuda.is_available()
image_size = 256
out_channels = 128

default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_ae = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(
        brightness=0.15, 
        contrast=0.15, 
        saturation=0.1, 
        hue=0.02
    ),
])

def train_transform(image):  
    return default_transform(image), default_transform(transform_ae(image))

def train_single_category(category, config):
    print(f"\n Training {category}")
    set_global_seed(seed)
    
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'mpdd':
        dataset_path = config.mpdd_path
    elif config.dataset == 'visa':
        dataset_path = config.visa_path

    if config.dataset == 'mvtec_ad':
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, category, 'train'),
            transform=transforms.Lambda(train_transform))
        
        if config.data_limit is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(len(full_train_set))[:config.data_limit]
            full_train_set = torch.utils.data.Subset(full_train_set, indices)
            print(f"data limit: {len(full_train_set)}")
    
        train_set = full_train_set

    elif config.dataset == 'mpdd':
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, category, 'train'),
            transform=transforms.Lambda(train_transform))

        if config.data_limit is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(len(full_train_set))[:config.data_limit]
            full_train_set = torch.utils.data.Subset(full_train_set, indices)
            print(f"data limit: {len(full_train_set)}")
        
        train_set = full_train_set
    
    elif config.dataset == 'visa':
        full_train_set = ViSADataset(
            root_dir=dataset_path,
            category=category,
            transform=transforms.Lambda(train_transform),
            is_train=True,
            data_limit=config.data_limit
        )
        print(f"data limit: {len(full_train_set)}")
        train_set = full_train_set
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=1, pin_memory=True)
    teacher = ResNet18Teacher()

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    if on_gpu:
        teacher.cuda()
    for epoch in range(config.epochs):
        tqdm_obj = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (image_st, image_ae) in enumerate(tqdm_obj):
            if on_gpu:
                image_st = image_st.cuda()    
            with torch.no_grad():
                feat1, feat2, feat3 = teacher(image_st)
            noised_image, noise_mask = adaptive_gradcam_noise(teacher, image_st, 
                                        multiple_layers=['layer1', 'layer2', 'layer3'],
                                        ensemble_weights=[0.5, 0, 0.5],
                                        visualize=config.visualize_noise_image,
                                        save_dir=config.output_dir,
                                        )

def main():
    set_global_seed(seed)
    config = get_argparse()
    
    if config.subdataset == 'all':
        if config.dataset == 'mvtec_ad':
            categories = MVTEC_AD_CATEGORIES
        elif config.dataset == 'mpdd':
            categories = MPDD_CATEGORIES
        elif config.dataset == 'visa':
            categories = VISA_CATEGORIES
        else:
            raise ValueError("--subdataset all is only supported for mvtec_ad")
    else:
        categories = [config.subdataset]
    
    print(f" Making noise injection image for {len(categories)} categories: {categories}")
    print(f"If you want to stop making, press ctrl+c")
    print("="*50)
    
    for i, category in enumerate(categories):
        try:
            train_single_category(category, config)

            print(f"\n {category} completed!")
        except Exception as e:
            print(f" Error in {category}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
    
