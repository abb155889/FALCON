import os 
import random
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

MVTEC_AD_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

MPDD_CATEGORIES = [
    'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'
]

VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum',
    'macaroni1', 'macaroni2', 
    'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'pipe_fryum',
]

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            
def set_global_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ViSADataset(Dataset):
    def __init__(self, root_dir, category, transform=None, is_train=True, return_path=False, data_limit=None):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.is_train = is_train
        self.return_path = return_path
        self.data_limit = data_limit

        self.image_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        if self.is_train:
            normal_dir = os.path.join(self.root_dir, self.category, 'Data', 'Images', 'Normal')
            if not os.path.isdir(normal_dir):
                raise FileNotFoundError(f"Normal directory not found: {normal_dir}")
            files = [f for f in os.listdir(normal_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if self.data_limit is not None and len(files) > self.data_limit:
                random.seed(1001)
                random.shuffle(files)
                files = files[:self.data_limit]
            paths = [os.path.join(normal_dir, f) for f in files]
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
        else:
            normal_dir = os.path.join(self.root_dir, self.category, 'Data', 'Images', 'Normal')
            anomaly_dir = os.path.join(self.root_dir, self.category, 'Data', 'Images', 'Anomaly')

            if not os.path.isdir(normal_dir):
                raise FileNotFoundError(f"Normal test directory not found: {normal_dir}")
            if not os.path.isdir(anomaly_dir):
                raise FileNotFoundError(f"Anomaly test directory not found: {anomaly_dir}")

            normal_files = [f for f in os.listdir(normal_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            random.seed(1001)
            random.shuffle(normal_files)
            normal_files = sorted(normal_files)[:100]
            normal_paths = [os.path.join(normal_dir, f) for f in normal_files]
            self.image_paths.extend(normal_paths)
            self.labels.extend([0] * len(normal_paths))

            anomaly_files = [f for f in os.listdir(anomaly_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            anomaly_paths = [os.path.join(anomaly_dir, f) for f in anomaly_files]
            self.image_paths.extend(anomaly_paths)
            self.labels.extend([1] * len(anomaly_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            # transform(image) returns (image_st, image_ae)
            image_st, image_ae = self.transform(image)

        if self.is_train:
            return image_st, image_ae
        elif self.return_path:
            return image_st, image_ae, label, img_path
        else:
            return image_st, image_ae, label
                   
def get_visa_mask_path_from_image_path(image_path):
    parts = image_path.split(os.sep)
    if 'Images' in parts and 'Anomaly' in parts:
        mask_parts = []
        for part in parts:
            if part == 'Images':
                mask_parts.append('Masks')
            else:
                mask_parts.append(part)
        
        return os.sep.join(mask_parts)
    
    return None
