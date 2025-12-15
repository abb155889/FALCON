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
from utils.falcon_arch import ResNet18Teacher, Student, Autoencoder, FusionConv, AnomalyDetector
from utils.noise_injection import GradCAM, adaptive_gradcam_noise

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mpdd', 'visa'])
    parser.add_argument('-s', '--subdataset', default='all',
                        help='One of 15 sub-datasets of Mvtec AD or "all" for all categories')
    parser.add_argument('-o', '--output_dir', default='./experiment_results')
    parser.add_argument('-a', '--mvtec_ad_path', default='./MVTEC')
    parser.add_argument('-c', '--mpdd_path', default='./MPDD')
    parser.add_argument('-e', '--visa_path', default='.//ViSA')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data_limit', type=int, default=5, help='Limit the number of training samples (e.g., 2, 5, 10)')
    
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


# Weighted FusuionConv loss: 노이즈 영역에 가중치를 주어 loss에 포함합니다.
def reconstruction_loss(decoder_out, teacher_feat1, noise_mask):
    if noise_mask.shape[2:] != decoder_out.shape[2:]:
        noise_mask = F.interpolate(noise_mask.float(), size=decoder_out.shape[2:], mode='bilinear', align_corners=False)
    reconstruction_error = F.mse_loss(decoder_out, teacher_feat1, reduction='none')
    normal_mask = 1.0 - noise_mask
    weighted_error = reconstruction_error * (normal_mask + 2.0 * noise_mask) # 현재 가중치 2로 설정됨
    return torch.mean(weighted_error)

# 학습 함수 
def train_single_category(category, config):
    print(f"\n Training {category}")
    set_global_seed(seed)
    
    # 데이터셋 설정 
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'mpdd':
        dataset_path = config.mpdd_path
    elif config.dataset == 'visa':
        dataset_path = config.visa_path

    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, category)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, category, 'test')
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    if config.dataset == 'mvtec_ad':
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, category, 'train'),
            transform=transforms.Lambda(train_transform))
        
        if config.data_limit is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(len(full_train_set))[:config.data_limit]
            full_train_set = torch.utils.data.Subset(full_train_set, indices)
            print(f"data limit: {len(full_train_set)}")
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, category, 'test'))
    
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
        
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, category, 'test'))
        
        train_set = full_train_set
    
    elif config.dataset == 'visa':
        full_train_set = ViSADataset(
            root_dir=dataset_path,
            category=category,
            transform=transforms.Lambda(train_transform),
            is_train=True,
            data_limit=config.data_limit
        )
        test_set = ViSADataset(
            root_dir=dataset_path,
            category=category,
            is_train=False,
            return_path=True
        )
        print(f"data limit: {len(full_train_set)}")
        train_set = full_train_set
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=1, pin_memory=True)
    
    # 모델 구성 (모델의 자세한 구현은 utils/falcon_arch.py 참조)
    teacher = ResNet18Teacher()
    student = Student(in_channels=out_channels, out_channels=out_channels*2)
    autoencoder = Autoencoder(in_channels=out_channels, out_channels=out_channels*2)
    fusion_conv = FusionConv(st_channels=out_channels*2, ae_channels=out_channels*2, out_channels=out_channels*2)
    detector = AnomalyDetector(decoder_channels=out_channels*2, teacher_channels=out_channels*2)

    # teacher 모델은 eval 모드로 설정하여, 학습에서 제외
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # student, autoencoder, fusion_conv, detector는 모두 학습 대상임 
    student.train()
    autoencoder.train()
    fusion_conv.train()
    detector.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        fusion_conv.cuda()
        detector.cuda()
    
    # student, autoencoder, fusion_conv // detector 로 나누어서 2-stage 훈련 방식임 => 2개의 optimizer와 scheduler를 각각 구현 
    main_optimizer = torch.optim.Adam(
        itertools.chain(
            student.parameters(),
            autoencoder.parameters(),
            fusion_conv.parameters(),
        ),
        lr=3e-4, weight_decay=1e-5, betas=(0.9, 0.99),
    )
    
    detector_optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=3e-4, weight_decay=1e-5, betas=(0.9, 0.99),
    )
    
    main_scheduler = torch.optim.lr_scheduler.StepLR(
        main_optimizer, step_size=max(1, config.epochs//2), gamma=0.5
    )
    
    detector_scheduler = torch.optim.lr_scheduler.StepLR(
        detector_optimizer, step_size=max(1, config.epochs//2), gamma=0.5
    )

    # 학습 시작 
    for epoch in range(config.epochs):
        student.train()
        autoencoder.train()
        fusion_conv.train()
        detector.train()
        epoch_loss = 0.0
        epoch_st_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_detector_loss = 0.0
        tqdm_obj = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, (image_st, image_ae) in enumerate(tqdm_obj):
            if on_gpu:
                # image_st, image_ae는 하나의 이미지를 다른 transform 함수에 넣은 것 
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
                
            #########################################################################################################################     
            ### Stage 1: Student, Autoencoder, Fusion conv (훈련은 2-stage로 진행됨)
            #########################################################################################################################  
            
            # gradcam을 이용하여 이미지에 노이즈를 주입 => 가상의 결함 이미지 생성 (자세한 구현은 utils/noise_injection.py 참조)
            noised_image, noise_mask = adaptive_gradcam_noise(teacher, image_st, 
                                        multiple_layers=['layer1', 'layer2', 'layer3'],
                                        ensemble_weights=[0.5, 0, 0.5],) # 각 layer에서 cam을 뽑고 합칠 때 사용하는 가중치 (합이 1)
            with torch.no_grad():
                feat1, feat2, feat3 = teacher(image_st) # 정상에 대한 teacher의 1, 2, 3 layer 출력을 뽑음 
                feat1_noisy, feat2_noisy, feat3_noisy = teacher(noised_image) # 이상에 대한 teacher의 1, 2, 3 layer 출력을 뽑음 
            
            # stduent, autoencoder의 구조는 teacher 2nd output size => teacher 3rd output size로 작아지는 형태 
            st_out_clean = student(feat2) # 정상 teacher 2nd feature(input)에 대한 student의 output 
            st_out_noisy = student(feat2_noisy) # 이상 teacher 2nd feature(input)에 대한 student의 output 
            
            ae_out_clean = autoencoder(feat2) # 정상 teacher 2nd feature(input)에 대한 autoencoder의 output 
            ae_out_noisy = autoencoder(feat2_noisy) # 이상 teacher 2nd feature(input)에 대한 autoencoder의 output 
            
            # Goal: student, autoencoder는 (정상/이상)을 모두 => (정상)으로 재구성함
            # 따라서 이상에 대해서 st, auto는 (정상), teacher는 (이상) 특징을 뽑아내고, 이 차이를 계산하여 결함 영역을 판단 
            
            student_loss = F.mse_loss(st_out_noisy, feat3) # student가 재구성한 (이상) - 실제 teacher가 뽑은 (정상) 
            student_loss += F.mse_loss(st_out_clean, feat3) # student가 뽑은 (정상) - 실제 teacher가 뽑은 (정상)

            ae_loss = F.mse_loss(ae_out_noisy, feat3) # autoencoder가 재구성한 (이상) - 실제 teacher가 뽑은 (정상) 
            ae_loss += F.mse_loss(ae_out_clean, feat3) # autoencoder가 뽑은 (정상) - 실제 teacher가 뽑은 (정상)

            # fusion conv는 st, ae의 출력을 받아, 다시 한 번 (정상)으로의 재구성을 수행 
            fusion_out_noise = fusion_conv(st_out_noisy, ae_out_noisy, feat1, feat2, feat3) # (이상) => (정상)으로 
            fusion_out_clean = fusion_conv(st_out_clean, ae_out_clean, feat1, feat2, feat3) # (정상) => (정상)으로 

            noisy_target = F.interpolate(noise_mask.float(), size=feat3.shape[2:], mode='bilinear', align_corners=False) 
            
            # fusion conv가 (정상)으로 재구성하지 못한 영역에 가중치를 더하는 loss 설계 
            recon_loss = reconstruction_loss(fusion_out_clean, feat3, torch.zeros_like(noise_mask))
            recon_loss += reconstruction_loss(fusion_out_noise, feat3, noise_mask)
            
            # st, ae, fusion cov 끼리 훈련 시킴
            total_loss = student_loss + ae_loss + recon_loss
            
            main_optimizer.zero_grad()
            total_loss.backward()
            main_optimizer.step()
            
            #########################################################################################################################     
            ### Stage 2: Detector (fusion conv의 출력(정상으로 재구성)과 teacher의 출력(이상)을 비교하여 결함 영역을 찾아내는 역할)
            #########################################################################################################################  
            detector_optimizer.zero_grad()
            
            anomaly_map_clean = detector(fusion_out_clean.detach(), feat3)  # (정상)에 대한 fusion conv 출력
            anomaly_map_noisy = detector(fusion_out_noise.detach(), feat3_noisy) # (이상)에 대한 fusion conv 출력
      
            clean_target = torch.zeros_like(anomaly_map_clean) # (정상)에 대해서는 detector가 모두 0이라고 찾아야함 (이상이라고 생각하는 부분이 없어야함)
            noisy_target = F.interpolate(noise_mask.float(), size=anomaly_map_noisy.shape[2:], mode='bilinear', align_corners=False) # (이상)에 대해서는 detector가 노이즈 주입된 영역을 찾아야함
            
            detector_loss = F.binary_cross_entropy(anomaly_map_clean, clean_target) + \
                            F.binary_cross_entropy(anomaly_map_noisy, noisy_target)
            
            detector_loss.backward()
            detector_optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_st_loss += student_loss.item()
            epoch_ae_loss += ae_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_detector_loss += detector_loss.item()
            tqdm_obj.set_postfix({
                'Total': total_loss.item(),
                'ST': student_loss.item(),
                'AE': ae_loss.item(),
                'FC': recon_loss.item(),
                'Det': detector_loss.item()
            })
        
        main_scheduler.step()
        detector_scheduler.step()
        
    # 훈련 종료 시 평가 
    teacher.eval()
    student.eval()
    autoencoder.eval()
    fusion_conv.eval()
    detector.eval()
    
    final_metrics = test_image_level_only(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, fusion_conv=fusion_conv, 
        pixel_detector=detector,
        desc='test ...', 
        fixed_threshold=0.1,
        save_anomaly_map_dir=test_output_dir
    )

    test_auc = final_metrics['image_auc']
    print(f"\n Training completed for {category}!")
    print(f"Final image auroc: {test_auc:.2f}")

    final_checkpoint = {
        'category': category,
        'teacher_state_dict': teacher.state_dict(),
        'student_state_dict': student.state_dict(),
        'autoencoder_state_dict': autoencoder.state_dict(),
        'fusion_conv_state_dict': fusion_conv.state_dict(),
        'pixel_detector_state_dict': detector.state_dict(),
        'final_metrics': final_metrics,
        'final_image_auc': test_auc,
    }

    final_model_dir = os.path.join(train_output_dir, 'final_model')
    os.makedirs(final_model_dir, exist_ok=True)
    torch.save(final_checkpoint, os.path.join(final_model_dir, 'final_model.pth'))
    
    result = {
        'category': category,
        'image_auc': test_auc,
    }
    
    return result

@torch.no_grad()
def predict(image, teacher, student, autoencoder, fusion_conv, pixel_detector):
    feat1, feat2, feat3 = teacher(image) 
    # st, auto, fc 와 teacher의 차이, detector의 예측을 모두 반영 
    st_out = student(feat2)    
    ae_out = autoencoder(feat2)
    
    fusion_out = fusion_conv(st_out, ae_out, feat1, feat2, feat3)  
    anomaly_map = pixel_detector(fusion_out, feat3) # detector 
    
    map_st = torch.mean((feat3 - st_out)**2, dim=1, keepdim=True) # teacher - st
    map_ae = torch.mean((feat3 - ae_out)**2, dim=1, keepdim=True)  # teacher - ae
    map_fusion = torch.mean((feat3 - fusion_out)**2, dim=1, keepdim=True)  # teacher - fc 
    target_size = anomaly_map.shape[2:] 
    
    map_st_upsampled = F.interpolate(map_st, size=target_size, mode='bilinear', align_corners=False) # 사이즈 맞춰줌 
    map_ae_upsampled = F.interpolate(map_ae, size=target_size, mode='bilinear', align_corners=False) # 사이즈 맞춰줌 

    map_combined = 0.3 * map_st_upsampled + 0.3 * map_ae_upsampled + 0.2 * anomaly_map + 0.2 * map_fusion # 가중합 
    
    return map_combined, map_st_upsampled, map_ae_upsampled, map_fusion

def test_image_level_only(test_set, teacher, student, autoencoder, fusion_conv, pixel_detector,
                         desc='Running image-level inference', fixed_threshold=0.1,
                         save_anomaly_map_dir=None):
    y_true_image = []
    y_score_image = []
    save_data = []
    
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image_tensor = default_transform(image)
        image_tensor = image_tensor[None]
        if on_gpu:
            image_tensor = image_tensor.cuda()
            
        map_combined, map_st, map_ae, map_fusion = predict(
            image=image_tensor, teacher=teacher, student=student,
            autoencoder=autoencoder, fusion_conv=fusion_conv,
            pixel_detector=pixel_detector
        )
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined_np = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        
        y_true_image_single = 0 if (defect_class == 'good' or defect_class == 'ok_front' or defect_class == 'Normal') else 1
        y_score_image_single = np.max(map_combined_np) 
        y_true_image.append(y_true_image_single)
        y_score_image.append(y_score_image_single)

        if save_anomaly_map_dir is not None:
            save_data.append({
                'map_combined_np': map_combined_np,
                'path': path,
                'target': y_true_image_single,
                'defect_class': defect_class
            })

    if save_anomaly_map_dir is not None:
  
        global_min = min(np.min(data['map_combined_np']) for data in save_data)
        global_max = max(np.max(data['map_combined_np']) for data in save_data)
        
        for data in save_data:
            map_combined_np = data['map_combined_np']
            path = data['path']
            target = data['target']
            defect_class = data['defect_class']
            
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            save_dir = os.path.join(save_anomaly_map_dir, defect_class)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base_name}_anomaly_map.png")
            
            if global_max > global_min:
                norm_map = (map_combined_np - global_min) / (global_max - global_min)
            else:
                norm_map = np.zeros_like(map_combined_np)
 
            anomaly_map_uint8 = (norm_map * 255).astype(np.uint8)
            Image.fromarray(anomaly_map_uint8).save(save_path)
 
    y_true = np.array(y_true_image)
    y_score = np.array(y_score_image)
    auroc = roc_auc_score(y_true, y_score) * 100 
    return {   
        'image_auc': auroc,   
    }

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
    
    print(f" Starting experiments for {len(categories)} categories: {categories}")
    
    all_results = []
    
    for i, category in enumerate(categories):
        print(f"\n{'='*100}")
        print(f" CATEGORY {i+1}/{len(categories)}: {category.upper()}")
        print(f"{'='*100}")
        
        try:
            result = train_single_category(category, config)
            all_results.append(result)
            
            print(f"\n {category} completed!")
            print(f" Results: Image AUC: {result['image_auc']:.2f}")

            
        except Exception as e:
            print(f" Error in {category}: {e}")
            import traceback
            traceback.print_exc()
            continue


    print(f"\n{'='*100}")
    print(f" Result summary")
    print(f"{'='*100}")

    if all_results:
        results_df = pd.DataFrame([
            {
                'Category': result['category'],
                'Image_AUC': f"{result['image_auc']:.2f}%",
            }
            for result in all_results
        ])
        
        print(f"\n RESULTS:")
        print(results_df.to_string(index=False))
    
        avg_image_auc = np.mean([r['image_auc'] for r in all_results])
        print(f"Average Image AUC: {avg_image_auc:.2f}%")

        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        detailed_results = {
            'experiment_info': {
                'dataset': config.dataset,
                'data_limit': config.data_limit,
                'completed_categories': len(all_results),
                'total_categories': len(categories)
            },
            'average_performance': {
                'image_auc': avg_image_auc,
            },
            'detailed_results': all_results
        }
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n Results saved to:")
        print(f"  - {os.path.join(output_dir, 'results.csv')}")
        print(f"  - {os.path.join(output_dir, 'results.json')}")
        
    print(f"\n{'='*100}")
    print(f" All experiment complete")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
    
