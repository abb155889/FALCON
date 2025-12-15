##########################################################################################################################################################
# training 중 test set에 대한 margin(모델의 확신도)를 측정하는 코드입니다. (test set을 해당 epoch에서의 margin 측정에만 이용하기 때문에, 성능엔 영향 없습니다.)
# 훈련 시 측정을 위해 100epoch을 수행하며, margin을 기록하고, csv 파일로 저장합니다. 이때 저장된 csv 파일은 make_margin_plot.py에 연결해주면 됩니다.
##########################################################################################################################################################

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_curve, classification_report, roc_curve
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import time
import pandas as pd
import json
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import cv2
from torch.autograd import grad
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('./google-research')   # clone한
from demogen import margin_utils     # 공식 구현 import
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from PIL import Image
from ...utils.data import ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader, ViSADataset, get_visa_mask_path_from_image_path, MVTEC_AD_CATEGORIES, MPDD_CATEGORIES, VISA_CATEGORIES, set_global_seed
from ...utils.falcon_arch import ResNet18Teacher, Student, Autoencoder, FusionConv, AnomalyDetector
from ...utils.noise_injection import GradCAM, adaptive_gradcam_noise

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco', 'casting', 'mpdd', 'visa'])
    parser.add_argument('-s', '--subdataset', default='cable',
                        help='One of 15 sub-datasets of Mvtec AD or "all" for all categories')
    parser.add_argument('-o', '--output_dir', default='experiment_results/seed/quan_99/cable/5shot')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='../mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded MVTEC LOCO dataset')
    parser.add_argument('-c', '--mpdd_path',
                        default='./dataset/MPDD/MPDD',
                        help='Downloaded MPDD dataset')
    parser.add_argument('-e', '--visa_path',
                        default='./dataset/ViSA',
                        help='Downloaded VISA dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--data_limit', type=int, default=5, 
                        help='Limit the number of training samples (e.g., 2, 4, 8)')
    return parser.parse_args()

on_gpu = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if on_gpu else 'cpu')
seed = 616
image_size = 256

def collate_test(batch):
    imgs, labels, paths = [], [], []
    for img, lab, p in batch:
        if isinstance(img, Image.Image):
            img = default_transform(img)
        imgs.append(img)
        labels.append(lab)
        paths.append(p)
    return torch.stack(imgs, 0), torch.as_tensor(labels), paths


# data loading
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

def reconstruction_loss(decoder_out, teacher_feat1, noise_mask):
    if noise_mask.shape[2:] != decoder_out.shape[2:]:
        noise_mask = F.interpolate(noise_mask.float(), size=decoder_out.shape[2:], mode='bilinear', align_corners=False)
    reconstruction_error = F.mse_loss(decoder_out, teacher_feat1, reduction='none')
    normal_mask = 1.0 - noise_mask
    weighted_error = reconstruction_error * (normal_mask + 2.0 * noise_mask)
    return torch.mean(weighted_error)

def probs_to_logits_binary(p_torch):
    p = p_torch.clamp(1e-6, 1-1e-6)
    log_p   = torch.log(p)
    log_1mp = torch.log(1 - p)
    return torch.stack([log_1mp, log_p], dim=1)  # [B,2]

@torch.no_grad()
def collect_probs_labels(dataloader, teacher, student, autoencoder, unified_decoder, detector, split_name):
    scores_all, labels_all = [], []
    for batch in dataloader:
        # train loader: (img_st, img_ae)
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and split_name == 'train':
            img_st, _ = batch
            y = torch.zeros(img_st.size(0), dtype=torch.long)
        # test loader: (imgs, labels, paths)
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            img_st, y, _ = batch
            y = y.long()
        else:
            raise ValueError(f"{split_name} batch format not supported: {type(batch)} / len={len(batch)}")

        if on_gpu:
            img_st = img_st.to(DEVICE, non_blocking=True)

        map_combined, *_ = predict(
            image=img_st, teacher=teacher, student=student,
            autoencoder=autoencoder, unified_decoder=unified_decoder,
            pixel_detector=detector
        )

        s_img = torch.amax(map_combined, dim=(2, 3)).squeeze(1)

        scores_all.append(s_img.cpu().numpy())
        labels_all.append(y.cpu().numpy())

    return np.concatenate(scores_all), np.concatenate(labels_all)


def compute_tf_margin(sess, logits_np, labels_np):
    g = sess.graph
    logits_ph = g.get_tensor_by_name('logits:0')
    labels_ph = g.get_tensor_by_name('labels:0')
    margin_t  = g.get_tensor_by_name('margin_out:0')
    return sess.run(margin_t, {logits_ph: logits_np, labels_ph: labels_np})


def train_single_category(category, config): #단일 카테고리에 대한 학습 및 평가
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

    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, category)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, category, 'test')
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    final_validation_auc = 0.0
    final_validation_f1 = 0.0
    final_validation_balanced_acc = 0.0
    final_pixel_auc = 0.0
    final_epoch = 0

    training_history = {
        'epochs': [],
        'validation_auc': [],
        'validation_f1': [],
        'validation_balanced_acc': [],
        'train_loss': [],
        'reconstruction_loss': [],
        'detector_loss': [],
        'final_epoch': 0,
        'final_validation_auc': 0.0,
        'final_validation_f1': 0.0,
        'final_validation_balanced_acc': 0.0,
        'final_train_loss': 0.0,
        'final_reconstruction_loss': 0.0,
        'final_detector_loss': 0.0
    }

    training_history.update({
        'margin_train_mean': [], 'margin_train_p25': [], 'margin_train_p75': [],
        'margin_test_mean':  [], 'margin_test_p25':  [], 'margin_test_p75':  []
    })

    if config.dataset == 'mvtec_ad':
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, category, 'train'),
            transform=transforms.Lambda(train_transform))

        if config.data_limit is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(len(full_train_set))[:config.data_limit]
            full_train_set = torch.utils.data.Subset(full_train_set, indices)
            print(f" 학습 데이터 {len(full_train_set)}개로 제한됨")

        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, category, 'test'))
    
        train_size = int(1 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mpdd':
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, category, 'train'),
            transform=transforms.Lambda(train_transform))

        if config.data_limit is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(len(full_train_set))[:config.data_limit]
            full_train_set = torch.utils.data.Subset(full_train_set, indices)
            print(f" 학습 데이터 {len(full_train_set)}개로 제한됨")
        
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, category, 'test'))
        
        train_size = int(1 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
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
        print(f" ViSA 학습 데이터: {len(full_train_set)}개")
        print(f" ViSA 테스트 데이터: {len(test_set)}개")
        train_set = full_train_set
        validation_set = full_train_set
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=1, pin_memory=True)
    validation_loader = DataLoader(full_train_set, batch_size=config.batch_size)

    teacher = ResNet18Teacher()
    student = Student(in_channels=128, out_channels=256)
    autoencoder = Autoencoder(in_channels=128, out_channels=256)
    unified_decoder = FusionConv(st_channels=256, ae_channels=256, out_channels=256)
    detector = AnomalyDetector(decoder_channels=256, teacher_channels=256)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    student.train()
    autoencoder.train()
    unified_decoder.train()
    detector.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        unified_decoder.cuda()
        detector.cuda()
        
    main_optimizer = torch.optim.Adam(
        itertools.chain(
            student.parameters(),
            autoencoder.parameters(),
            unified_decoder.parameters(),
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


    NUM_CLASSES = 2  # normal vs anomaly
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        logits_ph = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='logits')
        labels_ph = tf.placeholder(tf.int32,   shape=[None],            name='labels')
        labels_oh = tf.one_hot(labels_ph, NUM_CLASSES)

        margins_list = margin_utils.margin(logits_ph, labels_oh, [logits_ph])
        margin_tensor = tf.identity(margins_list[0], name='margin_out') 

        tf_config = tf.ConfigProto(device_count={'GPU': 0}) 
        sess = tf.Session(graph=tf_graph, config=tf_config)
        sess.run(tf.global_variables_initializer())
        
    train_start_time = time.perf_counter()

    total_loss_list = []
    st_loss_list = []
    ae_loss_list = []
    recon_loss_list = []
    det_loss_list = []

    for epoch in range(config.epochs):
        student.train()
        autoencoder.train()
        unified_decoder.train()
        detector.train()
        epoch_loss = 0.0
        epoch_st_loss = 0.0
        epoch_ae_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_detector_loss = 0.0
        tqdm_obj = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        # 학습
        for batch_idx, (image_st, image_ae) in enumerate(tqdm_obj):
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
            
            with torch.no_grad():
                feat1, feat2, feat3 = teacher(image_st)

            # 시각화 디렉토리: 시각화 안할거면 옵션 끄기
            noise_vis_dir = os.path.join(train_output_dir, 'noise_visualization')
            
            noised_image, noise_mask = adaptive_gradcam_noise(teacher, image_st, 0.05, 
                                        multiple_layers=['layer1', 'layer2', 'layer3'],
                                        ensemble_weights=[0.5, 0, 0.5],
                                        visualize=False, 
                                        save_dir=config.output_dir)
            
            with torch.no_grad():
                feat1_noisy, feat2_noisy, feat3_noisy = teacher(noised_image)
            
            st_out_clean = student(feat2)
            st_out = student(feat2_noisy)
            
            ae_out_clean = autoencoder(feat2)
            ae_out = autoencoder(feat2_noisy)
            
            student_loss = F.mse_loss(st_out, feat3)
            student_loss += F.mse_loss(st_out_clean, feat3)

            ae_loss = F.mse_loss(ae_out, feat3)
            ae_loss += F.mse_loss(ae_out_clean, feat3)

            decoder_out = unified_decoder(st_out, ae_out, feat1, feat2, feat3)
            decoder_out_clean = unified_decoder(st_out_clean, ae_out_clean, feat1, feat2, feat3)

            noisy_target = F.interpolate(noise_mask.float(), size=feat3.shape[2:], mode='bilinear', align_corners=False)
            

            recon_loss = reconstruction_loss(decoder_out_clean, feat3, torch.zeros_like(noise_mask))
            recon_loss += reconstruction_loss(decoder_out, feat3, noise_mask) # noisy output 
            
            total_loss = student_loss + ae_loss + recon_loss
            
            main_optimizer.zero_grad()
            total_loss.backward()
            main_optimizer.step()
            
            detector_optimizer.zero_grad()
            
            decoder_out_clean = decoder_out_clean.detach()

            anomaly_map_clean = detector(decoder_out_clean, feat3)
            anomaly_map_noisy = detector(decoder_out.detach(), feat3_noisy)
            
            clean_target = torch.zeros_like(anomaly_map_clean)
            noisy_target = F.interpolate(noise_mask.float(), size=anomaly_map_noisy.shape[2:], mode='bilinear', align_corners=False)
            
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
                'Recon': recon_loss.item(),
                'Det': detector_loss.item()
            })
        
        main_scheduler.step()
        detector_scheduler.step()
        
##########################################################################################################
#Margin 값 계산
##########################################################################################################
        
        train_eval_loader = DataLoader(train_set, batch_size=config.batch_size,
                               shuffle=False, num_workers=0) 
        test_eval_loader  = DataLoader(test_set,  batch_size=config.batch_size,
                               shuffle=False, num_workers=0,
                               collate_fn=collate_test)

        # 확률/라벨 수집
        probs_tr, labels_tr = collect_probs_labels(train_eval_loader, teacher, student, autoencoder, unified_decoder, detector, 'train')
        probs_te, labels_te = collect_probs_labels(test_eval_loader,  teacher, student, autoencoder, unified_decoder, detector, 'test')

        # 확률 -> 2-class logits
        # logits_tr = probs_to_logits_binary(torch.from_numpy(probs_tr)).numpy()
        # logits_te = probs_to_logits_binary(torch.from_numpy(probs_te)).numpy()
        def normalize01(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        p_tr = normalize01(probs_tr)
        p_te = normalize01(probs_te)

        logits_tr = probs_to_logits_binary(torch.from_numpy(p_tr)).numpy()
        logits_te = probs_to_logits_binary(torch.from_numpy(p_te)).numpy()

        # TF 공식 margin 계산
        m_tr = compute_tf_margin(sess, logits_tr, labels_tr)
        m_te = compute_tf_margin(sess, logits_te, labels_te)

        #정규화
        def q(x, p): return float(np.percentile(x, p))
        training_history['margin_train_mean'].append(float(m_tr.mean()))
        training_history['margin_train_p25'].append(q(m_tr, 25))
        training_history['margin_train_p75'].append(q(m_tr, 75))

        training_history['margin_test_mean'].append(float(m_te.mean()))
        training_history['margin_test_p25'].append(q(m_te, 25))
        training_history['margin_test_p75'].append(q(m_te, 75))

        np.savez(os.path.join(train_output_dir, f'epoch_{epoch:03d}_train_margin_tf.npz'),
                 logits=logits_tr, labels=labels_tr, margin=m_tr)
        np.savez(os.path.join(train_output_dir, f'epoch_{epoch:03d}_test_margin_tf.npz'),
                 logits=logits_te, labels=labels_te, margin=m_te)

        training_history['epochs'].append(epoch)
        training_history['train_loss'].append(epoch_loss / len(train_loader))
        training_history['reconstruction_loss'].append(epoch_recon_loss / len(train_loader))
        training_history['detector_loss'].append(epoch_detector_loss / len(train_loader))

        total_loss_list.append(epoch_loss / len(train_loader))
        st_loss_list.append(epoch_st_loss / len(train_loader))
        ae_loss_list.append(epoch_ae_loss / len(train_loader))
        recon_loss_list.append(epoch_recon_loss / len(train_loader))
        det_loss_list.append(epoch_detector_loss / len(train_loader))

        if epoch == config.epochs - 1:
            print(f'\nFinal evaluation for {category}...')
            #마지막 에폭에서만 평가 
            student.eval()
            autoencoder.eval()
            unified_decoder.eval()
            detector.eval()
            
            metrics = test_image_level_only(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, unified_decoder=unified_decoder,
                pixel_detector=detector,
                desc=f' Final Evaluation (epoch {epoch+1})', 
                fixed_threshold=0.1,
                save_anomaly_map_dir=None
            )

            current_auc = metrics['image_auc']
            current_f1 = metrics['image_f1'] 
            current_balanced_acc = metrics['image_balanced_accuracy']
            
            training_history['validation_auc'].append(current_auc)
            training_history['validation_f1'].append(current_f1)
            training_history['validation_balanced_acc'].append(current_balanced_acc)
            
            final_validation_auc = current_auc
            final_validation_f1 = current_f1
            final_validation_balanced_acc = current_balanced_acc
            final_epoch = epoch
            
            print(f'\n {category} Final Results (Epoch {epoch+1}):')
            print(f'Image AUC: {final_validation_auc:.2f}%  |  F1: {final_validation_f1:.2f}%')
            print(f'Final model at epoch {final_epoch+1} - Image AUC: {final_validation_auc:.2f}%')
            
            student.train()
            autoencoder.train()
            unified_decoder.train()
            detector.train()
        else:
            training_history['validation_auc'].append(0.0)
            training_history['validation_f1'].append(0.0)
            training_history['validation_balanced_acc'].append(0.0)

    train_end_time = time.perf_counter()
    total_train_time = (train_end_time - train_start_time)
    
    # 에폭별 loss를 csv로 저장
    epoch_losses = []
    for epoch in range(len(total_loss_list)):
        epoch_losses.append([
            epoch+1,
            total_loss_list[epoch],
            st_loss_list[epoch],
            ae_loss_list[epoch],
            recon_loss_list[epoch],
            det_loss_list[epoch]
        ])
    csv_path = os.path.join(train_output_dir, 'loss_per_epoch.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'st_loss', 'ae_loss', 'recon_loss', 'det_loss'])
        writer.writerows(epoch_losses)
    print(f"Per-epoch loss saved to {csv_path}")

####################################################################################################
#Margin 값 저장
####################################################################################################
    margin_csv = os.path.join(train_output_dir, 'margin_stats_per_epoch.csv')
    with open(margin_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch',
                    'train_mean','train_p25','train_p75',
                    'test_mean','test_p25','test_p75'])
        for e in range(len(training_history['epochs'])):
            w.writerow([
                e+1,
                training_history['margin_train_mean'][e],
                training_history['margin_train_p25'][e],
                training_history['margin_train_p75'][e],
                training_history['margin_test_mean'][e],
                training_history['margin_test_p25'][e],
                training_history['margin_test_p75'][e]
            ])

    # Plot
    epochs_arr = np.arange(1, len(training_history['epochs'])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_arr, training_history['margin_train_mean'], label='Train mean', linewidth=2)
    plt.plot(epochs_arr, training_history['margin_test_mean'],  label='Test mean',  linewidth=2, linestyle='--')
    plt.fill_between(epochs_arr,
                     training_history['margin_train_p25'],
                     training_history['margin_train_p75'],
                     alpha=0.2, label='Train IQR')
    plt.fill_between(epochs_arr,
                     training_history['margin_test_p25'],
                     training_history['margin_test_p75'],
                     alpha=0.2, label='Test IQR')
    plt.xlabel('Epoch'); plt.ylabel('Margin')
    plt.title('Margin distribution')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'margin_train_test_curve.png'))
    plt.close()

###############################################################################################################
    # 마지막 에폭의 모델로 최종 평가
    print(f"\nFinal evaluation for {category} with LAST EPOCH MODEL...")

    teacher.eval()
    student.eval()
    autoencoder.eval()
    unified_decoder.eval()
    detector.eval()
    
    final_metrics = test_image_level_only(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, unified_decoder=unified_decoder, 
        pixel_detector=detector,
        desc='FINAL TEST', 
        fixed_threshold=0.1,
        save_anomaly_map_dir=config.output_dir
    )

    inference_time = final_metrics.get('inference_time', 0.0)
    inference_time_per_image = final_metrics.get('inference_time_per_image', 0.0)
    
    print(f"\n Training completed for {category}!")
    print(f" Total training time: {total_train_time:.2f}s")
    print(f" Final Image AUC: {final_validation_auc:.2f}% at epoch {final_epoch+1}")
    print(f" Inference time: {inference_time:.2f}s ({inference_time_per_image:.2f}ms per image)")

    final_pixel_auc = 0.0  
    
    print(f"\n Training completed for {category}!")
    print(f" Total training time: {total_train_time:.2f}s")
    print(f" Final Image AUC: {final_validation_auc:.2f}% at epoch {final_epoch+1}")

    final_checkpoint = {
        'epoch': final_epoch,
        'category': category,
        'teacher_state_dict': teacher.state_dict(),
        'student_state_dict': student.state_dict(),
        'autoencoder_state_dict': autoencoder.state_dict(),
        'unified_decoder_state_dict': unified_decoder.state_dict(),
        'pixel_detector_state_dict': detector.state_dict(),
        'final_metrics': final_metrics,
        'final_image_auc': final_validation_auc,
        'final_f1': final_validation_f1,
        'final_balanced_acc': final_validation_balanced_acc,
        'training_history': training_history
    }

    final_model_dir = os.path.join(train_output_dir, 'final_model')
    os.makedirs(final_model_dir, exist_ok=True)
    torch.save(final_checkpoint, os.path.join(final_model_dir, 'final_model.pth'))
    
    result = {
        'category': category,
        'final_epoch': final_epoch,
        'training_time': total_train_time,
        'image_auc': final_validation_auc,
        'image_f1': final_validation_f1,
        'image_balanced_accuracy': final_validation_balanced_acc,
        'training_history': training_history,
        'inference_time': inference_time,
        'inference_time_per_image': inference_time_per_image,
        'test_samples': len(test_set)
    }

    plt.figure(figsize=(10,6))
    plt.plot(total_loss_list, label='Total Loss', color='black')
    plt.plot(st_loss_list, label='ST Loss', color='blue')
    plt.plot(ae_loss_list, label='AE Loss', color='green')
    plt.plot(recon_loss_list, label='Recon Loss', color='red')
    plt.plot(det_loss_list, label='Det Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (500 epochs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'loss_curve.png'))
    plt.close()
    print(f"Loss curve saved to {os.path.join(config.output_dir, 'loss_curve.png')}")
    
    return result

@torch.no_grad()
def predict(image, teacher, student, autoencoder, unified_decoder, pixel_detector):
    feat1, feat2, feat3 = teacher(image)  # feat1: 64x64x64, feat2: 128x32x32, feat3: 256x16x16

    st_out = student(feat2)    
    ae_out = autoencoder(feat2)
    
    decoder_out = unified_decoder(st_out, ae_out, feat1, feat2, feat3)  
    anomaly_map = pixel_detector(decoder_out, feat3)
    
    map_st = torch.mean((feat3 - st_out)**2, dim=1, keepdim=True)  
    map_ae = torch.mean((feat3 - ae_out)**2, dim=1, keepdim=True)  
    map_recon = torch.mean((feat3 - decoder_out)**2, dim=1, keepdim=True) 
    
    # 해상도 통일
    target_size = anomaly_map.shape[2:] 
    
    map_st_upsampled = F.interpolate(map_st, size=target_size, mode='bilinear', align_corners=False)
    map_ae_upsampled = F.interpolate(map_ae, size=target_size, mode='bilinear', align_corners=False)
    
    # Combined map
    map_combined = 0.2 * anomaly_map + 0.3 * map_st_upsampled + 0.3 * map_ae_upsampled + 0.2 * map_recon
    
    return map_combined, map_st_upsampled, map_ae_upsampled, map_recon

def test_image_level_only(test_set, teacher, student, autoencoder, unified_decoder, pixel_detector,
                         desc='Running image-level inference', fixed_threshold=0.1,
                         save_anomaly_map_dir=None):

    pure_inference_start = time.perf_counter()
    y_true_image = []
    y_score_image = []
    save_data = []
    
    # 순수 inference만 시간 측정
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image_tensor = default_transform(image)
        image_tensor = image_tensor[None]
        if on_gpu:
            image_tensor = image_tensor.cuda()

        map_combined, map_st, map_ae, map_recon = predict(
            image=image_tensor, teacher=teacher, student=student,
            autoencoder=autoencoder, unified_decoder=unified_decoder,
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

    pure_inference_end = time.perf_counter()
    pure_inference_time = (pure_inference_end - pure_inference_start)
    pure_inference_time_per_image = pure_inference_time * 1000 / len(test_set)
    
    print(f" Pure inference time: {pure_inference_time:.2f}s ({pure_inference_time_per_image:.2f}ms per image)")

    if save_anomaly_map_dir is not None:
        save_start_time = time.perf_counter()

        global_min = min(np.min(data['map_combined_np']) for data in save_data)
        global_max = max(np.max(data['map_combined_np']) for data in save_data)
        
        print(f"Global anomaly score range: {global_min:.6f} ~ {global_max:.6f}")

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

            raw_save_path = os.path.join(save_dir, f"{base_name}_raw_score.txt")
            with open(raw_save_path, 'w') as f:
                f.write(f"Max anomaly score: {np.max(map_combined_np):.6f}\n")
                f.write(f"Mean anomaly score: {np.mean(map_combined_np):.6f}\n")
                f.write(f"Target: {target} ({'Normal' if target == 0 else 'Anomaly'})\n")
        
        save_end_time = time.perf_counter()
        save_time = (save_end_time - save_start_time)
        save_time_per_image = save_time * 1000 / len(test_set)
        
        print(f"Save time: {save_time:.2f}s ({save_time_per_image:.2f}ms per image)")
        print(f"Total time: {pure_inference_time + save_time:.2f}s")

        total_time = pure_inference_time + save_time
        total_time_per_image = total_time * 1000 / len(test_set)
    else:
        total_time = pure_inference_time
        total_time_per_image = pure_inference_time_per_image

    percentile = 99
    fixed_threshold = np.percentile(y_score_image, percentile)

    image_metrics = calculate_metrics_fixed_threshold(y_true_image, y_score_image, fixed_threshold)
    
    return {
        'image_auc': image_metrics['auc'],       
        'image_auroc': image_metrics['auroc'],   
        'image_f1': image_metrics['f1'],          
        'image_balanced_accuracy': image_metrics['balanced_accuracy'],  
        'threshold_fixed': fixed_threshold,
        'pure_inference_time': pure_inference_time,
        'pure_inference_time_per_image': pure_inference_time_per_image,
        'inference_time': total_time,
        'inference_time_per_image': total_time_per_image
    }


def calculate_metrics_fixed_threshold(y_true, y_score, fixed_threshold=0.1): # threshold 0.1로 고정 
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    y_pred_fixed = (y_score >= fixed_threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_score) * 100
    auroc = roc_auc_score(y_true, y_score) * 100

    f1_fixed = f1_score(y_true, y_pred_fixed) * 100
    balanced_acc_fixed = balanced_accuracy_score(y_true, y_pred_fixed) * 100

    return {
        'auc': auc,
        'auroc': auroc,
        'f1': f1_fixed,
        'balanced_accuracy': balanced_acc_fixed,
        'threshold_fixed': fixed_threshold
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
    
    print(f" Starting WideResNet101 experiments for {len(categories)} categories: {categories}")
    
    all_results = []
    total_start_time = time.perf_counter()
    
    for i, category in enumerate(categories):
        print(f"\n{'='*80}")
        print(f" CATEGORY {i+1}/{len(categories)}: {category.upper()}")
        print(f"{'='*80}")
        
        try:
            result = train_single_category(category, config)
            all_results.append(result)
            
            print(f"\n {category} completed!")
            print(f" Results: Image AUC: {result['image_auc']:.2f}%, F1: {result['image_f1']:.2f}%")
            print(f" Inference: {result['inference_time']:.2f}s ({result['inference_time_per_image']:.2f}ms per image)")
            
            
        except Exception as e:
            print(f" Error in {category}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    
    print(f"\n{'='*100}")
    print(f" FINAL RESULTS SUMMARY - ALL CATEGORIES")
    print(f"{'='*100}")
    print(f"Total experiment time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Successfully completed: {len(all_results)}/{len(categories)} categories")
    
    if all_results:
        inference_times = [r['inference_time'] for r in all_results]
        inference_times_per_image = [r['inference_time_per_image'] for r in all_results]
        total_test_samples = sum([r['test_samples'] for r in all_results])
        
        avg_inference_time = np.mean(inference_times)
        avg_inference_time_per_image = np.mean(inference_times_per_image)
        std_inference_time = np.std(inference_times)
        std_inference_time_per_image = np.std(inference_times_per_image)
        
        results_df = pd.DataFrame([
            {
                'Category': result['category'],
                'Image_AUC': f"{result['image_auc']:.2f}%",
                'Image_F1': f"{result['image_f1']:.2f}%", 
                'Balanced_Acc': f"{result['image_balanced_accuracy']:.2f}%",
                'Final_Epoch': result['final_epoch'],
                'Training_Time': f"{result['training_time']:.1f}s",
                'Inference_Time': f"{result['inference_time']:.2f}s",
                'Inference_Per_Image': f"{result['inference_time_per_image']:.2f}ms",
                'Test_Samples': result['test_samples']
            }
            for result in all_results
        ])
        
        print(f"\n DETAILED RESULTS:")
        print(results_df.to_string(index=False))
    
        avg_image_auc = np.mean([r['image_auc'] for r in all_results])
        avg_image_f1 = np.mean([r['image_f1'] for r in all_results])
        avg_balanced_acc = np.mean([r['image_balanced_accuracy'] for r in all_results])
        
        print(f"\n AVERAGE PERFORMANCE:")
        print(f"Average Image AUC: {avg_image_auc:.2f}%")
        print(f"Average Image F1: {avg_image_f1:.2f}%")
        print(f"Average Balanced Accuracy: {avg_balanced_acc:.2f}%")

        print(f"\n INFERENCE TIME STATISTICS:")
        print(f"Average Inference Time: {avg_inference_time:.2f}s ± {std_inference_time:.2f}s")
        print(f"Average Inference Time Per Image: {avg_inference_time_per_image:.2f}ms ± {std_inference_time_per_image:.2f}ms")
        print(f"Total Test Samples: {total_test_samples}")
        print(f"Total Inference Time: {sum(inference_times):.2f}s")
        
        fastest_result = min(all_results, key=lambda x: x['inference_time_per_image'])
        slowest_result = max(all_results, key=lambda x: x['inference_time_per_image'])
        
        print(f"Fastest: {fastest_result['category']} ({fastest_result['inference_time_per_image']:.2f}ms per image)")
        print(f"Slowest: {slowest_result['category']} ({slowest_result['inference_time_per_image']:.2f}ms per image)")
        
        best_image_auc_result = max(all_results, key=lambda x: x['image_auc'])
        
        print(f"\n BEST PERFORMERS:")
        print(f"Best Image AUC: {best_image_auc_result['category']} ({best_image_auc_result['image_auc']:.2f}%)")
        
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(os.path.join(output_dir, 'wideresnet_results.csv'), index=False)
        
        detailed_results = {
            'experiment_info': {
                'architecture': 'WideResNet101_Modified',
                'dataset': config.dataset,
                'data_limit': config.data_limit,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'total_time': total_time,
                'completed_categories': len(all_results),
                'total_categories': len(categories)
            },
            'average_performance': {
                'image_auc': avg_image_auc,
                'image_f1': avg_image_f1,
                'balanced_accuracy': avg_balanced_acc
            },
            'inference_statistics': {
                'average_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'average_inference_time_per_image': avg_inference_time_per_image,
                'std_inference_time_per_image': std_inference_time_per_image,
                'total_test_samples': total_test_samples,
                'total_inference_time': sum(inference_times),
                'fastest_category': fastest_result['category'],
                'fastest_time': fastest_result['inference_time_per_image'],
                'slowest_category': slowest_result['category'],
                'slowest_time': slowest_result['inference_time_per_image']
            },
            'best_performers': {
                'best_image_auc': {
                    'category': best_image_auc_result['category'],
                    'score': best_image_auc_result['image_auc']
                }
            },
            'detailed_results': all_results
        }
        
        with open(os.path.join(output_dir, 'wideresnet_detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n Results saved to:")
        print(f"  - {os.path.join(output_dir, 'wideresnet_results.csv')}")
        print(f"  - {os.path.join(output_dir, 'wideresnet_detailed_results.json')}")
        
    print(f"\n{'='*100}")
    print(f" ALL WIDERESNET101 EXPERIMENTS COMPLETED")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
    