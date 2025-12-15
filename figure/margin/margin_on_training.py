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
                                        visualize_noise=False, 
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
    
    print(f"🚀 Pure inference time: {pure_inference_time:.2f}s ({pure_inference_time_per_image:.2f}ms per image)")

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

################################################################################################################
####grad cam++ 기반 노이즈 주입 
################################################################################################################

from ...utils.noise_injection import GradCAM

def adaptive_gradcam_noise(teacher, image, noise_std=0.05, 
                          multiple_layers=['layer1', 'layer2'], 
                          ensemble_weights=[0.3, 0.7],
                          visualize_noise=False, save_dir=None):
    
    #layer1,2의 Grad-CAM++ 사용하여 객체 영역 탐지
    B, C, H, W = image.shape
    device = image.device

    cams = [] 
    for layer_name in multiple_layers:
        gradcam = GradCAM(teacher, layer_name)
        try:
            for b in range(B):
                single_image = image[b:b+1]
                cam = gradcam.generate_cam_lightweight(single_image, score_method='adaptive') 
                cam_full = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
                cams.append(cam_full[0, 0])
        finally:
            gradcam.remove_hooks()
            del gradcam
            torch.cuda.empty_cache() 
    
    # CAM 결과 합침
    ensemble_cam = torch.zeros((B, H, W), device=device)
    for b in range(B):
        weighted_cam = torch.zeros((H, W), device=device)
        for i, weight in enumerate(ensemble_weights):
            cam_idx = b * len(multiple_layers) + i
            if cam_idx < len(cams):
                weighted_cam += weight * cams[cam_idx]
        ensemble_cam[b] = weighted_cam
    
    noised_images = []
    noise_masks = []
    
    # 시각화를 위한 정보 저장
    visualization_info = []
    
    for b in range(B):
        single_image = image[b:b+1]
        cam_np = ensemble_cam[b].detach().cpu().numpy() 

        cam_uint8 = (cam_np * 255).astype(np.uint8)
        threshold_val, _ = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold_val / 255.0
        
        object_mask = (cam_np > threshold).astype(np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
 
        noised_image = single_image.clone()
        obj_pixels = np.where(object_mask > 0.7)
        
        # 시각화 정보 초기화
        batch_vis_info = {
            'original_image': single_image[0].cpu().numpy(),
            'cam': cam_np,
            'object_mask': object_mask,
            'noise_patches': [],
            'defect_patterns': [],
            'patch_coordinates': []
        }
        
        if len(obj_pixels[0]) > 0:
            object_area = np.sum(object_mask)
            num_patches = max(1, min(3, int(object_area / (H * W) * 10)))  # 최대 3개까지 패치 생성
            
            noise_mask_feat1 = torch.zeros((1, 1, H // 4, W // 4), device=device)
            
            for patch_idx in range(num_patches):  
                random_idx = random.randint(0, len(obj_pixels[0]) - 1)
                center_y, center_x = obj_pixels[0][random_idx], obj_pixels[1][random_idx]
                
                defect_pattern = random.choice(['spot', 'line', 'area'])
                
                # 시각화 정보 저장
                patch_info = {
                    'center': (center_y, center_x),
                    'defect_pattern': defect_pattern
                }
                
                if defect_pattern == 'spot':
                    # 4x4 크기 패치
                    patch_size = 8
                    half_patch = patch_size // 2
                    
                    target_y1 = max(0, center_y - half_patch)
                    target_y2 = min(H, center_y + half_patch)
                    target_x1 = max(0, center_x - half_patch)
                    target_x2 = min(W, center_x + half_patch)
                    
                    # 다른 부분에서 4x4 잘라오기
                    source_y = random.randint(0, H - patch_size)
                    source_x = random.randint(0, W - patch_size)
                    
                    # 소스 영역이 타겟 영역과 겹치지 않도록 조정
                    while (abs(source_y - center_y) < patch_size and abs(source_x - center_x) < patch_size):
                        source_y = random.randint(0, H - patch_size)
                        source_x = random.randint(0, W - patch_size)
                    
                    # 복사할 영역
                    source_patch = noised_image[0, :, source_y:source_y+patch_size, source_x:source_x+patch_size].clone()
                    
                    # 타겟 영역에 붙이기
                    if target_y2 - target_y1 > 0 and target_x2 - target_x1 > 0:
                        noised_image[0, :, target_y1:target_y2, target_x1:target_x2] = source_patch[:, :target_y2-target_y1, :target_x2-target_x1]
                    
                    patch_info['coordinates'] = (target_y1, target_y2, target_x1, target_x2)
                    patch_info['source_coordinates'] = (source_y, source_y+patch_size, source_x, source_x+patch_size)
                    patch_info['patch_size'] = patch_size
                    patch_info['noise_type'] = 'spot_augmentation'
                
                elif defect_pattern == 'line':
                    # 얇고 긴 선 모양 (width=2~4, length=15~30)
                    line_width = random.randint(2, 4)
                    line_length = random.randint(20, 60)
                    
                    # 수직 또는 수평 선 선택
                    is_vertical = random.choice([True, False])
                    
                    if is_vertical:
                        # 수직선
                        line_y1 = max(0, center_y - line_length // 2)
                        line_y2 = min(H, center_y + line_length // 2)
                        line_x1 = max(0, center_x - line_width // 2)
                        line_x2 = min(W, center_x + line_width // 2)
                        
                        # 소스 영역 선택 (수직선)
                        source_y = random.randint(0, max(1, H - (line_y2 - line_y1)))
                        source_x = random.randint(0, max(1, W - line_width))
                        
                        # 소스 라인 추출
                        source_line = noised_image[0, :, source_y:source_y+(line_y2-line_y1), source_x:source_x+line_width].clone()
                        
                        # 타겟에 붙이기
                        if line_y2 - line_y1 > 0 and line_x2 - line_x1 > 0:
                            noised_image[0, :, line_y1:line_y2, line_x1:line_x2] = source_line[:, :line_y2-line_y1, :line_x2-line_x1]
                        
                        patch_info['coordinates'] = (line_y1, line_y2, line_x1, line_x2)
                        patch_info['source_coordinates'] = (source_y, source_y+(line_y2-line_y1), source_x, source_x+line_width)
                    else:
                        # 수평선
                        line_y1 = max(0, center_y - line_width // 2)
                        line_y2 = min(H, center_y + line_width // 2)
                        line_x1 = max(0, center_x - line_length // 2)
                        line_x2 = min(W, center_x + line_length // 2)
                        
                        # 소스 영역 선택 (수평선)
                        source_y = random.randint(0, max(1, H - line_width))
                        source_x = random.randint(0, max(1, W - (line_x2 - line_x1)))
                        
                        # 소스 라인 추출
                        source_line = noised_image[0, :, source_y:source_y+line_width, source_x:source_x+(line_x2-line_x1)].clone()
                        
                        # 타겟에 붙이기
                        if line_y2 - line_y1 > 0 and line_x2 - line_x1 > 0:
                            noised_image[0, :, line_y1:line_y2, line_x1:line_x2] = source_line[:, :line_y2-line_y1, :line_x2-line_x1]
                        
                        patch_info['coordinates'] = (line_y1, line_y2, line_x1, line_x2)
                        patch_info['source_coordinates'] = (source_y, source_y+line_width, source_x, source_x+(line_x2-line_x1))
                    
                    patch_info['line_width'] = line_width
                    patch_info['line_length'] = line_length
                    patch_info['is_vertical'] = is_vertical
                    patch_info['noise_type'] = 'line_augmentation'
                
                elif defect_pattern == 'area':
                    # 랜덤 크기 사각형 (8x8 ~ 20x20)
                    rect_width = random.randint(20, 40)
                    rect_height = random.randint(20, 40)
                    
                    area_y1 = max(0, center_y - rect_height // 2)
                    area_y2 = min(H, center_y + rect_height // 2)
                    area_x1 = max(0, center_x - rect_width // 2)
                    area_x2 = min(W, center_x + rect_width // 2)
                    
                    actual_height = area_y2 - area_y1
                    actual_width = area_x2 - area_x1
                    
                    # 소스 영역 선택
                    source_y = random.randint(0, max(1, H - actual_height))
                    source_x = random.randint(0, max(1, W - actual_width))
                    
                    # 소스 영역이 타겟 영역과 겹치지 않도록 조정
                    while (abs(source_y - center_y) < actual_height and abs(source_x - center_x) < actual_width):
                        source_y = random.randint(0, max(1, H - actual_height))
                        source_x = random.randint(0, max(1, W - actual_width))
                    
                    # 소스 영역 추출
                    source_area = noised_image[0, :, source_y:source_y+actual_height, source_x:source_x+actual_width].clone()
                    
                    # 타겟에 붙이기
                    if actual_height > 0 and actual_width > 0:
                        noised_image[0, :, area_y1:area_y2, area_x1:area_x2] = source_area
                    
                    patch_info['coordinates'] = (area_y1, area_y2, area_x1, area_x2)
                    patch_info['source_coordinates'] = (source_y, source_y+actual_height, source_x, source_x+actual_width)
                    patch_info['rect_width'] = rect_width
                    patch_info['rect_height'] = rect_height
                    patch_info['noise_type'] = 'area_augmentation'
                
                # noise 마스크 생성 (각 패턴에 맞게)
                y1, y2, x1, x2 = patch_info['coordinates']
                
                # Feature level 좌표로 변환
                y1_feat = max(0, y1 // 4)
                y2_feat = min(H // 4, y2 // 4)
                x1_feat = max(0, x1 // 4)
                x2_feat = min(W // 4, x2 // 4)
                
                if y2_feat > y1_feat and x2_feat > x1_feat:
                    if defect_pattern == 'spot':
                        # 점 모양 마스크 (4x4 정도)
                        noise_mask_feat1[0, 0, y1_feat:y2_feat, x1_feat:x2_feat] = 1.0
                    
                    elif defect_pattern == 'line':
                        # 선 모양 마스크
                        if patch_info['is_vertical']:
                            # 수직선 마스크
                            center_x_feat = (x1_feat + x2_feat) // 2
                            for y_feat in range(y1_feat, y2_feat):
                                for x_offset in range(max(1, patch_info['line_width'] // 4)):
                                    x_feat = center_x_feat + x_offset - patch_info['line_width'] // 8
                                    if 0 <= x_feat < W // 4:
                                        noise_mask_feat1[0, 0, y_feat, x_feat] = 1.0
                        else:
                            # 수평선 마스크
                            center_y_feat = (y1_feat + y2_feat) // 2
                            for x_feat in range(x1_feat, x2_feat):
                                for y_offset in range(max(1, patch_info['line_width'] // 4)):
                                    y_feat = center_y_feat + y_offset - patch_info['line_width'] // 8
                                    if 0 <= y_feat < H // 4:
                                        noise_mask_feat1[0, 0, y_feat, x_feat] = 1.0
                    
                    elif defect_pattern == 'area':
                        # 사각형 영역 마스크
                        noise_mask_feat1[0, 0, y1_feat:y2_feat, x1_feat:x2_feat] = 1.0
                
                # 패치 정보를 시각화 정보에 추가
                batch_vis_info['noise_patches'].append(patch_info)
                batch_vis_info['defect_patterns'].append(defect_pattern)
                batch_vis_info['patch_coordinates'].append((y1, y2, x1, x2))
        else:
            noise_mask_feat1 = torch.zeros((1, 1, H // 4, W // 4), device=device)
        
        # 최종 노이즈된 이미지 저장
        batch_vis_info['noised_image'] = noised_image[0].cpu().numpy()
        batch_vis_info['noise_mask'] = noise_mask_feat1[0, 0].cpu().numpy()
        
        noised_images.append(noised_image)
        noise_masks.append(noise_mask_feat1)
        visualization_info.append(batch_vis_info)
    
    noised_image_batch = torch.cat(noised_images, dim=0)
    noise_mask_batch = torch.cat(noise_masks, dim=0)
    
    # 시각화가 요청된 경우
    if visualize_noise and save_dir is not None:
        visualize_noise_injection(visualization_info, save_dir)
    
    return noised_image_batch, noise_mask_batch


def visualize_noise_injection(visualization_info, save_dir):

    #노이즈 주입 과정 시각화하는 함수
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    os.makedirs(save_dir, exist_ok=True)
    
    for batch_idx, vis_info in enumerate(visualization_info):
        # 이미지 정규화 함수
        def normalize_image(img):
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        # 원본 이미지
        original_img = normalize_image(vis_info['original_image'].transpose(1, 2, 0))
        
        # 노이즈된 이미지
        noised_img = normalize_image(vis_info['noised_image'].transpose(1, 2, 0))
        
        # CAM
        cam_img = vis_info['cam']
        
        # 객체 마스크
        object_mask = vis_info['object_mask']
        
        # 노이즈 마스크
        noise_mask = vis_info['noise_mask']
        
        # 시각화 생성 (더 큰 크기로 변경)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'Noise Injection Visualization - Batch {batch_idx}', fontsize=16)
        
        # 1. 원본 이미지
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. CAM
        axes[0, 1].imshow(cam_img, cmap='jet')
        axes[0, 1].set_title('Grad-CAM++')
        axes[0, 1].axis('off')
        
        # 3. 객체 마스크
        axes[0, 2].imshow(object_mask, cmap='gray')
        axes[0, 2].set_title('Object Mask')
        axes[0, 2].axis('off')
        
        # 4. 노이즈된 이미지
        axes[1, 0].imshow(noised_img)
        axes[1, 0].set_title('Noised Image')
        axes[1, 0].axis('off')
        
        # 5. 노이즈 마스크
        axes[1, 1].imshow(noise_mask, cmap='hot')
        axes[1, 1].set_title('Noise Mask (Feature Level)')
        axes[1, 1].axis('off')
        
        # 6. Object Mask 오버랩 시각화
        axes[1, 2].imshow(noised_img)
        axes[1, 2].imshow(object_mask, alpha=0.3, cmap='Reds')
        axes[1, 2].set_title('Noised Image + Object Mask Overlay')
        axes[1, 2].axis('off')
        
        # 노이즈 패치들을 사각형으로 표시
        for patch_info in vis_info['noise_patches']:
            y1, y2, x1, x2 = patch_info['coordinates']
            pattern = patch_info['defect_pattern']
            
            # 패턴에 따른 색상 설정
            if pattern == 'spot':
                color = 'red'
                linewidth = 3
            elif pattern == 'line':
                color = 'blue'
                linewidth = 3
            elif pattern == 'area':
                color = 'green'
                linewidth = 3
            elif pattern == 'object_replacement':
                color = 'orange'
                linewidth = 4
            elif pattern == 'shape_distortion':
                color = 'purple'
                linewidth = 3
            else:
                color = 'yellow'
                linewidth = 3
            
            # 경계 박스 표시 (투명한 채우기)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=linewidth, edgecolor=color, facecolor='none')
            axes[1, 2].add_patch(rect)
            
            # 패치 정보 텍스트 (박스 바깥에 표시)
            center_y, center_x = patch_info['center']
            # 박스 아래쪽에 텍스트 배치
            text_y = y2 + 5  # 박스 아래쪽에 약간의 간격
            if text_y >= noised_img.shape[0]:  # 이미지 경계를 벗어나면 위쪽에 배치
                text_y = y1 - 5
            
            axes[1, 2].text(center_x, text_y, pattern, 
                           color=color, fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color))
        
        # 노이즈 패치 정보 텍스트
        info_text = f"Total patches: {len(vis_info['noise_patches'])}\n"
        for i, patch_info in enumerate(vis_info['noise_patches']):
            pattern = patch_info['defect_pattern']
            size = patch_info['patch_size']
            info_text += f"Patch {i+1}: {pattern} (size: {size})\n"
        
        # 정보 텍스트를 별도 창에 표시
        fig_info, ax_info = plt.subplots(figsize=(8, 6))
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        ax_info.set_title('Noise Patch Information')
        
        # 저장
        plt.savefig(os.path.join(save_dir, f'noise_visualization_batch_{batch_idx}.png'), 
                   dpi=300, bbox_inches='tight')
        fig_info.savefig(os.path.join(save_dir, f'noise_info_batch_{batch_idx}.png'), 
                        dpi=300, bbox_inches='tight')
        
        # 별도의 오버랩 시각화 생성
        fig_overlap, ax_overlap = plt.subplots(figsize=(12, 8))
        ax_overlap.imshow(noised_img)
        ax_overlap.set_title(f'Noised Image with Noise Patches Boundaries - Batch {batch_idx}', fontsize=14)
        ax_overlap.axis('off')
        
        # 노이즈 패치들을 오버랩
        for patch_info in vis_info['noise_patches']:
            y1, y2, x1, x2 = patch_info['coordinates']
            pattern = patch_info['defect_pattern']
            
            # 패턴에 따른 색상 설정
            if pattern == 'spot':
                color = 'red'
                linewidth = 3
            elif pattern == 'line':
                color = 'blue'
                linewidth = 3
            elif pattern == 'area':
                color = 'green'
                linewidth = 3
            elif pattern == 'object_replacement':
                color = 'orange'
                linewidth = 4
            elif pattern == 'shape_distortion':
                color = 'purple'
                linewidth = 3
            else:
                color = 'yellow'
                linewidth = 3
            
            # 경계 박스 표시 (투명한 채우기)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=linewidth, edgecolor=color, facecolor='none')
            ax_overlap.add_patch(rect)
            
            # 패치 정보 텍스트 (박스 바깥에 표시)
            center_y, center_x = patch_info['center']
            # 박스 아래쪽에 텍스트 배치
            text_y = y2 + 8  # 박스 아래쪽에 약간의 간격
            if text_y >= noised_img.shape[0]:  # 이미지 경계를 벗어나면 위쪽에 배치
                text_y = y1 - 8
            
            ax_overlap.text(center_x, text_y, pattern, 
                           color=color, fontsize=10, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
        
        # 범례 추가
        legend_elements = [
            patches.Patch(color='red', alpha=0.3, label='Spot'),
            patches.Patch(color='blue', alpha=0.4, label='Line'),
            patches.Patch(color='green', alpha=0.3, label='Area'),
            patches.Patch(color='orange', alpha=0.5, label='Object Replacement'),
            patches.Patch(color='purple', alpha=0.4, label='Shape Distortion')
        ]
        ax_overlap.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 오버랩 시각화 저장
        plt.savefig(os.path.join(save_dir, f'noise_overlap_batch_{batch_idx}.png'), 
                   dpi=300, bbox_inches='tight')
        
        # 7. Object Mask 품질 분석 (4번째 열)
        axes[0, 3].imshow(original_img)
        axes[0, 3].imshow(object_mask, alpha=0.4, cmap='Reds')
        axes[0, 3].set_title('Original + Object Mask')
        axes[0, 3].axis('off')
        
        # 8. Object Mask 통계 정보
        object_area_ratio = np.sum(object_mask) / (object_mask.shape[0] * object_mask.shape[1])
        axes[1, 3].text(0.1, 0.8, f'Object Area Ratio: {object_area_ratio:.3f}', 
                        transform=axes[1, 3].transAxes, fontsize=12)
        axes[1, 3].text(0.1, 0.6, f'Object Pixels: {np.sum(object_mask)}', 
                        transform=axes[1, 3].transAxes, fontsize=12)
        axes[1, 3].text(0.1, 0.4, f'Total Pixels: {object_mask.size}', 
                        transform=axes[1, 3].transAxes, fontsize=12)
        axes[1, 3].text(0.1, 0.2, f'Mask Coverage: {len(vis_info["noise_patches"])} patches', 
                        transform=axes[1, 3].transAxes, fontsize=12)
        axes[1, 3].set_title('Object Mask Statistics')
        axes[1, 3].axis('off')
        
        plt.close(fig)
        plt.close(fig_info)
        plt.close(fig_overlap)
        
        print(f"Noise visualization saved to {save_dir}")

if __name__ == '__main__':
    main()
    