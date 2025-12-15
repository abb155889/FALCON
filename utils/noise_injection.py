import torch 
import cv2
import torch.nn.functional as F
import numpy as np
import random

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Forward/Backward hook 등록"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                handle_f = module.register_forward_hook(forward_hook)
                handle_b = module.register_backward_hook(backward_hook)
                self.hooks.extend([handle_f, handle_b])
                break
    
    def generate_cam_lightweight(self, input_tensor, score_method='mean'):
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break
        
        if target_module is None:
            return torch.zeros((1, 1, 64, 64), device=input_tensor.device)
        
        original_requires_grad = {}
        for param in target_module.parameters():
            original_requires_grad[id(param)] = param.requires_grad
            param.requires_grad_(True)
        
        self.model.eval()
        
        try:
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                if 'layer1' in self.target_layer_name:
                    target_output = output[0]
                elif 'layer2' in self.target_layer_name:
                    target_output = output[1]
                elif 'layer3' in self.target_layer_name:
                    target_output = output[2]
                else:
                    target_output = output[0]
            else:
                target_output = output
            
            if score_method == 'mean':
                score = torch.mean(target_output)
            elif score_method == 'max':
                score = torch.max(target_output)
            else:  
                variance = torch.var(target_output, dim=1, keepdim=True)
                weighted_score = torch.sum(target_output * variance) / torch.sum(variance)
                score = weighted_score
            
            self.model.zero_grad()
            score.backward(retain_graph=False)
            
            if self.gradients is None or self.activations is None:
                return torch.zeros((1, 1, target_output.shape[2], target_output.shape[3]), 
                                 device=input_tensor.device)
            
            gradients = self.gradients
            activations = self.activations
            
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            
            alpha_denom = grad_2.mul(2.0)
            alpha_denom.add_(torch.sum(activations * grad_3, dim=[2, 3], keepdim=True))
            alpha = grad_2.div(alpha_denom + 1e-7)
            
            weights = torch.sum(alpha * F.relu(gradients), dim=[2, 3], keepdim=True)

            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
 
            B, _, H, W = cam.shape
            cam_flat = cam.view(B, -1)
            cam_min = torch.min(cam_flat, dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            cam_max = torch.max(cam_flat, dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
            
            return cam
            
        finally:
            for param in target_module.parameters():
                param.requires_grad_(original_requires_grad[id(param)])
            
            if hasattr(self, 'gradients') and self.gradients is not None:
                del self.gradients
            if hasattr(self, 'activations') and self.activations is not None:
                del self.activations
            self.gradients = None
            self.activations = None
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if hasattr(self, 'gradients'):
            del self.gradients
        if hasattr(self, 'activations'):
            del self.activations
        self.gradients = None
        self.activations = None

def adaptive_gradcam_noise(teacher, image, 
                          multiple_layers=['layer1', 'layer2', 'layer3'], 
                          ensemble_weights=[0.5, 0, 0.5],
                          visualize=False, save_dir=None, ###노이즈 주입 이미지 시각화 옵션, 기본 설정은 꺼둠 
                          ):
    
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
     
    if visualize and save_dir is not None:
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
    
        if visualize and save_dir is not None:
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
            num_patches = max(1, min(3, int(object_area / (H * W) * 10)))  
            
            noise_mask_feat1 = torch.zeros((1, 1, H // 4, W // 4), device=device)
            
            for patch_idx in range(num_patches):  
                random_idx = random.randint(0, len(obj_pixels[0]) - 1)
                center_y, center_x = obj_pixels[0][random_idx], obj_pixels[1][random_idx]
                
                defect_pattern = random.choice(['spot', 'line', 'area'])

                patch_info = {
                    'center': (center_y, center_x),
                    'defect_pattern': defect_pattern
                }
                
                if defect_pattern == 'spot':
                    patch_size = 8
                    half_patch = patch_size // 2
                    target_y1 = max(0, center_y - half_patch)
                    target_y2 = min(H, center_y + half_patch)
                    target_x1 = max(0, center_x - half_patch)
                    target_x2 = min(W, center_x + half_patch)
 
                    source_y = random.randint(0, H - patch_size)
                    source_x = random.randint(0, W - patch_size)
                    
                    while (abs(source_y - center_y) < patch_size and abs(source_x - center_x) < patch_size):
                        source_y = random.randint(0, H - patch_size)
                        source_x = random.randint(0, W - patch_size)

                    source_patch = noised_image[0, :, source_y:source_y+patch_size, source_x:source_x+patch_size].clone()
                    if target_y2 - target_y1 > 0 and target_x2 - target_x1 > 0:
                        noised_image[0, :, target_y1:target_y2, target_x1:target_x2] = source_patch[:, :target_y2-target_y1, :target_x2-target_x1]

                    patch_info['coordinates'] = (target_y1, target_y2, target_x1, target_x2)
                    patch_info['source_coordinates'] = (source_y, source_y+patch_size, source_x, source_x+patch_size)
                    patch_info['patch_size'] = patch_size
                    patch_info['noise_type'] = 'spot_augmentation'

                elif defect_pattern == 'line':
                    line_width = random.randint(2, 4)
                    line_length = random.randint(20, 60)
                    is_vertical = random.choice([True, False])
                    
                    if is_vertical:
                        line_y1 = max(0, center_y - line_length // 2)
                        line_y2 = min(H, center_y + line_length // 2)
                        line_x1 = max(0, center_x - line_width // 2)
                        line_x2 = min(W, center_x + line_width // 2)
                        source_y = random.randint(0, max(1, H - (line_y2 - line_y1)))
                        source_x = random.randint(0, max(1, W - line_width))
                        source_line = noised_image[0, :, source_y:source_y+(line_y2-line_y1), source_x:source_x+line_width].clone()
                        
                        if line_y2 - line_y1 > 0 and line_x2 - line_x1 > 0:
                            noised_image[0, :, line_y1:line_y2, line_x1:line_x2] = source_line[:, :line_y2-line_y1, :line_x2-line_x1]
                        
                        patch_info['coordinates'] = (line_y1, line_y2, line_x1, line_x2)
                        patch_info['source_coordinates'] = (source_y, source_y+(line_y2-line_y1), source_x, source_x+line_width)
                    
                    else:
                        line_y1 = max(0, center_y - line_width // 2)
                        line_y2 = min(H, center_y + line_width // 2)
                        line_x1 = max(0, center_x - line_length // 2)
                        line_x2 = min(W, center_x + line_length // 2)
 
                        source_y = random.randint(0, max(1, H - line_width))
                        source_x = random.randint(0, max(1, W - (line_x2 - line_x1)))
 
                        source_line = noised_image[0, :, source_y:source_y+line_width, source_x:source_x+(line_x2-line_x1)].clone()

                        if line_y2 - line_y1 > 0 and line_x2 - line_x1 > 0:
                            noised_image[0, :, line_y1:line_y2, line_x1:line_x2] = source_line[:, :line_y2-line_y1, :line_x2-line_x1]

                        patch_info['coordinates'] = (line_y1, line_y2, line_x1, line_x2)
                        patch_info['source_coordinates'] = (source_y, source_y+line_width, source_x, source_x+(line_x2-line_x1))
                    
                    patch_info['line_width'] = line_width
                    patch_info['line_length'] = line_length
                    patch_info['is_vertical'] = is_vertical
                    patch_info['noise_type'] = 'line_augmentation'

                elif defect_pattern == 'area':
                    rect_width = random.randint(20, 40)
                    rect_height = random.randint(20, 40)
                    
                    area_y1 = max(0, center_y - rect_height // 2)
                    area_y2 = min(H, center_y + rect_height // 2)
                    area_x1 = max(0, center_x - rect_width // 2)
                    area_x2 = min(W, center_x + rect_width // 2)
                    
                    actual_height = area_y2 - area_y1
                    actual_width = area_x2 - area_x1

                    source_y = random.randint(0, max(1, H - actual_height))
                    source_x = random.randint(0, max(1, W - actual_width))
                    
                    while (abs(source_y - center_y) < actual_height and abs(source_x - center_x) < actual_width):
                        source_y = random.randint(0, max(1, H - actual_height))
                        source_x = random.randint(0, max(1, W - actual_width))

                    source_area = noised_image[0, :, source_y:source_y+actual_height, source_x:source_x+actual_width].clone()

                    if actual_height > 0 and actual_width > 0:
                        noised_image[0, :, area_y1:area_y2, area_x1:area_x2] = source_area
                    
                    patch_info['coordinates'] = (area_y1, area_y2, area_x1, area_x2)
                    patch_info['source_coordinates'] = (source_y, source_y+actual_height, source_x, source_x+actual_width)
                    patch_info['rect_width'] = rect_width
                    patch_info['rect_height'] = rect_height
                    patch_info['noise_type'] = 'area_augmentation'

                y1, y2, x1, x2 = patch_info['coordinates']

                y1_feat = max(0, y1 // 4)
                y2_feat = min(H // 4, y2 // 4)
                x1_feat = max(0, x1 // 4)
                x2_feat = min(W // 4, x2 // 4)
                
                if y2_feat > y1_feat and x2_feat > x1_feat:
                    if defect_pattern == 'spot':
                        noise_mask_feat1[0, 0, y1_feat:y2_feat, x1_feat:x2_feat] = 1.0
                    
                    elif defect_pattern == 'line':
                        if patch_info['is_vertical']:
                            center_x_feat = (x1_feat + x2_feat) // 2
                            for y_feat in range(y1_feat, y2_feat):
                                for x_offset in range(max(1, patch_info['line_width'] // 4)):
                                    x_feat = center_x_feat + x_offset - patch_info['line_width'] // 8
                                    if 0 <= x_feat < W // 4:
                                        noise_mask_feat1[0, 0, y_feat, x_feat] = 1.0
                        else:
                            center_y_feat = (y1_feat + y2_feat) // 2
                            for x_feat in range(x1_feat, x2_feat):
                                for y_offset in range(max(1, patch_info['line_width'] // 4)):
                                    y_feat = center_y_feat + y_offset - patch_info['line_width'] // 8
                                    if 0 <= y_feat < H // 4:
                                        noise_mask_feat1[0, 0, y_feat, x_feat] = 1.0
                    
                    elif defect_pattern == 'area':
                        noise_mask_feat1[0, 0, y1_feat:y2_feat, x1_feat:x2_feat] = 1.0
                
                if visualize:
                    batch_vis_info['noise_patches'].append(patch_info)
                    batch_vis_info['defect_patterns'].append(defect_pattern)
                    batch_vis_info['patch_coordinates'].append((y1, y2, x1, x2))
        else:
            noise_mask_feat1 = torch.zeros((1, 1, H // 4, W // 4), device=device)
        
        if visualize:     
            batch_vis_info['noised_image'] = noised_image[0].cpu().numpy()
            batch_vis_info['noise_mask'] = noise_mask_feat1[0, 0].cpu().numpy()
            visualization_info.append(batch_vis_info)
        
        noised_images.append(noised_image)
        noise_masks.append(noise_mask_feat1)
    
    noised_image_batch = torch.cat(noised_images, dim=0)
    noise_mask_batch = torch.cat(noise_masks, dim=0)
    
    if visualize and save_dir is not None:
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
