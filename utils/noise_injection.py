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
                          ensemble_weights=[0.5, 0, 0.5]):
    
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
        else:
            noise_mask_feat1 = torch.zeros((1, 1, H // 4, W // 4), device=device)
        
        noised_images.append(noised_image)
        noise_masks.append(noise_mask_feat1)
    
    noised_image_batch = torch.cat(noised_images, dim=0)
    noise_mask_batch = torch.cat(noise_masks, dim=0)
    
    return noised_image_batch, noise_mask_batch
