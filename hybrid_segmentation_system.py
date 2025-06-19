#!/usr/bin/env python3
"""
Hybrid Segmentation System - PSPNet/PSANet ve SegFormer Entegre Sistem
Bu sistem hem klasik PSPNet/PSANet modellerini hem de modern SegFormer modelini destekler.
Öğrenci okul rotalarındaki yeşil alan maruziyetini analiz eder.
"""

import os
import logging
import argparse
import sys
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO
import time
import warnings
from sklearn.cluster import KMeans
import openpyxl
from pathlib import Path

# SegFormer imports
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Disable warnings
warnings.filterwarnings('ignore')
cv2.ocl.setUseOpenCL(False)

# Path configurations
GRAY_PATH = '/n/data2/hms/dbmi/patel/beacon/europe/segs/'
IMGS_PATH = '/n/data2/hms/dbmi/patel/beacon/europe/euimages/'
CHUNK_PATH = '/n/data2/hms/dbmi/patel/beacon/europe/metadata/'
CHUNK_PATH_SEG = '/n/data2/hms/dbmi/patel/beacon/europe/metadata/eu_meta_seg/'

class HybridSegmentationSystem:
    """Hibrit segmentasyon sistemi - PSPNet/PSANet ve SegFormer modellerini destekler"""
    3 
    def __init__(self, model_type='segformer', api_key=None, config_path=None):
        """
        Initialize hybrid system
        
        Args:
            model_type: 'segformer', 'pspnet', or 'psanet'
            api_key: Google Street View API key (SegFormer için)
            config_path: PSPNet/PSANet config path
        """
        self.model_type = model_type
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = self.get_logger()
        
        # Gece fotoğrafı tespit parametreleri
        self.night_detection_threshold = 50
        self.blue_dominance_threshold = 1.2
        
        self.setup_directories()
        
        if model_type == 'segformer':
            self.load_segformer_model()
        else:
            if config_path is None:
                raise ValueError("Config path required for PSPNet/PSANet models")
            self.load_classic_model(config_path)
    
    def get_logger(self):
        """Logger oluştur"""
        logger_name = "hybrid-segmentation"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
        
        return logger
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        self.directories = {
            'images': 'street_view_images',
            'analysis': 'analysis_results',
            'visualizations': 'visualizations',
            'reports': 'reports',
            'masks': 'vegetation_masks',
            'segmentation_output': 'segmentation_output'
        }
        
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def load_segformer_model(self):
        """SegFormer modelini yükle"""
        self.logger.info("🤖 SegFormer modeli yükleniyor...")
        try:
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"✅ SegFormer model yüklendi ({self.device})")
        except Exception as e:
            self.logger.error(f"❌ SegFormer model yükleme hatası: {e}")
            raise
    
    def load_classic_model(self, config_path):
        """PSPNet/PSANet modelini yükle"""
        self.args = self.load_config(config_path)
        self.check_config(self.args)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.args.test_gpu)
        
        self.logger.info(self.args)
        self.logger.info("=> Creating classic model ...")
        self.logger.info("Classes: {}".format(self.args.classes))
        
        # Model parametreleri
        self.value_scale = 255
        self.mean = [0.485, 0.456, 0.406]
        self.mean = [item * self.value_scale for item in self.mean]
        self.std = [0.229, 0.224, 0.225]
        self.std = [item * self.value_scale for item in self.std]
        self.colors = np.loadtxt(self.args.colors_path).astype('uint8')
        
        # Model yükleme
        if self.args.arch == 'psp':
            from model.pspnet import PSPNet
            model = PSPNet(layers=self.args.layers, classes=self.args.classes, 
                          zoom_factor=self.args.zoom_factor, pretrained=False)
        elif self.args.arch == 'psa':
            from model.psanet import PSANet
            model = PSANet(layers=self.args.layers, classes=self.args.classes, 
                          zoom_factor=self.args.zoom_factor, compact=self.args.compact,
                          shrink_factor=self.args.shrink_factor, mask_h=self.args.mask_h, 
                          mask_w=self.args.mask_w, normalization_factor=self.args.normalization_factor, 
                          psa_softmax=self.args.psa_softmax, pretrained=False)
        
        self.model = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        
        if os.path.isfile(self.args.model_path):
            self.logger.info("=> Loading checkpoint '{}'".format(self.args.model_path))
            checkpoint = torch.load(self.args.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("=> Loaded checkpoint '{}'".format(self.args.model_path))
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.args.model_path))
    
    def load_config(self, config_path):
        """Config dosyasını yükle"""
        # Bu fonksiyon config dosyasından parametreleri yükler
        # Basit bir implementasyon - gerçek config dosyanızı kullanın
        class Args:
            def __init__(self):
                # Varsayılan değerler - config dosyanızdan yükleyin
                self.arch = 'psp'  # or 'psa'
                self.layers = 50
                self.classes = 150
                self.zoom_factor = 8
                self.test_gpu = [0]
                self.model_path = 'path/to/your/model.pth'
                self.colors_path = 'config/cityscapes_colors.txt'
                self.base_size = 2048
                self.test_h = 713
                self.test_w = 713
                self.scales = [1.0]
                # PSANet specific
                self.compact = False
                self.shrink_factor = 2
                self.mask_h = None
                self.mask_w = None
                self.normalization_factor = 1.0
                self.psa_softmax = True
        
        return Args()
    
    def check_config(self, args):
        """Config parametrelerini kontrol et"""
        assert args.classes > 1
        assert args.zoom_factor in [1, 2, 4, 8]
        
        if args.arch == 'psp':
            assert (args.test_h - 1) % 8 == 0 and (args.test_w - 1) % 8 == 0
        elif args.arch == 'psa':
            if args.compact:
                args.mask_h = (args.test_h - 1) // (8 * args.shrink_factor) + 1
                args.mask_w = (args.test_w - 1) // (8 * args.shrink_factor) + 1
            else:
                assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
                if args.mask_h is None and args.mask_w is None:
                    args.mask_h = 2 * ((args.test_h - 1) // (8 * args.shrink_factor) + 1) - 1
                    args.mask_w = 2 * ((args.test_w - 1) // (8 * args.shrink_factor) + 1) - 1
    
    def download_street_view_image(self, coords, angle, location_id, size="640x640", fov=90, reject_night=True):
        """Street View görüntüsü indir"""
        if not self.api_key:
            self.logger.error("API key gerekli")
            return None, None
            
        url = f"https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': size,
            'location': coords,
            'fov': fov,
            'heading': angle,
            'pitch': 0,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if len(response.content) > 1000:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                if reject_night:
                    location_info = f"Lokasyon {location_id}, Açı {angle}°"
                    is_acceptable, night_info = self.filter_daylight_images(image, location_info)
                    
                    if not is_acceptable:
                        return None, night_info
                
                # Görüntü kalitesini artır
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                
                return image, None
        except Exception as e:
            self.logger.error(f"⚠️ Görüntü indirme hatası: {e}")
        
        return None, None
    
    def filter_daylight_images(self, image, location_info=""):
        """Gece fotoğraflarını filtrele"""
        try:
            img_array = np.array(image)
            
            # 1. Ortalama parlaklık kontrolü
            avg_brightness = np.mean(img_array)
            
            # 2. RGB kanallarının dağılımı
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # 3. Mavi baskınlık oranı (gece görüntülerinde mavi ton artar)
            blue_dominance = b_mean / (r_mean + 1e-6)
            
            # 4. Kontrast analizi
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            
            # Karar verme
            is_night = (avg_brightness < self.night_detection_threshold or 
                       blue_dominance > self.blue_dominance_threshold or
                       contrast < 20)
            
            night_info = {
                'location': location_info,
                'avg_brightness': float(avg_brightness),
                'blue_dominance': float(blue_dominance),
                'contrast': float(contrast),
                'is_night': is_night,
                'reject_reason': []
            }
            
            if avg_brightness < self.night_detection_threshold:
                night_info['reject_reason'].append(f"Düşük parlaklık ({avg_brightness:.1f} < {self.night_detection_threshold})")
            
            if blue_dominance > self.blue_dominance_threshold:
                night_info['reject_reason'].append(f"Mavi baskınlık ({blue_dominance:.2f} > {self.blue_dominance_threshold})")
            
            if contrast < 20:
                night_info['reject_reason'].append(f"Düşük kontrast ({contrast:.1f} < 20)")
            
            return not is_night, night_info
            
        except Exception as e:
            self.logger.error(f"Gece tespit hatası: {e}")
            return True, {'error': str(e)}
    
    def analyze_image_segformer(self, image):
        """SegFormer ile görüntü analizi"""
        try:
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predicted_map = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            ).argmax(dim=1).squeeze().cpu().numpy()
            
            # Cityscapes sınıfları
            classes = {
                'vegetation': 8,  # Bitki örtüsü
                'sky': 10,       # Gökyüzü
                'building': 2,   # Binalar
                'road': 0,       # Yol
                'sidewalk': 1,   # Kaldırım
                'car': 13,       # Arabalar
                'person': 11,    # İnsanlar
                'pole': 5,       # Direkler
                'fence': 4       # Çitler
            }
            
            analysis = {}
            total_pixels = predicted_map.size
            
            for class_name, class_id in classes.items():
                mask = (predicted_map == class_id)
                pixel_count = np.sum(mask)
                percentage = (pixel_count / total_pixels) * 100
                analysis[class_name] = {
                    'pixels': int(pixel_count),
                    'percentage': float(percentage),
                    'mask': mask
                }
            
            # Yeşil alan detayları
            vegetation_mask = analysis['vegetation']['mask']
            mask_uint8 = (vegetation_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            green_regions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    green_regions.append({
                        'id': i,
                        'area': float(area),
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'center': (int(x + w/2), int(y + h/2)),
                        'aspect_ratio': float(w/h) if h > 0 else 0
                    })
            
            analysis['green_regions'] = sorted(green_regions, key=lambda x: x['area'], reverse=True)
            analysis['num_green_regions'] = len(green_regions)
            analysis['largest_green_area'] = green_regions[0]['area'] if green_regions else 0
            
            # GVI hesaplama
            gvi = analysis['vegetation']['percentage']
            analysis['gvi'] = gvi
            
            # Bellek temizleme
            del inputs, outputs, logits, predicted_map
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ SegFormer analiz hatası: {e}")
            return None
    
    def analyze_image_classic(self, image_path, gray_path):
        """PSPNet/PSANet ile görüntü analizi"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            prediction = np.zeros((h, w, self.args.classes), dtype=float)
            
            for scale in self.args.scales:
                long_size = round(scale * self.args.base_size)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size/float(h)*w)
                else:
                    new_h = round(long_size/float(w)*h)
                
                image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += self.scale_process(self.model.eval(), image_scale, self.args.classes, 
                                               self.args.test_h, self.args.test_w, h, w, self.mean, self.std)
            
            prediction = np.argmax(prediction, axis=2)
            gray = np.uint8(prediction)
            
            # Sınıf yüzdelerini hesapla
            perclist = []
            for i in range(self.args.classes):
                perc = np.sum(prediction == i) / (prediction.shape[1] * prediction.shape[0])
                perclist.append(perc)
            
            # Gray image kaydet
            cv2.imwrite(gray_path, gray)
            self.logger.info("=> Prediction saved in {}".format(gray_path))
            
            return perclist
            
        except Exception as e:
            self.logger.error(f"❌ Classic model analiz hatası: {e}")
            return None
    
    def net_process(self, model, image, mean, std=None, flip=True):
        """Network processing for classic models"""
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if std is None:
            for t, m in zip(input, mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, mean, std):
                t.sub_(m).div_(s)
        
        input = input.unsqueeze(0)
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        
        with torch.no_grad():
            output = model(input)
        
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        
        output = F.softmax(output, dim=1)
        
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output
    
    def scale_process(self, model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
        """Scale processing for classic models"""
        ori_h, ori_w, _ = image.shape
        pad_h = max(crop_h - ori_h, 0)
        pad_w = max(crop_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, 
                                     pad_w_half, pad_w - pad_w_half, 
                                     cv2.BORDER_CONSTANT, value=mean)
        
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(crop_h * stride_rate))
        stride_w = int(np.ceil(crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
        
        prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + crop_h, new_h)
                s_h = e_h - crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + crop_w, new_w)
                s_w = e_w - crop_w
                
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(model, image_crop, mean, std)
        
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction
    
    def create_visualization(self, image, analysis, save_path=None):
        """Görselleştirme oluştur"""
        if self.model_type == 'segformer':
            return self.create_segformer_visualization(image, analysis, save_path)
        else:
            return self.create_classic_visualization(image, analysis, save_path)
    
    def create_segformer_visualization(self, image, analysis, save_path=None):
        """SegFormer için görselleştirme"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SegFormer - Detaylı Yeşil Alan Analizi', fontsize=16, fontweight='bold')
        
        # 1. Orijinal görüntü
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Orijinal Görüntü')
        axes[0, 0].axis('off')
        
        # 2. Yeşil alanlar overlay
        img_array = np.array(image)
        overlay = np.zeros_like(img_array)
        if analysis and 'vegetation' in analysis:
            vegetation_mask = analysis['vegetation']['mask']
            overlay[vegetation_mask] = [0, 255, 0]  # Yeşil renk
            
            combined = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
            axes[0, 1].imshow(combined)
            axes[0, 1].set_title(f'Yeşil Alanlar (GVI: {analysis["gvi"]:.1f}%)')
            axes[0, 1].axis('off')
        
        # 3. Sınıf dağılımı
        if analysis:
            class_names = []
            percentages = []
            colors = []
            
            class_colors = {
                'vegetation': '#228B22', 'sky': '#87CEEB', 'building': '#8B4513',
                'road': '#696969', 'sidewalk': '#D3D3D3', 'car': '#FF4500',
                'person': '#FFB6C1', 'pole': '#2F4F4F', 'fence': '#DEB887'
            }
            
            for class_name in ['vegetation', 'building', 'road', 'sky', 'sidewalk']:
                if class_name in analysis and analysis[class_name]['percentage'] > 0.5:
                    class_names.append(class_name.title())
                    percentages.append(analysis[class_name]['percentage'])
                    colors.append(class_colors.get(class_name, '#808080'))
            
            if class_names:
                axes[0, 2].pie(percentages, labels=class_names, colors=colors, autopct='%1.1f%%')
                axes[0, 2].set_title('Sınıf Dağılımı')
        
        # 4. Yeşil bölge analizi
        if analysis and analysis.get('green_regions'):
            img_regions = img_array.copy()
            for i, region in enumerate(analysis['green_regions'][:5]):  # İlk 5 bölge
                x, y, w, h = region['bbox']
                cv2.rectangle(img_regions, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_regions, f"{i+1}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            axes[1, 0].imshow(img_regions)
            axes[1, 0].set_title(f'Yeşil Bölgeler ({analysis["num_green_regions"]} adet)')
            axes[1, 0].axis('off')
        
        # 5. İstatistikler
        axes[1, 1].axis('off')
        if analysis:
            stats_text = f"""
            📊 YEŞIL ALAN ANALİZİ
            
            🌿 GVI Skoru: {analysis.get('gvi', 0):.1f}%
            
            🌳 Yeşil Bölge Sayısı: {analysis.get('num_green_regions', 0)}
            
            📏 En Büyük Yeşil Alan: {analysis.get('largest_green_area', 0):.0f} piksel
            
            🏢 Bina Oranı: {analysis.get('building', {}).get('percentage', 0):.1f}%
            
            🛣️ Yol Oranı: {analysis.get('road', {}).get('percentage', 0):.1f}%
            
            ☁️ Gökyüzü Oranı: {analysis.get('sky', {}).get('percentage', 0):.1f}%
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # 6. Renk analizi
        if analysis:
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            green_pixels = img_hsv[analysis['vegetation']['mask']]
            
            if len(green_pixels) > 0:
                axes[1, 2].scatter(green_pixels[:, 0], green_pixels[:, 1], 
                                 c=green_pixels[:, 2], cmap='viridis', alpha=0.6, s=1)
                axes[1, 2].set_xlabel('Hue')
                axes[1, 2].set_ylabel('Saturation')
                axes[1, 2].set_title('Yeşil Piksel Dağılımı (HSV)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Görselleştirme kaydedildi: {save_path}")
        
        return fig
    
    def create_classic_visualization(self, image_path, analysis, save_path=None):
        """Classic model için görselleştirme"""
        # Bu fonksiyon classic model çıktıları için görselleştirme yapar
        # Implementasyon gerektiğinde eklenebilir
        pass
    
    def analyze_location(self, location, reject_night=True):
        """Lokasyon analizi - her iki model türü için"""
        if self.model_type == 'segformer':
            return self.analyze_location_segformer(location, reject_night)
        else:
            return self.analyze_location_classic(location)
    
    def analyze_location_segformer(self, location, reject_night=True):
        """SegFormer ile lokasyon analizi"""
        self.logger.info(f"📍 Lokasyon: {location['name']} - {location['coords']}")
        
        angles = [0, 90, 180, 270]  # 4 yön
        results = {}
        rejected_images = []
        total_gvi = 0
        valid_images = 0
        
        for angle in angles:
            self.logger.info(f"  📸 {angle}° açısından görüntü indiriliyor...")
            
            image, night_info = self.download_street_view_image(
                location['coords'], angle, location['id'], reject_night=reject_night
            )
            
            if image is None:
                if night_info:
                    rejected_images.append({
                        'angle': angle,
                        'reason': 'night_detection',
                        'details': night_info
                    })
                    self.logger.info(f"  🌙 {angle}° - Gece fotoğrafı reddedildi")
                else:
                    self.logger.info(f"  ❌ {angle}° - Görüntü indirilemedi")
                
                results[angle] = {
                    'image_available': False,
                    'gvi': 0,
                    'error': night_info or 'Download failed'
                }
                continue
            
            # Görüntüyü kaydet
            image_filename = f"{location['id']}_{angle}.jpg"
            image_path = os.path.join(self.directories['images'], image_filename)
            image.save(image_path, quality=95)
            
            # Analiz et
            analysis = self.analyze_image_segformer(image)
            
            if analysis:
                gvi = analysis.get('gvi', 0)
                total_gvi += gvi
                valid_images += 1
                
                self.logger.info(f"  ✅ {angle}° - GVI: {gvi:.1f}%")
                
                # Görselleştirme oluştur
                viz_filename = f"{location['id']}_{angle}_analysis.png"
                viz_path = os.path.join(self.directories['visualizations'], viz_filename)
                self.create_visualization(image, analysis, viz_path)
                
                results[angle] = {
                    'image_available': True,
                    'gvi': gvi,
                    'analysis': analysis,
                    'image_path': image_path,
                    'viz_path': viz_path
                }
            else:
                self.logger.info(f"  ❌ {angle}° - Analiz başarısız")
                results[angle] = {
                    'image_available': True,
                    'gvi': 0,
                    'error': 'Analysis failed'
                }
        
        # Lokasyon özeti
        avg_gvi = total_gvi / valid_images if valid_images > 0 else 0
        
        summary = {
            'location_id': location['id'],
            'location_name': location['name'],
            'coordinates': location['coords'],
            'total_images': len(angles),
            'valid_images': valid_images,
            'rejected_images': len(rejected_images),
            'average_gvi': avg_gvi,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"  📊 Ortalama GVI: {avg_gvi:.1f}% ({valid_images}/{len(angles)} görüntü)")
        
        return {
            'location_info': summary,
            'angles': results,
            'rejected_images': rejected_images,
            'summary': summary
        }
    
    def analyze_location_classic(self, location):
        """Classic model ile lokasyon analizi"""
        # Bu fonksiyon batch processing için tasarlanmış
        # Metadata CSV dosyasından lokasyonları okur
        pass
    
    def process_batch_classic(self, chunk_code):
        """Classic model için batch processing"""
        if os.path.exists(CHUNK_PATH_SEG + f'{chunk_code}_seg.csv'):
            self.logger.info('Bu chunk için segmentasyon zaten tamamlanmış.')
            return
        
        start = datetime.now()
        
        # Metadata yükle
        if os.path.exists(CHUNK_PATH_SEG + f'{chunk_code}_seg_m.csv'):
            metadf = pd.read_csv(CHUNK_PATH_SEG + f'{chunk_code}_seg_m.csv')
        else:
            metadf = pd.read_csv(CHUNK_PATH + f'meta_lau_split_500_{chunk_code}.csv')
            metadf['segmented'] = 0
            new_columns = pd.DataFrame(0, index=metadf.index, columns=[f'c{nclass}' for nclass in range(1, 151)])
            metadf = pd.concat([metadf, new_columns], axis=1)
        
        # Her satır için işlem yap
        for index, row in metadf.iterrows():
            if row['segmented'] == 0:
                # Görüntü yollarını oluştur
                COUNTRYPATH = os.path.join(IMGS_PATH, str(row['country']))
                SUBPATH = os.path.join(COUNTRYPATH, str(row['lau_code']))
                imageid = str(row['lau_code']) + '_' + str(int(row['grid_id'])).zfill(6) + '_' + str(int(row['gsv_year']))[-2:]
                
                # Segmentasyon çıktı klasörlerini oluştur
                COUNTRYPATH_SEG = os.path.join(GRAY_PATH, str(row['country']))
                if not os.path.exists(COUNTRYPATH_SEG):
                    os.mkdir(COUNTRYPATH_SEG)
                SUBPATH_SEG = os.path.join(COUNTRYPATH_SEG, str(row['lau_code']))
                if not os.path.exists(SUBPATH_SEG):
                    os.mkdir(SUBPATH_SEG)
                
                plist = np.zeros(shape=(4, 150))
                flag_img = 0
                
                # 4 yön için işlem yap
                for head, hid in zip([0, 1, 2, 3], ['N', 'E', 'S', 'W']):
                    img_path = os.path.join(SUBPATH, str(imageid) + str(hid) + '.png')
                    gray_path = os.path.join(SUBPATH_SEG, str(imageid) + str(hid) + '_seg.png')
                    
                    try:
                        plist_ = self.analyze_image_classic(img_path, gray_path)
                        if plist_ is not None:
                            plist[head] = plist_
                        else:
                            flag_img = 1
                    except Exception as e:
                        self.logger.error(f"Görüntü işleme hatası {img_path}: {e}")
                        flag_img = 1
                
                # Sonuçları kaydet
                if flag_img == 0:
                    metadf.loc[index, 'segmented'] = 1
                    metadf.iloc[index, 12:162] = list(np.mean(plist, axis=0))
                else:
                    metadf.loc[index, 'segmented'] = 0
                    metadf.iloc[index, 12:162] = 0
                
                # Ara kayıt
                if index % 100 == 0:
                    metadf.to_csv(CHUNK_PATH_SEG + f'{chunk_code}_seg_m.csv', index=False)
        
        # Final kayıt
        metadf.to_csv(CHUNK_PATH_SEG + f'{chunk_code}_seg.csv', index=False)
        if os.path.exists(CHUNK_PATH_SEG + f'{chunk_code}_seg_m.csv'):
            os.remove(CHUNK_PATH_SEG + f'{chunk_code}_seg_m.csv')
        
        end = datetime.now()
        self.logger.info(f"Toplam işlem süresi {len(metadf)} görüntü için: {end - start}")
    
    def create_comprehensive_report(self, results):
        """Kapsamlı rapor oluştur"""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV raporu
        csv_path = os.path.join(self.directories['reports'], f'green_analysis_report_{timestamp}.csv')
        
        csv_data = []
        for result in results:
            location_info = result['location_info']
            for angle, angle_data in result['angles'].items():
                csv_data.append({
                    'location_id': location_info['location_id'],
                    'location_name': location_info['location_name'],
                    'coordinates': location_info['coordinates'],
                    'angle': angle,
                    'gvi': angle_data.get('gvi', 0),
                    'image_available': angle_data.get('image_available', False),
                    'analysis_timestamp': location_info['analysis_timestamp']
                })
        
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        self.logger.info(f"CSV raporu kaydedildi: {csv_path}")
    
    def load_student_routes(self, excel_path):
        """Excel dosyasından öğrenci rotalarını yükle"""
        try:
            df = pd.read_excel(excel_path)
            self.logger.info(f"📊 Excel dosyası yüklendi: {len(df)} koordinat noktası")
            
            # Veriyi student kodlarına göre grupla
            routes = {}
            for code in df['code'].unique():
                student_data = df[df['code'] == code].sort_values('point_id')
                routes[code] = {
                    'coordinates': [],
                    'point_count': len(student_data)
                }
                
                for _, row in student_data.iterrows():
                    # x = longitude, y = latitude formatında
                    coord = f"{row['y']},{row['x']}"  # lat,lng format
                    routes[code]['coordinates'].append({
                        'point_id': row['point_id'],
                        'coords': coord,
                        'lat': row['y'],
                        'lng': row['x']
                    })
            
            self.logger.info(f"🎒 {len(routes)} öğrenci rotası yüklendi")
            for code, route in routes.items():
                self.logger.info(f"   {code}: {route['point_count']} koordinat noktası")
            
            return routes
            
        except Exception as e:
            self.logger.error(f"❌ Excel dosyası yükleme hatası: {e}")
            return None
    
    def calculate_route_green_exposure(self, route_data, student_code, angles_per_point=4):
        """Öğrenci rotasının yeşil alan maruziyetini hesapla"""
        self.logger.info(f"🎒 {student_code} rotası analiz ediliyor...")
        
        coordinates = route_data['coordinates']
        total_gvi = 0
        valid_points = 0
        point_analyses = []
        rejected_images_total = 0
        
        for i, point in enumerate(coordinates, 1):
            self.logger.info(f"  📍 Nokta {point['point_id']}: {point['coords']}")
            
            # Her koordinat noktası için bir lokasyon objesi oluştur
            location = {
                'id': f"{student_code}_point_{point['point_id']}",
                'name': f"{student_code} - Nokta {point['point_id']}",
                'coords': point['coords']
            }
            
            # Bu nokta için yeşil alan analizi yap
            point_result = self.analyze_location_segformer(location, reject_night=True)
            
            if point_result['summary']['valid_images'] > 0:
                point_gvi = point_result['summary']['average_gvi']
                total_gvi += point_gvi
                valid_points += 1
                
                self.logger.info(f"    ✅ Nokta {point['point_id']} GVI: {point_gvi:.1f}%")
            else:
                self.logger.info(f"    ❌ Nokta {point['point_id']}: Geçerli görüntü yok")
            
            point_analyses.append(point_result)
            rejected_images_total += len(point_result.get('rejected_images', []))
        
        # Rota özeti
        average_route_gvi = total_gvi / valid_points if valid_points > 0 else 0
        
        route_summary = {
            'student_code': student_code,
            'total_points': len(coordinates),
            'analyzed_points': len(point_analyses),
            'valid_points': valid_points,
            'average_route_gvi': average_route_gvi,
            'total_gvi': total_gvi,
            'rejected_images_count': rejected_images_total,
            'analysis_timestamp': datetime.now().isoformat(),
            'route_length_km': self.calculate_route_distance(coordinates)
        }
        
        self.logger.info(f"🎯 {student_code} rota özeti:")
        self.logger.info(f"   📊 Ortalama GVI: {average_route_gvi:.1f}%")
        self.logger.info(f"   📏 Rota uzunluğu: {route_summary['route_length_km']:.2f} km")
        self.logger.info(f"   ✅ Geçerli nokta: {valid_points}/{len(coordinates)}")
        
        return {
            'summary': route_summary,
            'point_analyses': point_analyses
        }
    
    def calculate_route_distance(self, coordinates):
        """Rota koordinatları arasındaki mesafeyi hesapla (Haversine formula)"""
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Dünya yarıçapı (km)
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lat2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        total_distance = 0
        for i in range(len(coordinates) - 1):
            lat1, lng1 = coordinates[i]['lat'], coordinates[i]['lng']
            lat2, lng2 = coordinates[i+1]['lat'], coordinates[i+1]['lng']
            
            distance = haversine(lat1, lng1, lat2, lng2)
            total_distance += distance
        
        return total_distance
    
    def analyze_all_student_routes(self, excel_path):
        """Tüm öğrenci rotalarını analiz et"""
        # Excel dosyasından rotaları yükle
        routes = self.load_student_routes(excel_path)
        if not routes:
            return None
        
        all_results = []
        
        # Her öğrenci rotasını analiz et
        for student_code, route_data in routes.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎒 {student_code} öğrencisinin rotası analiz ediliyor...")
            
            route_result = self.calculate_route_green_exposure(route_data, student_code)
            all_results.append(route_result)
        
        # Karşılaştırmalı rapor oluştur
        self.create_student_comparison_report(all_results)
        
        return all_results
    
    def create_student_comparison_report(self, all_results):
        """Öğrenciler arası karşılaştırmalı rapor oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV raporu - Öğrenci özetleri
        summary_data = []
        for result in all_results:
            summary = result['summary']
            summary_data.append({
                'Öğrenci_Kodu': summary['student_code'],
                'Toplam_Nokta': summary['total_points'],
                'Geçerli_Nokta': summary['valid_points'],
                'Ortalama_GVI_%': round(summary['average_route_gvi'], 2),
                'Toplam_GVI': round(summary['total_gvi'], 2),
                'Rota_Uzunluk_km': round(summary['route_length_km'], 3),
                'GVI_per_km': round(summary['average_route_gvi'] / summary['route_length_km'] if summary['route_length_km'] > 0 else 0, 2),
                'Reddedilen_Görüntü': summary['rejected_images_count'],
                'Analiz_Tarihi': summary['analysis_timestamp']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.directories['reports'], f'student_routes_summary_{timestamp}.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        # 2. Detaylı CSV raporu - Her nokta için
        detailed_data = []
        for result in all_results:
            student_code = result['summary']['student_code']
            for point_analysis in result['point_analyses']:
                location_info = point_analysis['location_info']
                for angle, angle_data in point_analysis['angles'].items():
                    detailed_data.append({
                        'Öğrenci_Kodu': student_code,
                        'Nokta_ID': location_info['location_id'],
                        'Koordinat': location_info['coordinates'],
                        'Açı': angle,
                        'GVI_%': round(angle_data.get('gvi', 0), 2),
                        'Görüntü_Mevcut': angle_data.get('image_available', False),
                        'Hata': angle_data.get('error', None)
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(self.directories['reports'], f'student_routes_detailed_{timestamp}.csv')
        detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
        
        # 3. Görselleştirme raporu
        self.create_route_visualization(all_results, timestamp)
        
        self.logger.info(f"📊 Raporlar oluşturuldu:")
        self.logger.info(f"   📄 Özet rapor: {summary_csv_path}")
        self.logger.info(f"   📄 Detaylı rapor: {detailed_csv_path}")
    
    def create_route_visualization(self, all_results, timestamp):
        """Öğrenci rotaları için görselleştirme oluştur"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Öğrenci Okul Rotaları - Yeşil Alan Maruziyeti Analizi', fontsize=16, fontweight='bold')
        
        # Veri hazırlama
        student_codes = []
        avg_gvis = []
        route_lengths = []
        valid_points = []
        gvi_per_km = []
        
        for result in all_results:
            summary = result['summary']
            student_codes.append(summary['student_code'])
            avg_gvis.append(summary['average_route_gvi'])
            route_lengths.append(summary['route_length_km'])
            valid_points.append(summary['valid_points'])
            if summary['route_length_km'] > 0:
                gvi_per_km.append(summary['average_route_gvi'] / summary['route_length_km'])
            else:
                gvi_per_km.append(0)
        
        # 1. Ortalama GVI karşılaştırması
        axes[0, 0].bar(student_codes, avg_gvis, color='green', alpha=0.7)
        axes[0, 0].set_title('Öğrenci Rotalarında Ortalama GVI (%)')
        axes[0, 0].set_xlabel('Öğrenci Kodu')
        axes[0, 0].set_ylabel('GVI (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Rota uzunluğu vs GVI
        axes[0, 1].scatter(route_lengths, avg_gvis, s=100, alpha=0.7, color='darkgreen')
        for i, code in enumerate(student_codes):
            axes[0, 1].annotate(code, (route_lengths[i], avg_gvis[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_title('Rota Uzunluğu vs Ortalama GVI')
        axes[0, 1].set_xlabel('Rota Uzunluğu (km)')
        axes[0, 1].set_ylabel('Ortalama GVI (%)')
        
        # 3. GVI/km karşılaştırması
        axes[1, 0].bar(student_codes, gvi_per_km, color='forestgreen', alpha=0.7)
        axes[1, 0].set_title('Kilometre Başına GVI Maruziyeti')
        axes[1, 0].set_xlabel('Öğrenci Kodu')
        axes[1, 0].set_ylabel('GVI/km')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Geçerli nokta sayısı
        axes[1, 1].bar(student_codes, valid_points, color='olive', alpha=0.7)
        axes[1, 1].set_title('Başarıyla Analiz Edilen Nokta Sayısı')
        axes[1, 1].set_xlabel('Öğrenci Kodu')
        axes[1, 1].set_ylabel('Geçerli Nokta Sayısı')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        viz_path = os.path.join(self.directories['visualizations'], f'student_routes_comparison_{timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"   📊 Görselleştirme: {viz_path}")
        
        # Her öğrenci için ayrı rota haritası oluştur
        self.create_individual_route_maps(all_results, timestamp)
    
    def create_individual_route_maps(self, all_results, timestamp):
        """Her öğrenci için ayrı rota haritası oluştur"""
        for result in all_results:
            student_code = result['summary']['student_code']
            
            # Rota noktalarını ve GVI değerlerini topla
            points = []
            gvi_values = []
            
            for point_analysis in result['point_analyses']:
                if point_analysis['summary']['valid_images'] > 0:
                    avg_gvi = point_analysis['summary']['average_gvi']
                    # Koordinatları parse et
                    coords_str = point_analysis['location_info']['coordinates']
                    lat, lng = map(float, coords_str.split(','))
                    
                    points.append((lat, lng))
                    gvi_values.append(avg_gvi)
            
            if points:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Rota çizgisi
                lats, lngs = zip(*points)
                ax.plot(lngs, lats, 'b-', linewidth=2, alpha=0.7, label='Rota')
                
                # GVI değerlerine göre renkli noktalar
                scatter = ax.scatter(lngs, lats, c=gvi_values, cmap='RdYlGn', 
                                   s=100, alpha=0.8, edgecolors='black', linewidth=1)
                
                # Colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('GVI (%)', rotation=270, labelpad=20)
                
                # Nokta numaraları
                for i, (lat, lng) in enumerate(points):
                    ax.annotate(f'{i+1}', (lng, lat), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                ax.set_title(f'{student_code} - Okul Rotası Yeşil Alan Maruziyeti')
                ax.set_xlabel('Boylam (Longitude)')
                ax.set_ylabel('Enlem (Latitude)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                route_map_path = os.path.join(self.directories['visualizations'], 
                                            f'{student_code}_route_map_{timestamp}.png')
                plt.savefig(route_map_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"   🗺️ {student_code} rota haritası: {route_map_path}")


def main():
    """Ana fonksiyon"""
    print("🌿 Hybrid Segmentation System - Öğrenci Rota Analizi")
    print("=" * 60)
    
    # Analiz tipini seç
    print("📊 Analiz Tipi Seçin:")
    print("1. Öğrenci rotalarını analiz et (Excel dosyasından)")
    print("2. Tekli lokasyon analizi")
    print("3. Batch processing (Classic model)")
    
    analysis_choice = input("Seçiminiz (1/2/3): ").strip()
    
    # Model tipini seç
    model_type = input("Model tipini seçin (segformer/pspnet/psanet): ").lower()
    if model_type not in ['segformer', 'pspnet', 'psanet']:
        print("Geçersiz model tipi!")
        return
    
    # API key (SegFormer için)
    api_key = None
    if model_type == 'segformer':
        api_key = input("Google Street View API key girin: ").strip()
        if not api_key:
            print("API key gerekli!")
            return
    
    # Config path (Classic modeller için)
    config_path = None
    if model_type in ['pspnet', 'psanet']:
        config_path = input("Config dosya yolu girin: ").strip()
        if not config_path or not os.path.exists(config_path):
            print("Geçerli config dosyası gerekli!")
            return
    
    # Sistemi başlat
    try:
        system = HybridSegmentationSystem(
            model_type=model_type,
            api_key=api_key,
            config_path=config_path
        )
        
        if analysis_choice == "1":
            # Öğrenci rotalarını analiz et
            excel_path = input("Excel dosya yolu girin (varsayılan: coordinates.xlsx): ").strip()
            if not excel_path:
                excel_path = "coordinates.xlsx"
            
            if not os.path.exists(excel_path):
                print(f"❌ Excel dosyası bulunamadı: {excel_path}")
                return
            
            print(f"\n🎒 Öğrenci rotaları analiz ediliyor: {excel_path}")
            all_results = system.analyze_all_student_routes(excel_path)
            
            if all_results:
                print(f"\n🎉 {len(all_results)} öğrenci rotası analiz edildi!")
                print("📊 Raporlar 'reports' klasöründe oluşturuldu")
                print("📈 Görselleştirmeler 'visualizations' klasöründe oluşturuldu")
            
        elif analysis_choice == "2":
            # Tekli lokasyon analizi
            if model_type == 'segformer':
                LOCATIONS = [
                    {"id": "loc_001", "name": "Test Lokasyonu 1", "coords": "40.7128,-74.0060"},
                    {"id": "loc_002", "name": "Test Lokasyonu 2", "coords": "34.0522,-118.2437"},
                ]
                
                reject_night = input("Gece fotoğraflarını reddet? (y/n): ").lower() == 'y'
                
                all_results = []
                for i, location in enumerate(LOCATIONS, 1):
                    print(f"\n[{i}/{len(LOCATIONS)}] Lokasyon analizi başlatılıyor...")
                    result = system.analyze_location(location, reject_night=reject_night)
                    all_results.append(result)
                
                # Rapor oluştur
                system.create_comprehensive_report(all_results)
            
        elif analysis_choice == "3":
            # Classic model modu - batch processing
            if model_type in ['pspnet', 'psanet']:
                chunk_code = int(input("Chunk code girin: "))
                system.process_batch_classic(chunk_code)
            else:
                print("❌ Batch processing sadece PSPNet/PSANet modelleri için kullanılabilir!")
        
        else:
            print("❌ Geçersiz seçim!")
            return
        
        print("\n🎉 Analiz tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
