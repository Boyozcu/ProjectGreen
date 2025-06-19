#!/usr/bin/env python3
"""
ProjectGreen - Yeşil Alan Maruziyet Analiz Sistemi
Öğrencilerin okul rotalarındaki yeşil alan maruziyetini 3 farklı yöntemle hesaplar

Bu sistem şu amaçlar için tasarlanmıştır:
1. Excel'den öğrenci koordinatlarını okuma
2. Street View görüntülerini indirme
3. 3 farklı yöntemle yeşil alan analizi:
   - Yöntem 1: SegFormer Transformer modeli ile semantik segmentasyon
   - Yöntem 2: HSV renk aralığı ile geleneksel bilgisayar görüsü
   - Yöntem 3: Hibrit yaklaşım (SegFormer + renk filtreleme)

Google Colab uyumlu tasarım
"""

import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2
import os
import json
from io import BytesIO
import time
from datetime import datetime
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import openpyxl
from pathlib import Path
import concurrent.futures
from threading import Lock
import gc

warnings.filterwarnings('ignore')

# ================================================================================================
# KONFİGÜRASYON AYARLARI - Bu bölümü ihtiyacınıza göre düzenleyin
# ================================================================================================

# Google Street View API Anahtarı
GOOGLE_API_KEY = "YOUR_GOOGLE_STREET_VIEW_API_KEY_HERE"

# Excel dosyası yolu
EXCEL_FILE_PATH = "coordinates.xlsx"

# Görüntü parametreleri
IMAGE_SIZE = "640x640"  # Street View görüntü boyutu
FIELD_OF_VIEW = 90      # Görüş alanı (derece)
VIEWING_ANGLES = [0, 90, 180, 270]  # Analiz edilecek açılar

# Analiz parametreleri
NIGHT_DETECTION_THRESHOLD = 50      # Gece fotoğrafı tespit eşiği (0-255)
BLUE_DOMINANCE_THRESHOLD = 1.2      # Mavi baskınlık oranı
ENABLE_NIGHT_FILTERING = True       # Gece fotoğraflarını filtrele

# Model ayarları
SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
DEVICE_PREFERENCE = "auto"  # "auto", "cuda", "cpu"

# Çıktı klasörleri
OUTPUT_DIRECTORIES = {
    'images': 'street_view_images',
    'analysis': 'analysis_results', 
    'visualizations': 'visualizations',
    'reports': 'reports',
    'masks': 'vegetation_masks',
    'method_comparison': 'method_comparison'
}

# Renk filtreleme parametreleri (HSV)
HSV_GREEN_RANGES = {
    'light_green': {'lower': [35, 40, 40], 'upper': [85, 255, 255]},
    'dark_green': {'lower': [85, 40, 40], 'upper': [95, 255, 255]},
    'yellowish_green': {'lower': [25, 40, 40], 'upper': [35, 255, 255]}
}

# İşlem parametreleri
MAX_CONCURRENT_DOWNLOADS = 5       # Eş zamanlı indirme sayısı
REQUEST_DELAY = 0.1                # İstekler arası bekleme süresi (saniye)
ENABLE_MEMORY_CLEANUP = True       # Bellek temizleme

# Görselleştirme ayarları
FIGURE_DPI = 300                   # Grafik çözünürlüğü
FIGURE_SIZE = (15, 10)            # Grafik boyutu
COLOR_PALETTE = ['#2E8B57', '#228B22', '#32CD32', '#90EE90']

# Hibrit yöntem ağırlıkları
HYBRID_WEIGHTS = {
    'segformer': 0.7,
    'hsv': 0.3
}

# ================================================================================================

class GreenExposureAnalyzer:
    """
    Yeşil Alan Maruziyet Analiz Sistemi
    3 farklı yöntemle öğrenci rotalarındaki yeşil alan maruziyetini hesaplar
    """
    
    def __init__(self, api_key=None, excel_path=None):
        """
        Sistem başlatma
        
        Args:
            api_key (str): Google Street View API anahtarı (None ise konfigürasyondan alınır)
            excel_path (str): Koordinatlar Excel dosyasının yolu (None ise konfigürasyondan alınır)
        """
        print("🌱 ProjectGreen - Yeşil Alan Maruziyet Analiz Sistemi")
        print("=" * 60)
        
        # Konfigürasyondan parametreleri al
        self.api_key = api_key or GOOGLE_API_KEY
        self.excel_path = excel_path or EXCEL_FILE_PATH
        
        # Cihaz seçimi
        if DEVICE_PREFERENCE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = DEVICE_PREFERENCE
            
        self.results_lock = Lock()
        
        # Sonuçları saklayacak veri yapıları
        self.analysis_results = []
        self.method_comparison = {}
        self.student_routes = {}
        
        # Konfigürasyondan analiz parametrelerini al
        self.viewing_angles = VIEWING_ANGLES
        self.image_size = IMAGE_SIZE
        self.fov = FIELD_OF_VIEW
        self.night_threshold = NIGHT_DETECTION_THRESHOLD
        self.enable_night_filtering = ENABLE_NIGHT_FILTERING
        
        # Klasör yapısını oluştur
        self.setup_directories()
        
        # Koordinatları yükle
        self.load_coordinates()
        
        # AI modelini yükle
        self.load_segmentation_model()
        
        print(f"✅ Sistem hazır - {len(self.coordinates)} koordinat noktası yüklendi")
        print(f"🖥️  Cihaz: {self.device}")
        print(f"📁 Excel dosyası: {self.excel_path}")
        print(f"🔑 API anahtarı: {'✓ Ayarlandı' if self.api_key != 'YOUR_GOOGLE_STREET_VIEW_API_KEY_HERE' else '❌ Ayarlanmadı'}")
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        self.directories = OUTPUT_DIRECTORIES.copy()
        
        for dir_name, dir_path in self.directories.items():
            Path(dir_path).mkdir(exist_ok=True)
            
        print("📁 Klasör yapısı oluşturuldu")
    
    def load_coordinates(self):
        """Excel dosyasından koordinatları yükle"""
        try:
            print(f"📊 Koordinatlar yükleniyor: {self.excel_path}")
            self.coordinates = pd.read_excel(self.excel_path)
            
            # Gerekli sütunları kontrol et
            required_columns = ['code', 'x', 'y', 'point_id']
            missing_columns = [col for col in required_columns if col not in self.coordinates.columns]
            
            if missing_columns:
                raise ValueError(f"Eksik sütunlar: {missing_columns}")
            
            # Öğrenci rotalarını grupla
            self.student_routes = self.coordinates.groupby('code')
            
            print(f"✅ {len(self.coordinates)} koordinat yüklendi")
            print(f"👥 {len(self.student_routes)} farklı öğrenci rotası")
            
            # Koordinat özetini göster
            for student_code, route_data in self.student_routes:
                print(f"   - {student_code}: {len(route_data)} nokta")
                
        except Exception as e:
            print(f"❌ Koordinat yükleme hatası: {e}")
            raise
    
    def load_segmentation_model(self):
        """SegFormer modelini yükle (Yöntem 1 için)"""
        try:
            print("🤖 SegFormer modeli yükleniyor...")
            
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
                SEGFORMER_MODEL
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_MODEL
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Cityscapes sınıf etiketleri
            self.cityscapes_labels = {
                0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
                9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
            }
            
            # Yeşil alan sınıfları
            self.vegetation_classes = [8, 9]  # vegetation, terrain
            
            print("✅ SegFormer modeli yüklendi")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def download_street_view_image(self, lat, lon, angle, location_id, point_id):
        """
        Street View görüntüsü indir
        
        Args:
            lat (float): Enlem
            lon (float): Boylam  
            angle (int): Görüş açısı (0, 90, 180, 270)
            location_id (str): Öğrenci kodu
            point_id (int): Nokta ID
            
        Returns:
            PIL.Image or None: İndirilen görüntü
        """
        coords = f"{lat},{lon}"
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': self.image_size,
            'location': coords,
            'fov': self.fov,
            'heading': angle,
            'pitch': 0,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if len(response.content) > 1000:  # Geçerli görüntü kontrolü
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Görüntü kalitesini artır
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
                
                # Kaydet
                filename = f"{location_id}_point_{point_id}_{angle}.jpg"
                image_path = os.path.join(self.directories['images'], filename)
                image.save(image_path, quality=95)
                
                return image
            else:
                print(f"⚠️  Geçersiz görüntü: {location_id}_point_{point_id}_{angle}")
                return None
                
        except Exception as e:
            print(f"❌ Görüntü indirme hatası: {e}")
            return None
    
    def method1_segformer_analysis(self, image):
        """
        Yöntem 1: SegFormer Transformer modeli ile semantik segmentasyon
        
        Args:
            image (PIL.Image): Analiz edilecek görüntü
            
        Returns:
            dict: Analiz sonuçları
        """
        try:
            # Görüntüyü model için hazırla
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Segmentasyon yap
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Sonuçları işle
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            # En olası sınıfı al
            predicted_segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
            
            # Yeşil alan piksellerini say
            total_pixels = predicted_segmentation.size
            vegetation_pixels = np.sum(np.isin(predicted_segmentation, self.vegetation_classes))
            green_ratio = vegetation_pixels / total_pixels
            
            # Detaylı sınıf analizi
            class_counts = {}
            for class_id in np.unique(predicted_segmentation):
                if class_id in self.cityscapes_labels:
                    class_name = self.cityscapes_labels[class_id]
                    count = np.sum(predicted_segmentation == class_id)
                    class_counts[class_name] = count / total_pixels
            
            # Segmentasyon maskesini kaydet
            mask = np.zeros_like(predicted_segmentation, dtype=np.uint8)
            mask[np.isin(predicted_segmentation, self.vegetation_classes)] = 255
            
            return {
                'method': 'SegFormer',
                'green_ratio': green_ratio,
                'green_percentage': green_ratio * 100,
                'class_distribution': class_counts,
                'mask': mask,
                'segmentation_map': predicted_segmentation
            }
            
        except Exception as e:
            print(f"❌ SegFormer analizi hatası: {e}")
            return {
                'method': 'SegFormer',
                'green_ratio': 0.0,
                'green_percentage': 0.0,
                'class_distribution': {},
                'mask': None,
                'error': str(e)
            }
    
    def method2_hsv_analysis(self, image):
        """
        Yöntem 2: HSV renk aralığı ile geleneksel bilgisayar görüsü
        
        Args:
            image (PIL.Image): Analiz edilecek görüntü
            
        Returns:
            dict: Analiz sonuçları
        """
        try:
            # PIL'den OpenCV formatına çevir
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Konfigürasyondan yeşil renk aralıklarını al
            masks = []
            for range_name, range_values in HSV_GREEN_RANGES.items():
                lower = np.array(range_values['lower'])
                upper = np.array(range_values['upper'])
                mask = cv2.inRange(img_hsv, lower, upper)
                masks.append(mask)
            
            # Maskeleri birleştir
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Gürültüyü temizle
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Yeşil piksel oranını hesapla
            total_pixels = img_array.shape[0] * img_array.shape[1]
            green_pixels = np.sum(combined_mask > 0)
            green_ratio = green_pixels / total_pixels
            
            # Renk dağılımı analizi
            avg_hue = np.mean(img_hsv[:,:,0])
            avg_saturation = np.mean(img_hsv[:,:,1])
            avg_value = np.mean(img_hsv[:,:,2])
            
            return {
                'method': 'HSV Color Range',
                'green_ratio': green_ratio,
                'green_percentage': green_ratio * 100,
                'mask': combined_mask,
                'color_stats': {
                    'avg_hue': avg_hue,
                    'avg_saturation': avg_saturation,
                    'avg_brightness': avg_value
                },
                'green_pixels': green_pixels,
                'total_pixels': total_pixels
            }
            
        except Exception as e:
            print(f"❌ HSV analizi hatası: {e}")
            return {
                'method': 'HSV Color Range',
                'green_ratio': 0.0,
                'green_percentage': 0.0,
                'mask': None,
                'error': str(e)
            }
    
    def method3_hybrid_analysis(self, image):
        """
        Yöntem 3: Hibrit yaklaşım (SegFormer + HSV renk filtreleme)
        
        Args:
            image (PIL.Image): Analiz edilecek görüntü
            
        Returns:
            dict: Analiz sonuçları
        """
        try:
            # Her iki yöntemi uygula
            segformer_result = self.method1_segformer_analysis(image)
            hsv_result = self.method2_hsv_analysis(image)
            
            # Maskeleri birleştir
            if segformer_result['mask'] is not None and hsv_result['mask'] is not None:
                # SegFormer maskesini HSV maskesi ile aynı boyuta getir
                segformer_mask = cv2.resize(segformer_result['mask'], 
                                          (hsv_result['mask'].shape[1], hsv_result['mask'].shape[0]))
                
                # İki maskeyi birleştir (intersection)
                combined_mask = cv2.bitwise_and(segformer_mask, hsv_result['mask'])
                
                # Birleşim (union) maskesi de oluştur
                union_mask = cv2.bitwise_or(segformer_mask, hsv_result['mask'])
                
                # Oranları hesapla
                total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
                intersection_pixels = np.sum(combined_mask > 0)
                union_pixels = np.sum(union_mask > 0)
                
                intersection_ratio = intersection_pixels / total_pixels
                union_ratio = union_pixels / total_pixels
                
                # Ağırlıklı ortalama (konfigürasyondan ağırlıkları al)
                weighted_ratio = (HYBRID_WEIGHTS['segformer'] * segformer_result['green_ratio'] + 
                                HYBRID_WEIGHTS['hsv'] * hsv_result['green_ratio'])
                
                # Güven skoru hesapla
                confidence = min(segformer_result['green_ratio'], hsv_result['green_ratio']) / max(segformer_result['green_ratio'], hsv_result['green_ratio'], 0.001)
                
                return {
                    'method': 'Hybrid (SegFormer + HSV)',
                    'green_ratio': weighted_ratio,
                    'green_percentage': weighted_ratio * 100,
                    'intersection_ratio': intersection_ratio,
                    'union_ratio': union_ratio,
                    'segformer_ratio': segformer_result['green_ratio'],
                    'hsv_ratio': hsv_result['green_ratio'],
                    'confidence_score': confidence,
                    'mask': combined_mask,
                    'union_mask': union_mask,
                    'method_agreement': abs(segformer_result['green_ratio'] - hsv_result['green_ratio'])
                }
            else:
                # Bir maske yoksa, mevcut sonucu kullan
                if segformer_result['mask'] is not None:
                    return segformer_result
                elif hsv_result['mask'] is not None:
                    return hsv_result
                else:
                    return {
                        'method': 'Hybrid (SegFormer + HSV)',
                        'green_ratio': 0.0,
                        'green_percentage': 0.0,
                        'error': 'Both methods failed'
                    }
                    
        except Exception as e:
            print(f"❌ Hibrit analiz hatası: {e}")
            return {
                'method': 'Hybrid (SegFormer + HSV)',
                'green_ratio': 0.0,
                'green_percentage': 0.0,
                'error': str(e)
            }
    
    def analyze_single_image(self, image, location_id, point_id, angle):
        """
        Tek bir görüntüyü 3 yöntemle analiz et
        
        Args:
            image (PIL.Image): Analiz edilecek görüntü
            location_id (str): Öğrenci kodu  
            point_id (int): Nokta ID
            angle (int): Görüş açısı
            
        Returns:
            dict: Tüm yöntemlerin sonuçları
        """
        print(f"🔍 Analiz ediliyor: {location_id}_point_{point_id}_{angle}°")
        
        try:
            # Her 3 yöntemi uygula
            method1_result = self.method1_segformer_analysis(image)
            method2_result = self.method2_hsv_analysis(image)
            method3_result = self.method3_hybrid_analysis(image)
            
            # Sonuçları birleştir
            analysis_result = {
                'location_id': location_id,
                'point_id': point_id,
                'angle': angle,
                'timestamp': datetime.now().isoformat(),
                'methods': {
                    'segformer': method1_result,
                    'hsv': method2_result,
                    'hybrid': method3_result
                },
                'summary': {
                    'segformer_green_percentage': method1_result['green_percentage'],
                    'hsv_green_percentage': method2_result['green_percentage'],
                    'hybrid_green_percentage': method3_result['green_percentage'],
                    'average_green_percentage': (method1_result['green_percentage'] + 
                                               method2_result['green_percentage'] + 
                                               method3_result['green_percentage']) / 3
                }
            }
            
            # Maskeleri kaydet
            self.save_analysis_masks(image, analysis_result)
            
            return analysis_result
        
        except Exception as e:
            print(f"❌ Görüntü analiz hatası: {e}")
            return None
    
    def save_analysis_masks(self, original_image, analysis_result):
        """Analiz maskelerini kaydet"""
        try:
            location_id = analysis_result['location_id']
            point_id = analysis_result['point_id']
            angle = analysis_result['angle']
            
            # Orijinal görüntüyü numpy array'e çevir
            img_array = np.array(original_image)
            
            # Her yöntem için visualizasyon oluştur
            fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
            fig.suptitle(f'Yeşil Alan Analizi - {location_id} Point {point_id} ({angle}°)', fontsize=16)
            
            # Orijinal görüntü
            axes[0, 0].imshow(img_array)
            axes[0, 0].set_title('Orijinal Görüntü')
            axes[0, 0].axis('off')
            
            # Method 1 - SegFormer
            if analysis_result['methods']['segformer']['mask'] is not None:
                axes[0, 1].imshow(analysis_result['methods']['segformer']['mask'], cmap='Greens')
                axes[0, 1].set_title(f"SegFormer\n{analysis_result['methods']['segformer']['green_percentage']:.1f}%")
            axes[0, 1].axis('off')
            
            # Method 2 - HSV
            if analysis_result['methods']['hsv']['mask'] is not None:
                axes[0, 2].imshow(analysis_result['methods']['hsv']['mask'], cmap='Greens')
                axes[0, 2].set_title(f"HSV Renk Analizi\n{analysis_result['methods']['hsv']['green_percentage']:.1f}%")
            axes[0, 2].axis('off')
            
            # Method 3 - Hybrid
            if analysis_result['methods']['hybrid']['mask'] is not None:
                axes[1, 0].imshow(analysis_result['methods']['hybrid']['mask'], cmap='Greens')
                axes[1, 0].set_title(f"Hibrit Yöntem\n{analysis_result['methods']['hybrid']['green_percentage']:.1f}%")
            axes[1, 0].axis('off')
            
            # Karşılaştırma grafiği
            methods = ['SegFormer', 'HSV', 'Hibrit']
            percentages = [
                analysis_result['methods']['segformer']['green_percentage'],
                analysis_result['methods']['hsv']['green_percentage'],
                analysis_result['methods']['hybrid']['green_percentage']
            ]
            
            axes[1, 1].bar(methods, percentages, color=COLOR_PALETTE[:3])
            axes[1, 1].set_title('Yöntem Karşılaştırması')
            axes[1, 1].set_ylabel('Yeşil Alan Yüzdesi (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Özet bilgiler
            summary_text = f"""
            Ortalama: {analysis_result['summary']['average_green_percentage']:.1f}%
            
            Detaylar:
            • SegFormer: {analysis_result['methods']['segformer']['green_percentage']:.1f}%
            • HSV: {analysis_result['methods']['hsv']['green_percentage']:.1f}%
            • Hibrit: {analysis_result['methods']['hybrid']['green_percentage']:.1f}%
            """
            axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='center')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Kaydet
            filename = f"{location_id}_point_{point_id}_{angle}_analysis.png"
            save_path = os.path.join(self.directories['visualizations'], filename)
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            # Maskeleri ayrı ayrı kaydet
            for method_name, method_data in analysis_result['methods'].items():
                if method_data['mask'] is not None:
                    mask_filename = f"{location_id}_point_{point_id}_{angle}_{method_name}_mask.png"
                    mask_path = os.path.join(self.directories['masks'], mask_filename)
                    cv2.imwrite(mask_path, method_data['mask'])
            
        except Exception as e:
            print(f"❌ Mask kaydetme hatası: {e}")
    
    def analyze_student_route(self, student_code):
        """
        Bir öğrencinin tüm rotasını analiz et
        
        Args:
            student_code (str): Öğrenci kodu
            
        Returns:
            dict: Rota analiz sonuçları
        """
        print(f"\n🎒 {student_code} öğrencisinin rotası analiz ediliyor...")
        
        student_data = self.student_routes.get_group(student_code)
        route_results = []
        
        for _, row in student_data.iterrows():
            lat, lon = row['y'], row['x']  # Excel'de x=boylam, y=enlem
            point_id = row['point_id']
            
            point_results = []
            
            # Her açı için analiz
            for angle in self.viewing_angles:
                # Görüntüyü indir
                image = self.download_street_view_image(lat, lon, angle, student_code, point_id)
                
                if image is not None:
                    # Analiz yap
                    analysis_result = self.analyze_single_image(image, student_code, point_id, angle)
                    point_results.append(analysis_result)
                    
                    # Sonucu kaydet
                    with self.results_lock:
                        self.analysis_results.append(analysis_result)
                
                # Rate limiting için kısa bekleme
                time.sleep(REQUEST_DELAY)
            
            if point_results:
                # Nokta özetini hesapla
                point_summary = self.calculate_point_summary(point_results, student_code, point_id, lat, lon)
                route_results.append(point_summary)
            
            print(f"   ✅ Point {point_id} tamamlandı ({len(point_results)} görüntü)")
        
        # Rota özetini hesapla
        route_summary = self.calculate_route_summary(route_results, student_code)
        
        return route_summary
    
    def calculate_point_summary(self, point_results, student_code, point_id, lat, lon):
        """Bir nokta için tüm açıların özetini hesapla"""
        
        # Her yöntem için ortalama hesapla
        segformer_avg = np.mean([r['methods']['segformer']['green_percentage'] for r in point_results])
        hsv_avg = np.mean([r['methods']['hsv']['green_percentage'] for r in point_results])
        hybrid_avg = np.mean([r['methods']['hybrid']['green_percentage'] for r in point_results])
        overall_avg = np.mean([r['summary']['average_green_percentage'] for r in point_results])
        
        return {
            'student_code': student_code,
            'point_id': point_id,
            'latitude': lat,
            'longitude': lon,
            'num_images': len(point_results),
            'method_averages': {
                'segformer': segformer_avg,
                'hsv': hsv_avg,
                'hybrid': hybrid_avg,
                'overall': overall_avg
            },
            'angle_results': point_results
        }
    
    def calculate_route_summary(self, route_results, student_code):
        """Bir öğrencinin tüm rotasının özetini hesapla"""
        
        if not route_results:
            return None
        
        # Tüm noktaların ortalamasını hesapla
        segformer_route_avg = np.mean([p['method_averages']['segformer'] for p in route_results])
        hsv_route_avg = np.mean([p['method_averages']['hsv'] for p in route_results])
        hybrid_route_avg = np.mean([p['method_averages']['hybrid'] for p in route_results])
        overall_route_avg = np.mean([p['method_averages']['overall'] for p in route_results])
        
        # En yüksek ve en düşük maruziyet noktalarını bul
        max_exposure_point = max(route_results, key=lambda x: x['method_averages']['overall'])
        min_exposure_point = min(route_results, key=lambda x: x['method_averages']['overall'])
        
        route_summary = {
            'student_code': student_code,
            'total_points': len(route_results),
            'total_images': sum(p['num_images'] for p in route_results),
            'route_averages': {
                'segformer': segformer_route_avg,
                'hsv': hsv_route_avg,
                'hybrid': hybrid_route_avg,
                'overall': overall_route_avg
            },
            'max_exposure': {
                'point_id': max_exposure_point['point_id'],
                'percentage': max_exposure_point['method_averages']['overall'],
                'coordinates': (max_exposure_point['latitude'], max_exposure_point['longitude'])
            },
            'min_exposure': {
                'point_id': min_exposure_point['point_id'],
                'percentage': min_exposure_point['method_averages']['overall'],
                'coordinates': (min_exposure_point['latitude'], min_exposure_point['longitude'])
            },
            'point_details': route_results
        }
        
        # Rota özetini kaydet
        self.student_routes[student_code] = route_summary
        
        print(f"📊 {student_code} rota özeti:")
        print(f"   • Toplam nokta: {route_summary['total_points']}")
        print(f"   • Toplam görüntü: {route_summary['total_images']}")
        print(f"   • Ortalama yeşil maruziyet: {overall_route_avg:.1f}%")
        print(f"   • En yüksek: Point {max_exposure_point['point_id']} ({max_exposure_point['method_averages']['overall']:.1f}%)")
        print(f"   • En düşük: Point {min_exposure_point['point_id']} ({min_exposure_point['method_averages']['overall']:.1f}%)")
        
        return route_summary
    
    def analyze_all_routes(self):
        """Tüm öğrenci rotalarını analiz et"""
        print("\n🚀 Tüm rotalar analiz ediliyor...")
        print("=" * 60)
        
        all_route_summaries = []
        
        for student_code in self.student_routes.groups.keys():
            try:
                route_summary = self.analyze_student_route(student_code)
                if route_summary:
                    all_route_summaries.append(route_summary)
                    
                # Memory cleanup
                if ENABLE_MEMORY_CLEANUP:
                    gc.collect()
                
            except Exception as e:
                print(f"❌ {student_code} analiz hatası: {e}")
                continue
        
        # Genel karşılaştırma yap
        self.create_comparative_analysis(all_route_summaries)
        
        # Sonuçları kaydet
        self.save_all_results(all_route_summaries)
        
        print(f"\n✅ Analiz tamamlandı! {len(all_route_summaries)} rota işlendi")
        
        return all_route_summaries
    
    def create_comparative_analysis(self, route_summaries):
        """Tüm rotaların karşılaştırmalı analizini oluştur"""
        
        if not route_summaries:
            return
        
        print("\n📈 Karşılaştırmalı analiz oluşturuluyor...")
        
        # Veri hazırlığı
        comparison_data = []
        for route in route_summaries:
            comparison_data.append({
                'Student': route['student_code'],
                'SegFormer': route['route_averages']['segformer'],
                'HSV': route['route_averages']['hsv'],
                'Hybrid': route['route_averages']['hybrid'],
                'Overall': route['route_averages']['overall'],
                'Total_Points': route['total_points'],
                'Max_Exposure': route['max_exposure']['percentage'],
                'Min_Exposure': route['min_exposure']['percentage']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # 1. Yöntem karşılaştırma grafiği
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Yeşil Alan Maruziyet Analizi - Tüm Öğrenciler', fontsize=16, fontweight='bold')
        
        # Bar chart - Yöntem karşılaştırması
        methods = ['SegFormer', 'HSV', 'Hybrid']
        x = np.arange(len(df_comparison))
        width = 0.25
        
        for i, method in enumerate(methods):
            axes[0, 0].bar(x + i*width, df_comparison[method], width, 
                          label=method, alpha=0.8)
        
        axes[0, 0].set_xlabel('Öğrenciler')
        axes[0, 0].set_ylabel('Yeşil Alan Yüzdesi (%)')
        axes[0, 0].set_title('Yöntem Karşılaştırması')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(df_comparison['Student'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot - Yöntem dağılımları
        method_data = [df_comparison['SegFormer'], df_comparison['HSV'], df_comparison['Hybrid']]
        axes[0, 1].boxplot(method_data, labels=methods)
        axes[0, 1].set_ylabel('Yeşil Alan Yüzdesi (%)')
        axes[0, 1].set_title('Yöntem Dağılımları')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot - Max vs Min exposure
        scatter = axes[1, 0].scatter(df_comparison['Min_Exposure'], df_comparison['Max_Exposure'], 
                                   c=df_comparison['Overall'], cmap='Greens', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Min Maruziyet (%)')
        axes[1, 0].set_ylabel('Max Maruziyet (%)')
        axes[1, 0].set_title('Maruziyet Aralığı')
        
        # Öğrenci etiketleri ekle
        for i, txt in enumerate(df_comparison['Student']):
            axes[1, 0].annotate(txt, (df_comparison['Min_Exposure'].iloc[i], 
                                    df_comparison['Max_Exposure'].iloc[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[1, 0], label='Ortalama Maruziyet (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram - Overall distribution
        axes[1, 1].hist(df_comparison['Overall'], bins=10, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(df_comparison['Overall'].mean(), color='red', linestyle='--', 
                         label=f'Ortalama: {df_comparison["Overall"].mean():.1f}%')
        axes[1, 1].set_xlabel('Ortalama Yeşil Alan Yüzdesi (%)')
        axes[1, 1].set_ylabel('Öğrenci Sayısı')
        axes[1, 1].set_title('Maruziyet Dağılımı')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        comparison_path = os.path.join(self.directories['method_comparison'], 'route_comparison.png')
        plt.savefig(comparison_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # 2. İnteraktif harita oluştur
        self.create_interactive_map(route_summaries)
        
        # 3. Detaylı rapor tablosu
        self.create_detailed_report(df_comparison)
    
    def create_interactive_map(self, route_summaries):
        """İnteraktif marita oluştur"""
        try:
            # Merkez koordinatı hesapla
            all_coords = []
            for route in route_summaries:
                for point in route['point_details']:
                    all_coords.append([point['latitude'], point['longitude']])
            
            center_lat = np.mean([coord[0] for coord in all_coords])
            center_lon = np.mean([coord[1] for coord in all_coords])
            
            # Harita oluştur
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, 
                          tiles='OpenStreetMap')
            
            # Her öğrenci için farklı renk
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                     'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                     'gray', 'black', 'lightgray']
            
            for i, route in enumerate(route_summaries):
                color = colors[i % len(colors)]
                student_code = route['student_code']
                
                # Rota noktalarını ekle
                route_coords = []
                for point in route['point_details']:
                    lat, lon = point['latitude'], point['longitude']
                    route_coords.append([lat, lon])
                    
                    # Popup bilgisi hazırla
                    popup_text = f"""
                    <b>{student_code} - Point {point['point_id']}</b><br>
                    <b>Koordinat:</b> {lat:.6f}, {lon:.6f}<br>
                    <b>Ortalama Maruziyet:</b> {point['method_averages']['overall']:.1f}%<br>
                    <b>SegFormer:</b> {point['method_averages']['segformer']:.1f}%<br>
                    <b>HSV:</b> {point['method_averages']['hsv']:.1f}%<br>
                    <b>Hibrit:</b> {point['method_averages']['hybrid']:.1f}%<br>
                    <b>Görüntü Sayısı:</b> {point['num_images']}
                    """
                    
                    # Marker ekle
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
                
                # Rota çizgisini ekle
                if len(route_coords) > 1:
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=3,
                        opacity=0.8,
                        popup=f"{student_code} Rotası"
                    ).add_to(m)
            
            # Haritayı kaydet
            map_path = os.path.join(self.directories['visualizations'], 'interactive_route_map.html')
            m.save(map_path)
            
            print(f"🗺️  İnteraktif harita oluşturuldu: {map_path}")
            
        except Exception as e:
            print(f"❌ Harita oluşturma hatası: {e}")
    
    def create_detailed_report(self, df_comparison):
        """Detaylı rapor tablosu oluştur"""
        
        # İstatistiksel özetler
        summary_stats = df_comparison[['SegFormer', 'HSV', 'Hybrid', 'Overall']].describe()
        
        # Korelasyon matrisi
        correlation_matrix = df_comparison[['SegFormer', 'HSV', 'Hybrid', 'Overall']].corr()
        
        # Rapor oluştur
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # İstatistikler tablosu
        axes[0].axis('tight')
        axes[0].axis('off')
        table_data = summary_stats.round(2).reset_index()
        table = axes[0].table(cellText=table_data.values, colLabels=table_data.columns,
                            cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[0].set_title('İstatistiksel Özetler', fontsize=12, fontweight='bold', pad=20)
        
        # Korelasyon heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='Greens', center=0, 
                   square=True, ax=axes[1], fmt='.3f')
        axes[1].set_title('Yöntemler Arası Korelasyon', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Kaydet
        report_path = os.path.join(self.directories['reports'], 'statistical_report.png')
        plt.savefig(report_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # CSV olarak da kaydet
        csv_path = os.path.join(self.directories['reports'], 'detailed_results.csv')
        df_comparison.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"📊 Detaylı rapor oluşturuldu: {report_path}")
        print(f"📄 CSV raporu oluşturuldu: {csv_path}")
    
    def save_all_results(self, route_summaries):
        """Tüm sonuçları JSON formatında kaydet"""
        
        # Ana sonuç dosyası
        main_results = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'total_students': len(route_summaries),
                'total_analysis_points': sum(r['total_points'] for r in route_summaries),
                'total_images': sum(r['total_images'] for r in route_summaries),
                'methods_used': ['SegFormer', 'HSV Color Range', 'Hybrid'],
                'viewing_angles': self.viewing_angles
            },
            'route_summaries': route_summaries,
            'detailed_results': self.analysis_results
        }
        
        # JSON olarak kaydet
        results_path = os.path.join(self.directories['reports'], 'complete_analysis_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(main_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tüm sonuçlar kaydedildi: {results_path}")
        
        # Özet rapor
        self.print_final_summary(route_summaries)
    
    def print_final_summary(self, route_summaries):
        """Final özet raporu yazdır"""
        
        print("\n" + "="*80)
        print("🌱 PROJECTGREEN - YEŞIL ALAN MARUZİYET ANALİZİ - FINAL RAPOR")
        print("="*80)
        
        if not route_summaries:
            print("❌ Analiz edilecek rota bulunamadı!")
            return
        
        # Genel istatistikler
        total_points = sum(r['total_points'] for r in route_summaries)
        total_images = sum(r['total_images'] for r in route_summaries)
        
        all_overall_scores = [r['route_averages']['overall'] for r in route_summaries]
        avg_exposure = np.mean(all_overall_scores)
        max_exposure = max(all_overall_scores)
        min_exposure = min(all_overall_scores)
        
        print(f"📊 GENEL İSTATİSTİKLER:")
        print(f"   • Analiz edilen öğrenci sayısı: {len(route_summaries)}")
        print(f"   • Toplam analiz noktası: {total_points}")
        print(f"   • Toplam işlenen görüntü: {total_images}")
        print(f"   • Ortalama yeşil maruziyet: {avg_exposure:.1f}%")
        print(f"   • En yüksek maruziyet: {max_exposure:.1f}%")
        print(f"   • En düşük maruziyet: {min_exposure:.1f}%")
        
        print(f"\n👥 ÖĞRENCİ BAZLI SONUÇLAR:")
        for route in sorted(route_summaries, key=lambda x: x['route_averages']['overall'], reverse=True):
            print(f"   {route['student_code']:>3}: {route['route_averages']['overall']:>5.1f}% " +
                  f"({route['total_points']} nokta, {route['total_images']} görüntü)")
        
        # Yöntem karşılaştırması
        all_segformer = [r['route_averages']['segformer'] for r in route_summaries]
        all_hsv = [r['route_averages']['hsv'] for r in route_summaries]
        all_hybrid = [r['route_averages']['hybrid'] for r in route_summaries]
        
        print(f"\n🔬 YÖNTEM KARŞILAŞTIRMASI:")
        print(f"   • SegFormer ortalama: {np.mean(all_segformer):.1f}%")
        print(f"   • HSV Renk ortalama: {np.mean(all_hsv):.1f}%")
        print(f"   • Hibrit ortalama: {np.mean(all_hybrid):.1f}%")
        
        # En yüksek ve en düşük maruziyet noktaları
        all_points = []
        for route in route_summaries:
            for point in route['point_details']:
                all_points.append({
                    'student': route['student_code'],
                    'point_id': point['point_id'],
                    'exposure': point['method_averages']['overall'],
                    'coordinates': (point['latitude'], point['longitude'])
                })
        
        all_points.sort(key=lambda x: x['exposure'], reverse=True)
        
        print(f"\n🏆 EN YÜKSEK YEŞİL MARUZİYET NOKTALARı (İlk 5):")
        for i, point in enumerate(all_points[:5]):
            print(f"   {i+1}. {point['student']} Point {point['point_id']}: {point['exposure']:.1f}%")
        
        print(f"\n📉 EN DÜŞÜK YEŞİL MARUZİYET NOKTALARı (Son 5):")
        for i, point in enumerate(all_points[-5:], 1):
            print(f"   {i}. {point['student']} Point {point['point_id']}: {point['exposure']:.1f}%")
        
        print(f"\n📁 ÇIKTI DOSYALARI:")
        print(f"   • Görüntüler: {self.directories['images']}/")
        print(f"   • Analizler: {self.directories['visualizations']}/")
        print(f"   • Maskeler: {self.directories['masks']}/")
        print(f"   • Raporlar: {self.directories['reports']}/")
        print(f"   • İnteraktif harita: visualizations/interactive_route_map.html")
        
        print("\n" + "="*80)
        print("✅ ANALİZ TAMAMLANDI!")
        print("="*80)

def main():
    """
    Ana fonksiyon - Google Colab'da çalıştırılacak
    """
    
    print("🌱 ProjectGreen - Yeşil Alan Maruziyet Analiz Sistemi")
    print("Google Colab için optimize edilmiş versiyon")
    print("=" * 60)
    
    # Konfigürasyondan ayarları kontrol et
    if GOOGLE_API_KEY == "YOUR_GOOGLE_STREET_VIEW_API_KEY_HERE":
        print("❌ HATA: Google Street View API anahtarını ayarlamanız gerekiyor!")
        print("Lütfen dosyanın başındaki GOOGLE_API_KEY değişkenini güncelleyin.")
        return
    
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ HATA: {EXCEL_FILE_PATH} dosyası bulunamadı!")
        print(f"Lütfen {EXCEL_FILE_PATH} dosyasının mevcut olduğundan emin olun.")
        return
    
    try:
        # Analiz sistemini başlat (konfigürasyon parametreleri otomatik kullanılır)
        analyzer = GreenExposureAnalyzer()
        
        # Tüm rotaları analiz et
        results = analyzer.analyze_all_routes()
        
        print(f"\n🎉 Başarılı! {len(results)} rota analiz edildi.")
        
    except Exception as e:
        print(f"❌ HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
