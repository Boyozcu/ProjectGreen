#!/usr/bin/env python3
"""
Enhanced ProjectGreen - Gelişmiş Yeşil Alan Analiz Sistemi
Bu script, yeşil alan analizini geliştirmiş özelliklerle sunar.
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
import csv
import json
from io import BytesIO
import time
from datetime import datetime
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class EnhancedGreenAnalyzer:
    def __init__(self, api_key):
        """Gelişmiş yeşil alan analizörü"""
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.setup_directories()
        # Gece fotoğrafı tespit parametreleri
        self.night_detection_threshold = 50  # Ortalama parlaklık eşiği (0-255)
        self.blue_dominance_threshold = 1.2  # Mavi baskınlık oranı
        
    def load_model(self):
        """Model yükleme"""
        print("🤖 SegFormer modeli yükleniyor...")
        try:
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model yüklendi ({self.device})")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        self.directories = {
            'images': 'street_view_images',
            'analysis': 'analysis_results',
            'visualizations': 'visualizations',
            'reports': 'reports',
            'masks': 'vegetation_masks'
        }
        
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def download_street_view_image(self, coords, angle, location_id, size="640x640", fov=90, reject_night=True):
        """Street View görüntüsü indir ve filtreleme uygula"""
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
            
            if len(response.content) > 1000:  # Geçerli görüntü kontrolü
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Gece fotoğrafı kontrolü
                if reject_night:
                    location_info = f"Lokasyon {location_id}, Açı {angle}°"
                    is_acceptable, night_info = self.filter_daylight_images(image, location_info)
                    
                    if not is_acceptable:
                        return None, night_info  # Gece fotoğrafı reddedildi
                
                # Görüntü kalitesini artır
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                
                return image, None
        except Exception as e:
            print(f"⚠️ Görüntü indirme hatası: {e}")
        
        return None, None
    
    def get_vegetation_analysis(self, image):
        """Detaylı vegetation analizi"""
        try:
            # Model ile tahmin
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
            
            # Farklı sınıfları analiz et
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
            
            # Bağlı bileşenleri bul
            mask_uint8 = (vegetation_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            green_regions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum alan
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
            
            # Bellek temizleme
            del inputs, outputs, logits, predicted_map
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return analysis
            
        except Exception as e:
            print(f"❌ Analiz hatası: {e}")
            return None
    
    def create_advanced_visualization(self, image, analysis, save_path=None):
        """Gelişmiş görselleştirme"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detaylı Yeşil Alan Analizi', fontsize=16, fontweight='bold')
        
        # 1. Orijinal görüntü
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Orijinal Görüntü')
        axes[0, 0].axis('off')
        
        # 2. Yeşil alanlar overlay
        img_array = np.array(image)
        overlay = np.zeros_like(img_array)
        if analysis and 'vegetation' in analysis:
            vegetation_mask = analysis['vegetation']['mask']
            overlay[vegetation_mask] = [0, 255, 0]
            blended = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
            axes[0, 1].imshow(blended)
            
            gvi = analysis['vegetation']['percentage'] / 100
            axes[0, 1].set_title(f'Yeşil Alanlar (GVI: {gvi:.3f})')
        else:
            axes[0, 1].imshow(image)
            axes[0, 1].set_title('Yeşil Alanlar (Analiz Hatası)')
        axes[0, 1].axis('off')
        
        # 3. Segmentasyon haritası
        if analysis:
            # Sınıf dağılımını göster
            classes = ['vegetation', 'building', 'road', 'sidewalk', 'sky']
            percentages = [analysis.get(cls, {}).get('percentage', 0) for cls in classes]
            colors = ['green', 'gray', 'black', 'lightgray', 'lightblue']
            
            axes[0, 2].pie(percentages, labels=classes, colors=colors, autopct='%1.1f%%')
            axes[0, 2].set_title('Sınıf Dağılımı')
        
        # 4. Yeşil alan konturları
        if analysis and analysis.get('green_regions'):
            contour_img = img_array.copy()
            for region in analysis['green_regions'][:10]:  # İlk 10 alan
                x, y, w, h = region['bbox']
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(contour_img, f"{region['area']:.0f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            axes[1, 0].imshow(contour_img)
            axes[1, 0].set_title(f'Yeşil Bölgeler ({analysis["num_green_regions"]} adet)')
        else:
            axes[1, 0].imshow(image)
            axes[1, 0].set_title('Yeşil Bölgeler')
        axes[1, 0].axis('off')
        
        # 5. Vegetation mask
        if analysis and 'vegetation' in analysis:
            axes[1, 1].imshow(analysis['vegetation']['mask'], cmap='Greens')
            axes[1, 1].set_title('Bitki Örtüsü Maskesi')
        else:
            axes[1, 1].text(0.5, 0.5, 'Mask\nUnavailable', ha='center', va='center')
            axes[1, 1].set_title('Bitki Örtüsü Maskesi')
        axes[1, 1].axis('off')
        
        # 6. İstatistikler
        axes[1, 2].axis('off')
        if analysis and 'vegetation' in analysis:
            stats_text = f"""
📊 ANALİZ İSTATİSTİKLERİ

🌿 Yeşil Alan: {analysis['vegetation']['percentage']:.2f}%
🏢 Binalar: {analysis.get('building', {}).get('percentage', 0):.1f}%
🛣️  Yol: {analysis.get('road', {}).get('percentage', 0):.1f}%
🚶 Kaldırım: {analysis.get('sidewalk', {}).get('percentage', 0):.1f}%
☁️ Gökyüzü: {analysis.get('sky', {}).get('percentage', 0):.1f}%

🔢 Yeşil Bölge Sayısı: {analysis.get('num_green_regions', 0)}
📏 En Büyük Alan: {analysis.get('largest_green_area', 0):.0f} piksel

GVI Skoru: {analysis['vegetation']['percentage']/100:.4f}
            """
        else:
            stats_text = "❌ Analiz yapılamadı"
        
        axes[1, 2].text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Görselleştirme kaydedildi: {save_path}")
        
        plt.show()
        
        return fig
    
    def analyze_location(self, location_data, angles=[0, 90, 180, 270], reject_night=True):
        """Bir lokasyonu tüm açılardan analiz et"""
        location_results = {
            'location_info': location_data,
            'angles': {},
            'summary': {},
            'rejected_images': []  # Reddedilen gece fotoğrafları
        }
        
        print(f"\n📍 Analiz ediliyor: {location_data['name']}")
        
        all_gvi_scores = []
        
        for angle in angles:
            print(f"  🔄 Açı: {angle}°")
            
            # Görüntüyü indir
            image, night_info = self.download_street_view_image(
                location_data['coords'], 
                angle, 
                location_data['id'],
                reject_night=reject_night
            )
            
            if image:
                # Analiz yap
                analysis = self.get_vegetation_analysis(image)
                
                if analysis:
                    gvi = analysis['vegetation']['percentage'] / 100
                    all_gvi_scores.append(gvi)
                    
                    # Sonuçları kaydet
                    location_results['angles'][angle] = {
                        'gvi': gvi,
                        'analysis': analysis,
                        'image_available': True
                    }
                    
                    # Görüntüyü kaydet
                    img_filename = f"location_{location_data['id']}_angle_{angle}.jpg"
                    img_path = os.path.join(self.directories['images'], img_filename)
                    image.save(img_path, quality=95)
                    
                    # Görselleştirme oluştur
                    viz_filename = f"analysis_location_{location_data['id']}_angle_{angle}.png"
                    viz_path = os.path.join(self.directories['visualizations'], viz_filename)
                    self.create_advanced_visualization(image, analysis, viz_path)
                    
                    # Vegetation mask'i kaydet
                    mask_filename = f"mask_location_{location_data['id']}_angle_{angle}.png"
                    mask_path = os.path.join(self.directories['masks'], mask_filename)
                    vegetation_mask = (analysis['vegetation']['mask'] * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, vegetation_mask)
                
                else:
                    location_results['angles'][angle] = {
                        'gvi': 0,
                        'analysis': None,
                        'image_available': True,
                        'error': 'Analysis failed'
                    }
            else:
                # Görüntü indirilemedi veya gece fotoğrafı reddedildi
                error_reason = 'Image download failed'
                if night_info:
                    error_reason = 'Night image rejected'
                    location_results['rejected_images'].append({
                        'angle': angle,
                        'reason': 'night_image',
                        'night_info': night_info
                    })
                    
                location_results['angles'][angle] = {
                    'gvi': 0,
                    'analysis': None,
                    'image_available': False,
                    'error': error_reason
                }
            
            time.sleep(0.2)  # API rate limiting
        
        # Özet istatistikler
        if all_gvi_scores:
            location_results['summary'] = {
                'avg_gvi': np.mean(all_gvi_scores),
                'max_gvi': np.max(all_gvi_scores),
                'min_gvi': np.min(all_gvi_scores),
                'std_gvi': np.std(all_gvi_scores),
                'angles_analyzed': len(all_gvi_scores),
                'total_rejected': len(location_results['rejected_images'])
            }
        else:
            location_results['summary'] = {
                'avg_gvi': 0,
                'max_gvi': 0,
                'min_gvi': 0,
                'std_gvi': 0,
                'angles_analyzed': 0,
                'total_rejected': len(location_results['rejected_images'])
            }
        
        return location_results
    
    def create_comprehensive_report(self, all_results):
        """Kapsamlı rapor oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON raporu
        json_path = os.path.join(self.directories['reports'], f'green_analysis_report_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV raporu
        csv_path = os.path.join(self.directories['reports'], f'green_analysis_summary_{timestamp}.csv')
        csv_data = []
        
        for result in all_results:
            for angle, angle_data in result['angles'].items():
                csv_data.append({
                    'Location_ID': result['location_info']['id'],
                    'Location_Name': result['location_info']['name'],
                    'Coordinates': result['location_info']['coords'],
                    'Angle': angle,
                    'GVI_Score': angle_data['gvi'],
                    'Image_Available': angle_data['image_available'],
                    'Analysis_Success': angle_data.get('analysis') is not None
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Görsel rapor
        self.create_visual_report(all_results, timestamp)
        
        print(f"📊 Raporlar oluşturuldu:")
        print(f"  📄 JSON: {json_path}")
        print(f"  📊 CSV: {csv_path}")
        
        return json_path, csv_path
    
    def create_visual_report(self, all_results, timestamp):
        """Görsel rapor oluştur"""
        # Özet istatistikler
        locations = []
        avg_gvis = []
        
        for result in all_results:
            if result['summary']['angles_analyzed'] > 0:
                locations.append(result['location_info']['name'])
                avg_gvis.append(result['summary']['avg_gvi'])
        
        if not avg_gvis:
            print("❌ Görsel rapor için yeterli veri yok")
            return
        
        # Ana rapor figürü
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('İstiklal Caddesi Yeşil Alan Analizi - Kapsamlı Rapor', fontsize=16, fontweight='bold')
        
        # 1. Lokasyon bazlı GVI skorları
        colors = ['#2E8B57' if gvi > 0.1 else '#CD853F' if gvi > 0.05 else '#DC143C' for gvi in avg_gvis]
        bars = axes[0, 0].bar(range(len(locations)), avg_gvis, color=colors)
        axes[0, 0].set_xlabel('Lokasyonlar')
        axes[0, 0].set_ylabel('Ortalama GVI Skoru')
        axes[0, 0].set_title('Lokasyon Bazlı Yeşil Görünüm Endeksi')
        axes[0, 0].set_xticks(range(len(locations)))
        axes[0, 0].set_xticklabels([loc[:20] + '...' if len(loc) > 20 else loc 
                                   for loc in locations], rotation=45, ha='right')
        
        # Değerleri bar'ların üzerine yaz
        for bar, gvi in zip(bars, avg_gvis):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{gvi:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. GVI dağılımı histogram
        all_gvi_values = []
        for result in all_results:
            for angle_data in result['angles'].values():
                if angle_data['gvi'] > 0:
                    all_gvi_values.append(angle_data['gvi'])
        
        if all_gvi_values:
            axes[0, 1].hist(all_gvi_values, bins=20, color='green', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('GVI Skoru')
            axes[0, 1].set_ylabel('Frekans')
            axes[0, 1].set_title('GVI Skorlarının Dağılımı')
            axes[0, 1].axvline(np.mean(all_gvi_values), color='red', linestyle='--', 
                              label=f'Ortalama: {np.mean(all_gvi_values):.3f}')
            axes[0, 1].legend()
        
        # 3. Açı bazlı analiz
        angle_gvis = {0: [], 90: [], 180: [], 270: []}
        for result in all_results:
            for angle, angle_data in result['angles'].items():
                if angle_data['gvi'] > 0:
                    angle_gvis[angle].append(angle_data['gvi'])
        
        angle_means = [np.mean(gvis) if gvis else 0 for gvis in angle_gvis.values()]
        angles = list(angle_gvis.keys())
        
        axes[1, 0].bar(angles, angle_means, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        axes[1, 0].set_xlabel('Kamera Açısı (Derece)')
        axes[1, 0].set_ylabel('Ortalama GVI Skoru')
        axes[1, 0].set_title('Kamera Açısına Göre Yeşil Alan Yoğunluğu')
        axes[1, 0].set_xticks(angles)
        
        # 4. İstatistik özeti
        axes[1, 1].axis('off')
        if all_gvi_values:
            stats_text = f"""
📊 GENEL İSTATİSTİKLER

🌿 Toplam Analiz Edilen Görüntü: {len(all_gvi_values)}
📍 Toplam Lokasyon: {len(locations)}

📈 GVI İstatistikleri:
   • Ortalama: {np.mean(all_gvi_values):.4f}
   • Medyan: {np.median(all_gvi_values):.4f}
   • Maksimum: {np.max(all_gvi_values):.4f}
   • Minimum: {np.min(all_gvi_values):.4f}
   • Standart Sapma: {np.std(all_gvi_values):.4f}

🏆 En Yeşil Lokasyon:
   {locations[np.argmax(avg_gvis)]}
   (GVI: {max(avg_gvis):.4f})

📉 En Az Yeşil Lokasyon:
   {locations[np.argmin(avg_gvis)]}
   (GVI: {min(avg_gvis):.4f})

📅 Analiz Tarihi: {datetime.now().strftime("%d.%m.%Y %H:%M")}
            """
        else:
            stats_text = "❌ Yeterli veri bulunamadı"
        
        axes[1, 1].text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Raporu kaydet
        report_path = os.path.join(self.directories['reports'], f'visual_report_{timestamp}.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Görsel rapor kaydedildi: {report_path}")

    def test_night_detection(self, test_images_folder=None):
        """Gece tespit fonksiyonunu test etme"""
        print("🌙 Gece Fotoğrafı Tespit Testi")
        print("=" * 40)
        
        if test_images_folder and os.path.exists(test_images_folder):
            # Klasördeki fotoğrafları test et
            image_files = [f for f in os.listdir(test_images_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files[:10]:  # İlk 10 fotoğrafı test et
                img_path = os.path.join(test_images_folder, img_file)
                try:
                    image = Image.open(img_path).convert("RGB")
                    print(f"\n📷 Test ediliyor: {img_file}")
                    
                    night_result = self.is_night_image(image)
                    
                    status = "🌙 GECE" if night_result['is_night'] else "☀️ GÜNDÜZ"
                    print(f"   {status} (Güven: {night_result['confidence']:.2f})")
                    print(f"   Skor: {night_result['night_score']}/6")
                    if night_result['reasons']:
                        print(f"   Nedenler: {', '.join(night_result['reasons'])}")
                        
                except Exception as e:
                    print(f"   ❌ Hata: {e}")
        else:
            # Demo test - Street View'dan bir görüntü çekerek test
            print("Demo test - Street View görüntüsü ile test yapılacak...")
            
            # Akşam saatlerinde bir konum için test
            demo_coords = "41.0369,28.9850"  # Taksim
            test_image, _ = self.download_street_view_image(
                demo_coords, 0, "test", reject_night=False
            )
            
            if test_image:
                print(f"\n📷 Test görüntüsü indirildi")
                night_result = self.is_night_image(test_image)
                
                status = "🌙 GECE" if night_result['is_night'] else "☀️ GÜNDÜZ"
                print(f"   {status} (Güven: {night_result['confidence']:.2f})")
                print(f"   Skor: {night_result['night_score']}/6")
                if night_result['reasons']:
                    print(f"   Nedenler: {', '.join(night_result['reasons'])}")
                
                # Metrikleri göster
                print(f"\n📊 Detaylı Metrikler:")
                for key, value in night_result['metrics'].items():
                    print(f"   • {key}: {value:.3f}")
            else:
                print("❌ Test görüntüsü indirilemedi")
        
        print(f"\n✅ Test tamamlandı")

# Ana lokasyonlar - İstiklal Caddesi
LOCATIONS = [
    {"id": 1, "name": "Taksim Meydanı", "coords": "41.0369,28.9850"},
    {"id": 2, "name": "İstiklal Caddesi Başlangıç", "coords": "41.0365,28.9845"},
    {"id": 3, "name": "Galatasaray Lisesi", "coords": "41.0358,28.9838"},
    {"id": 4, "name": "Çiçek Pasajı", "coords": "41.0355,28.9835"},
    {"id": 5, "name": "Balık Pazarı", "coords": "41.0352,28.9832"},
]

def main():
    """Ana fonksiyon"""
    print("🌱 Enhanced ProjectGreen - Gelişmiş Yeşil Alan Analizi")
    print("🌙 Gece Fotoğrafı Reddetme Özelliği Aktif")
    print("=" * 60)
    
    # API anahtarı (güvenlik için environment variable kullanın)
    API_KEY = "AIzaSyAi9iP8f6EPp86aajanovmLn3QmMUlCZQs"
    
    # Analizörü başlat
    analyzer = EnhancedGreenAnalyzer(API_KEY)
    
    # Gece tespit fonksiyonunu test et (isteğe bağlı)
    test_night_detection = input("🌙 Gece tespit fonksiyonunu test etmek ister misiniz? (y/n): ").lower() == 'y'
    if test_night_detection:
        analyzer.test_night_detection()
        print("\n" + "="*60)
    
    # Gece fotoğrafı reddetme ayarını sor
    reject_night = input("🌙 Gece fotoğraflarını reddetmek ister misiniz? (y/n, varsayılan: y): ").lower()
    reject_night = reject_night != 'n'  # Varsayılan olarak True
    
    if reject_night:
        print("✅ Gece fotoğrafları reddedilecek - Sadece gündüz fotoğrafları analiz edilecek")
    else:
        print("⚠️ Tüm fotoğraflar (gece/gündüz) analiz edilecek")
    
    # Analizi çalıştır
    print(f"\n🚀 {len(LOCATIONS)} lokasyon analiz edilecek...")
    
    all_results = []
    total_rejected = 0
    
    for i, location in enumerate(LOCATIONS, 1):
        print(f"\n[{i}/{len(LOCATIONS)}] Lokasyon analizi başlatılıyor...")
        result = analyzer.analyze_location(location, reject_night=reject_night)
        all_results.append(result)
        
        # Reddedilen fotoğraf sayısını topla
        if 'rejected_images' in result:
            total_rejected += len(result['rejected_images'])
    
    # Kapsamlı rapor oluştur
    print(f"\n📊 Kapsamlı rapor oluşturuluyor...")
    analyzer.create_comprehensive_report(all_results)
    
    # Özet bilgiler
    print(f"\n🎉 Analiz tamamlandı!")
    print(f"📁 Tüm dosyalar ilgili klasörlerde saklandı")
    if reject_night and total_rejected > 0:
        print(f"🌙 Toplam {total_rejected} gece fotoğrafı reddedildi")
    
    # Sonuçları JSON olarak kaydet
    results_file = os.path.join(analyzer.directories['reports'], 
                               f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # JSON serializable hale getir
    json_results = []
    for result in all_results:
        json_result = {
            'location_info': result['location_info'],
            'summary': result['summary'],
            'rejected_count': len(result.get('rejected_images', [])),
            'angles_data': {}
        }
        
        for angle, data in result['angles'].items():
            json_result['angles_data'][str(angle)] = {
                'gvi': data.get('gvi', 0),
                'image_available': data.get('image_available', False),
                'error': data.get('error', None)
            }
        
        json_results.append(json_result)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Sonuçlar kaydedildi: {results_file}")

if __name__ == "__main__":
    main()
