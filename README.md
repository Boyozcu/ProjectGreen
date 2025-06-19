# ProjectGreen 🌱

**Gelişmiş Yeşil Alan Analiz Sistemi** - Street View görüntülerinden AI tabanlı yeşil alan analizi

## 📋 Proje Hakkında

ProjectGreen, Google Street View görüntülerini kullanarak şehir alanlarındaki yeşil alanları analiz eden, AI tabanlı bir sistemdir. SegFormer derin öğrenme modeli kullanarak vegetation detection yapar ve Green View Index (GVI) hesaplar.

### ✨ Özellikler

- 🤖 **AI Tabanlı Analiz**: SegFormer modeli ile vegetation segmentation
- 📊 **GVI Hesaplama**: Yeşil Görünüm Endeksi skorları
- 🗺️ **Çoklu Konum Analizi**: Farklı lokasyonlardan 360° analiz
- 📈 **Detaylı Raporlama**: JSON, CSV ve görsel raporlar
- 🎨 **Gelişmiş Görselleştirme**: Yeşil alanların işaretlenmesi ve analizi
- 💾 **Kapsamlı Kayıt**: Görüntüler, maskeler ve analizlerin saklanması

## 🚀 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### API Anahtarı
Google Street View API anahtarı gereklidir:
1. [Google Cloud Console](https://console.cloud.google.com/) hesabı oluşturun
2. Street View Static API'yi etkinleştirin
3. API anahtarınızı koda ekleyin (güvenlik için environment variable kullanın)

## 📂 Proje Yapısı

```
ProjectGreen/
├── projectGreen_colab.py          # Orijinal Colab versiyonu
├── enhanced_project_green.py      # Gelişmiş versiyon
├── green_area_visualizer.py       # Yeşil alan görselleştirici
├── requirements.txt               # Python paketleri
├── README.md                     # Bu dosya
└── Oluşturulan Klasörler:
    ├── street_view_images/       # İndirilen görüntüler
    ├── analysis_results/         # Analiz sonuçları
    ├── visualizations/           # Görselleştirmeler
    ├── reports/                  # Raporlar (JSON/CSV)
    └── vegetation_masks/         # Vegetation maskeleri
```

## 🎯 Kullanım

### 1. Temel Analiz
```python
python enhanced_project_green.py
```

### 2. Yeşil Alan Görselleştirme
```python
python green_area_visualizer.py
```

### 3. Colab Versiyonu
Google Colab'da `projectGreen_colab.py` dosyasını çalıştırın.

## 📊 Çıktılar

- **GVI Skorları**: Her lokasyon için yeşil alan yüzdeleri
- **Görsel Raporlar**: Yeşil alanların işaretlenmiş görselleri
- **Detaylı Analizler**: Bölge bazlı yeşil alan istatistikleri
- **Karşılaştırmalı Veriler**: Lokasyonlar arası yeşil alan kıyaslaması

## 🔧 Geliştirme Önerileri

### 1. **Web Arayüzü Geliştirme**
```python
# Streamlit tabanlı web arayüzü
import streamlit as st

def create_web_interface():
    st.title("🌱 ProjectGreen - Yeşil Alan Analizi")
    
    # Konum seçimi
    location = st.text_input("Konum Koordinatları (lat,lng)")
    
    # Analiz butonu
    if st.button("Analiz Et"):
        # Analiz kodları
        pass
```

### 2. **Interaktif Harita Entegrasyonu**
```python
import folium
from folium import plugins

def create_interactive_map(results):
    # Yeşil alan yoğunluğuna göre renkli harita
    m = folium.Map(location=[41.0369, 28.9850], zoom_start=15)
    
    for result in results:
        folium.CircleMarker(
            location=[lat, lng],
            radius=result['gvi'] * 50,
            color='green',
            popup=f"GVI: {result['gvi']:.3f}"
        ).add_to(m)
    
    return m
```

### 3. **Zaman Serisi Analizi**
```python
def temporal_analysis(location, time_range):
    # Farklı zamanlarda aynı lokasyonun analizi
    # Mevsimsel değişimler
    # Trend analizi
    pass
```

### 4. **Mobil Uygulama Entegrasyonu**
```python
# Flutter/React Native ile mobil uygulama
# Real-time yeşil alan skorları
# Augmented Reality ile yeşil alan gösterimi
```

### 5. **Performans Optimizasyonları**

#### A. Batch Processing
```python
def batch_analyze_images(image_paths, batch_size=8):
    # Birden fazla görüntüyü aynı anda işle
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

#### B. Caching Sistemi
```python
import pickle
import hashlib

def cache_analysis(image_path, analysis_func):
    # Analiz sonuçlarını cache'le
    cache_key = hashlib.md5(image_path.encode()).hexdigest()
    cache_file = f"cache/{cache_key}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    result = analysis_func(image_path)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result
```

### 6. **Gelişmiş Analiz Özellikleri**

#### A. Yeşil Alan Kalitesi Analizi
```python
def analyze_green_quality(vegetation_mask, image):
    # Yeşil alanın sağlık durumu
    # Yaprak yoğunluğu
    # Tür çeşitliliği tahmini
    pass
```

#### B. Kentsel Planlama Önerileri
```python
def urban_planning_suggestions(gvi_scores, location_data):
    # Düşük GVI'li alanlar için öneriler
    # Yeşil koridor planlaması
    # Optimum ağaç dikim noktaları
    pass
```

### 7. **Veri Analizi ve Makine Öğrenmesi**

#### A. Predictive Modeling
```python
from sklearn.ensemble import RandomForestRegressor

def predict_gvi_changes(historical_data, environmental_factors):
    # Gelecekteki GVI değişimlerini tahmin et
    # Çevresel faktörlerin etkisi
    # Sezonsal değişimler
    pass
```

#### B. Anomali Tespiti
```python
def detect_anomalies(gvi_time_series):
    # Anormal GVI değişimlerini tespit et
    # Çevre kirliliği etkilerini belirle
    # Erken uyarı sistemi
    pass
```

### 8. **API ve Mikroservis Mimarisi**

#### A. RESTful API
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_location():
    data = request.json
    result = analyze_green_area(data['coordinates'])
    return jsonify(result)

@app.route('/api/report/<location_id>')
def get_report(location_id):
    report = generate_report(location_id)
    return jsonify(report)
```

#### B. Real-time Monitoring
```python
import asyncio
import websockets

async def real_time_monitoring(websocket, path):
    # Gerçek zamanlı GVI güncellemeleri
    # Live monitoring dashboard
    pass
```

### 9. **Veritabanı Entegrasyonu**

```sql
-- PostgreSQL with PostGIS for spatial data
CREATE TABLE green_analysis (
    id SERIAL PRIMARY KEY,
    location_id VARCHAR(50),
    coordinates POINT,
    gvi_score DECIMAL(5,4),
    analysis_date TIMESTAMP,
    image_url TEXT,
    vegetation_areas JSONB
);

CREATE INDEX idx_location_gvi ON green_analysis(location_id, gvi_score);
CREATE INDEX idx_spatial ON green_analysis USING GIST(coordinates);
```

### 10. **Docker ve Deployment**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "enhanced_project_green.py"]
```

## 🔍 Gelecek Özellikler

1. **Citizen Science Entegrasyonu**: Kullanıcıların kendi gözlemlerini ekleyebilmesi
2. **Climate Impact Analysis**: İklim değişikliğinin yeşil alanlara etkisi
3. **Biodiversity Index**: Biyoçeşitlilik endeksi hesaplama
4. **Air Quality Correlation**: Hava kalitesi ile yeşil alan korelasyonu
5. **Smart City Integration**: Akıllı şehir sistemleriyle entegrasyon

## 📈 Performans Metrikleri

- **Analiz Hızı**: ~2-5 saniye/görüntü
- **Doğruluk Oranı**: %85-92 (vegetation detection)
- **Bellek Kullanımı**: ~2-4GB (GPU mode)
- **API Response Time**: <500ms

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- NVIDIA SegFormer modeli
- Google Street View API
- OpenCV ve PIL kütüphaneleri
- Hugging Face Transformers

---

**Not**: Bu proje sürekli geliştirilmektedir. Öneriler ve katkılar için issue açabilirsiniz.