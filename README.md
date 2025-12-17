🏭 Akıllı Üretim Hattı Ürün Takip ve Sayım Sistemi
Bu proje, endüstriyel üretim hatları üzerinde hareket eden ürünleri gerçek zamanlı olarak tespit etmek, takip etmek ve saymak için geliştirilmiş yapay zeka destekli bir masaüstü uygulamasıdır. YOLOv8 nesne algılama algoritması ve DeepSort takip algoritmasını kullanarak çift kameralı bir takip sistemi sunar.

🚀 Özellikler
-- Çift Kamera Desteği: İki farklı üretim hattını veya aynı hattın iki farklı açısını eş zamanlı izleyebilme.
-- Gerçek Zamanlı Nesne Algılama: YOLOv8 (PyTorch) ile yüksek doğrulukta ürün tespiti.
-- Gelişmiş Nesne Takibi: DeepSort algoritması ile her ürüne benzersiz bir ID atayarak mükerrer sayımların önlenmesi.
-- Çizgi Geçiş Analizi: Belirlenen sanal çizgiyi geçen ürünlerin otomatik sayılması.
-- Dinamik Hedef Takibi: Toplam üretim miktarına göre renk değiştiren (Kırmızı -> Turuncu -> Yeşil) hedef kutusu.
-- Kullanıcı Arayüzü (GUI): PyQt5 ile geliştirilmiş, FPS, sistem saati ve üretim verilerini gösteren modern dashboard.
-- Hat Bazlı Sıfırlama: Her hat için bağımsız üretim sayacını sıfırlama imkanı.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji                 | Kullanım Amacı                            |
| :-----------------------: | :---------------------------------------: |
| **Python 3.x**            | Ana Programlama Dili                      |
| **YOLOv8 (Ultralytics)**  | Nesne Algılama (Object Detection)         |
| **DeepSort**              | Nesne Takibi (Object Tracking)            |
| **PyQt5**                 | Grafiksel Kullanıcı Arayüzü (GUI)         |
| **OpenCV**                | Görüntü İşleme ve Kamera Yönetimi         |
| **PyTorch**               | Derin Öğrenme Modeli Çalıştırma (GPU/CPU) |

![licensed-image](https://github.com/user-attachments/assets/a9d9d8a7-2c8a-489f-a2f2-cb3256c66aa7)

📂 Proje Yapısı

Hat-Urun-Tanimlama/
├── models/
│   ├── best.pt            # Özel eğitilmiş YOLOv8 modeliniz
│   └── yolov8n.pt         # Temel YOLOv8 nano modeli
├── data/
│   └── video.mp4          # Test videoları
├── assets/
│   └── dino.png           # Kurumsal logo / Arayüz görseli
├── main.py                # Ana uygulama kodu
├── requirements.txt       # Gerekli kütüphaneler listesi
└── README.md              # Proje dokümantasyonu

⚙️ Kurulum 

1. Bu depoyu klonlayın:
   "git clone https://github.com/kullaniciadin/uretim-hatti-urun-sayma.git
cd uretim-hatti-urun-sayma"

2. Gerekli kütüphaneleri yükleyin:
   "pip install -r requirements.txt"


🖥️ Kullanım
Uygulamayı başlatmak için terminale şu komutu yazın:
  python main.py
  
  ⚠️Not: Kodun içindeki video_path, model_path ve dino_path yollarının kendi klasör yapınıza uygun olduğundan emin olun.


