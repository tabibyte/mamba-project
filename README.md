# Durum Uzay Modelleri ile Çok Etiketli EKG Sınıflandırması
**Multi-Label ECG Classification with Mamba State Space Models**

Bu depo, 12 kanallı EKG sinyallerinde çok etiketli (multi-label) kardiyak anormalliklerin tespiti için geliştirilen **1 Boyutlu Çift Yönlü Gelişmiş Mamba (1D Bidirectional Advanced Mamba)** ve **1 Boyutlu melez CNN-Transformer** modellerinin karşılaştırmalı analizini içermektedir. 

Çalışma, yeni nesil Durum Uzay Modellerinin (State Space Models - SSM) klinik EKG analizinde standart Transformer mimarilerine karşı tanısal doğruluğunu (özellikle imbalans/dengesiz veri setlerindeki AUPRC başarımını) ve donanımsal verimliliğini araştırmaktadır.

---

## Özellikler
* EKG'nin nedensel olmayan (non-causal) tanısal yapısını kavramak için özel olarak tasarlanmış **BiMambaBlock** (İleri ve Geri yönlü tarama).
*  Karesel O(N)^2 Transformer maliyetine karşı, doğrusal O(N) zaman karmaşıklığı ile sabit RAM/VRAM tüketimi.
* Toplamda 48.000'den fazla kayıt içeren 4 farklı açık kaynaklı uluslararası EKG veri setinde (PTB-XL, Georgia, Chapman-Shaoxing, CPSC 2018) SNOMED-CT kodları üzerinden standartlaştırılmış değerlendirme.

---

## Kullanılan Veri Setleri
Modeller, çok merkezli verilerin doğası gereği barındırdığı farklı örnekleme hızları ve süreleri, **PTB-XL'in 5 ana tanısal üst sınıfı (NORM, MI, STTC, CD, HYP)** temel alınarak uyumlaştırılmış 10 saniyelik pencereler halinde eğitilmiştir.

1. **PTB-XL** (21.841 kayıt, 100Hz)
2. **Georgia 12-Lead ECG Challenge** (10.344 kayıt, 500Hz -> 100Hz'e alt örneklendi)
3. **Chapman-Shaoxing** (10.247 kayıt, 100Hz, Dinamik Kırpma/Sıfır Dolgusu)
4. **China Physiological (CPSC 2018)** (6.877 kayıt, 500Hz -> 100Hz)

---

## Model Mimarileri

### 1. Advanced ECG Mamba (Önerilen)
* **Stem:** 1D Conv (Kernel 15) -> GELU -> 1D Conv (Kernel 7) [Sinyali 4 kat alt örnekler]
* **Backbone:** 6x `BiMambaBlock` ($D_{model}=256$, $D_{state}=32$). Sinyali eşzamanlı olarak ileri ve geri yönde tarayarak bağlam kurar.
* **Head:** RMSNorm -> **Global Max Pooling** -> MLP -> 5 Sınıflı Logit Çıkışı

### 2. Advanced ECG Transformer (Karşılaştırma / Baseline)
* **Stem:** Mamba ile tamamen aynı evrişimli kök mimarisi.
* **Backbone:** Positional Encoding -> 6x Transformer Encoder Katmanı (8 Head, Pre-LN kararlılık ayarı).
* **Head:** LayerNorm -> **Global Average Pooling** -> MLP -> 5 Sınıflı Logit Çıkışı

---

## Bulgular ve Karşılaştırmalı Performans

### Sınıflandırma Başarımı (30. Epok Nihai Değerleri)
Mamba mimarisi, özellikle nadir görülen EKG patolojilerinin tespitini gösteren **AUPRC** (Hassasiyet-Duyarlılık) metriğinde ezici bir üstünlük sağlamıştır.

| Veri Seti | Model | Makro AUC | Makro AUPRC |
| :--- | :--- | :--- | :--- |
| **Georgia** | **Bi-Mamba** | **0.9021** | **0.7782** |
| | CNN-Transformer | 0.8729 | 0.7334 |
| **CPSC 2018** | **Bi-Mamba** | **0.9669** | **0.8713** |
| | CNN-Transformer | 0.9520 | 0.8388 |
| **Chapman** | Bi-Mamba | **0.9248** | 0.5229 |
| | **CNN-Transformer** | 0.8945 | **0.5861** |
| **PTB-XL (Test)**| **Bi-Mamba** | **0.9016** | **0.7465** |
| | CNN-Transformer | 0.8861 | 0.7135 |

*(Not: Chapman verisindeki Mamba AUPRC düşüşünün temel nedeni, değişken uzunluklu sinyalleri 10 saniyeye zorlamak için atılan yapay sıfır dolgularının (zero-padding), Mamba'nın Global Max Pooling katmanında yarattığı gradyan hassasiyetidir.)*

### Hesaplama Maliyeti ve Verimlilik Analizi
Aşağıdaki değerler NVIDIA Tesla T4 GPU üzerinde elde edilmiştir. Mamba, çok daha düşük RAM ayak izi ve daha hızlı çıkarım süresi ile giyilebilir kardiyoloji teknolojileri (Edge AI) için ideal bir profil çizmiştir.

| Özellik | 1B Çift Yönlü Mamba | 1B CNN-Transformer |
| :--- | :--- | :--- |
| **Parametre Sayısı** | $\approx$ 5.2 Milyon | $\approx$ 4.8 Milyon |
| **Zaman Karmaşıklığı**| O(N) | O(N^2)|
| **Çıkarım Hızı (10sn EKG)*| **2.5 - 3.0 ms** | 4.5 - 5.5 ms |
| **Maks. VRAM (Eğitim)** | **2.4 GB** | 3.1 GB |

---