# KYK Oda Eslestirme Optimizasyonu

KYK yurt odalarina yeni gelen ogrencileri, mevcut oda profillerine gore **optimum sekilde eslestiren** bir sistem. Kural tabanli algoritma (v1) ve yapay zeka destekli (v2) olmak uzere iki asamadan olusmaktadir.

## Senaryo

```
5 Yil: Kural Tabanli       Her Yil Sonu:        5 Yillik Egitim    Random Forest     6. Yil+: AI Destekli
Eslestirme (v1)      --->   Geri Bildirim   ---> Verisi Birikimi -> Model Egitme  --> Eslestirme (v2)
```

1. **Ilk 5 yil:** Kural tabanli algoritma (v1) ile ogrenci yerlestirmeleri yapilir.
2. **Her yil sonunda:** Eslestirmelerin basari durumu (memnuniyet) geri bildirimi toplanir.
3. **5 yil sonunda:** Biriken veri ile Random Forest modeli egitilir.
4. **6. yil ve sonrasi:** AI destekli eslestirme (v2) kullanilir.

## Ozellikler

### Eslestirme Boyutlari

Her ogrenci ve oda 6 boyutta profillenir (1-10 arasi):

| Boyut | Ogrenci | Oda |
|---|---|---|
| Gurultu | `noise_tolerance` | `room_noise_profile` |
| Sigara | `smoking_level` | `room_smoking_profile` |
| Cevre hassasiyeti | `environment_sensitivity` | `room_environment_irritant_level` |
| Uyanis saati | `wake_time` | `room_wake_profile` |
| Giris saati | `entry_time` | `room_entry_profile` |
| Uyku kesintisi | `sleep_interrupt_sensitivity` | (odanin gurultu profilinden turetilir) |

### Algoritma (v1)
- **Min-Cost Max-Flow** ile global optimum eslestirme (greedy/sirali degil)
- Hard constraint filtreleri (sigara + cevre uyumsuzlugu)
- Agirlikli benzerlik skoru (dogrusal + dairesel)

### AI Modeli (v2)
- **Random Forest Regressor** (scikit-learn)
- Hibrit mod: `alpha * AI_score + (1-alpha) * kural_score` ile kademeli gecis
- Feature importance ile yorumlanabilir sonuclar

## Dosya Yapisi

```
.
├── kyk_matcher_v1.py       # Kural tabanli eslestirme algoritmasi
├── kyk_matcher_ui.py       # v1 Tkinter arayuzu
├── kyk_matcher_v2.py       # AI destekli eslestirme
├── kyk_matcher_v2_ui.py    # v2 Tkinter arayuzu
├── ai_model.py             # Random Forest modeli (egitim/tahmin)
├── generate_feedback.py    # Sentetik geri bildirim uretici
├── pipeline.py             # End-to-end pipeline
├── rooms.csv               # Ornek oda verisi
├── new_students.csv        # Ornek ogrenci verisi
├── assignments.csv         # v1 eslestirme sonuclari (ornek)
├── requirements.txt        # Python bagimliliklari
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanim

### Hizli Baslangic (Pipeline)
Tum adimlari sirayla calistirir:
```bash
python pipeline.py
```

### v1 - Kural Tabanli Eslestirme
```bash
python kyk_matcher_v1.py        # Komut satiri
python kyk_matcher_ui.py        # Arayuz
```

### v2 - AI Destekli Eslestirme
```bash
python kyk_matcher_v2.py        # Komut satiri
python kyk_matcher_v2_ui.py     # Arayuz
```

### v2 Arayuz Kullanim Adimlari
1. **Oda CSV Yukle** - Oda verilerini iceren CSV dosyasini secin
2. **Ogrenci CSV Yukle** - Yeni ogrenci verilerini secin
3. **Feedback Uret / Yukle** - Sentetik veri uretin veya mevcut feedback yukleyin
4. **Model Egit** - Random Forest modelini egitin
5. **v2 Eslestir** - AI destekli eslestirme yapin (alpha slider ile AI agirligini ayarlayin)
6. **Sonuclari Kaydet** - CSV dosyasina kaydedin

## Gereksinimler

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib
- tkinter (Python ile birlikte gelir)
