# -*- coding: utf-8 -*-
"""
========================================================
  Konya Tarım Fiyat Tahmini — DEMO BACKEND
  Proje: Local Agricultural Price & Climate Predictor
  Model: Random Forest Regressor (64 feature, panel data)
========================================================

KURULUM:
  pip install flask flask-cors joblib scikit-learn numpy pandas

MODELI KAYDET (notebook'ta bir kez çalıştır):
  import joblib
  joblib.dump(best_model, "model.joblib")

ÇALIŞTIR:
  python demo_server.py
  → http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ================================================================
#  PROJE AYARLARI
# ================================================================

MODEL_PATH  = "model.joblib"
SCALER_PATH = None          # Ağaç modelleri ölçekleme gerektirmez
MODEL_TYPE  = "regression"

# ── UI'da kullanıcıya gösterilecek 8 yorumlanabilir özellik ─────
# (Modelin 64 özelliğinden en önemlileri — geri kalanlar varsayılanla doldurulur)
FEATURE_NAMES = [
    "Onceki_Seans_En_Dusuk_Fiyat",
    "Onceki_Seans_Degisim_Yuzdesi",
    "Fiyat_Oynakligi_L3",
    "Ort_Sicaklik_C",
    "Toplam_Yagis_mm",
    "Don_Olayi",
    "Asiri_Sicak_Soku",
    "ay",
]

FEATURE_LABELS = [
    "Önceki Seans En Düşük Fiyat (₺/kg)",
    "Önceki Seans Fiyat Değişimi (%)",
    "Son 3 Seans Fiyat Oynaklığı (%)",
    "Ortalama Sıcaklık (°C)",
    "Günlük Toplam Yağış (mm)",
    "Don Olayı (0=Yok, 1=Var)",
    "Aşırı Sıcak Şoku (0=Yok, 1=Var)",
    "Ay (1=Ocak … 12=Aralık)",
]

# ── Hedef değişken bilgisi ──────────────────────────────────────
TARGET_LABEL = "Tahmin Edilen Fiyat Değişim Yüzdesi"
TARGET_UNIT  = "%"
TARGET_MIN   = -100.0
TARGET_MAX   = 150.0

# ── Model performans bilgisi (UI'da gösterilir) ─────────────────
MODEL_INFO = {
    "model_name"    : "Random Forest Regressor",
    "metric1_label" : "Back-test RMSE",
    "metric1_value" : "1.91 puan",
    "metric2_label" : "K-Fold RMSE (5-fold ort.)",
    "metric2_value" : "10.65 puan",
    "training_note" : (
        "Konya Hal Müdürlüğü verileri 2021-05-06 – 2025-12-15 "
        "(n=21.422 kayıt, 88 ürün, 64 özellik)"
    ),
}

# ================================================================
#  MODELİN BEKLEDİĞİ 64 ÖZELLİĞİN TAM LİSTESİ (eğitimdeki sırayla)
# ================================================================
ALL_FEATURE_NAMES = [
    "Onceki_Seans_En_Dusuk_Fiyat",
    "Maks_Sicaklik_C",
    "Min_Sicaklik_C",
    "Ort_Sicaklik_C",
    "Toplam_Yagis_mm",
    "Ort_Sicaklik_C_7_gun_once",
    "Ort_Sicaklik_C_14_gun_once",
    "Ort_Sicaklik_C_30_gun_once",
    "Toplam_Yagis_mm_7_gun_once",
    "Toplam_Yagis_mm_14_gun_once",
    "Toplam_Yagis_mm_30_gun_once",
    "Son_30_Gun_Ort_Yagis",
    "Urun_Frekans_Kodu",
    "Urun_Kategori_BROKOLİ",
    "Urun_Kategori_BİBER (CİN)",
    "Urun_Kategori_BİBER (DOLMA)",
    "Urun_Kategori_BİBER (KAPYA)",
    "Urun_Kategori_BİBER (KIL)",
    "Urun_Kategori_BİBER (KÖY)",
    "Urun_Kategori_BİBER (SİVRİ)",
    "Urun_Kategori_BİBER (ÇARLİ)",
    "Urun_Kategori_DERE OTU",
    "Urun_Kategori_DIGER",
    "Urun_Kategori_DOMATES (KÖY)",
    "Urun_Kategori_DOMATES (MUHTELİF)",
    "Urun_Kategori_DOMATES (SALKIM)",
    "Urun_Kategori_DOMATES (Çeri)",
    "Urun_Kategori_DOMATES (İKİNCİ)",
    "Urun_Kategori_EKŞİ OT (BAĞ)",
    "Urun_Kategori_FASULYE (MUHTELİF)",
    "Urun_Kategori_FESLEĞEN (REYHAN)",
    "Urun_Kategori_HAVUÇ",
    "Urun_Kategori_ISPANAK",
    "Urun_Kategori_KABAK (BEYAZ)",
    "Urun_Kategori_KABAK (KARA)",
    "Urun_Kategori_KARNABAHAR",
    "Urun_Kategori_LAHANA (BEYAZ)",
    "Urun_Kategori_LAHANA (MOR)",
    "Urun_Kategori_MANTAR (KÜLTÜR)",
    "Urun_Kategori_MARUL (AYSBERK)",
    "Urun_Kategori_MARUL (MOR)",
    "Urun_Kategori_MARUL (MUHTELİF)",
    "Urun_Kategori_MAYDONOZ",
    "Urun_Kategori_NANE (BAĞ)",
    "Urun_Kategori_PATATES",
    "Urun_Kategori_PATLICAN (KEMER)",
    "Urun_Kategori_PATLICAN (TOPAK)",
    "Urun_Kategori_ROKA (BAĞ)",
    "Urun_Kategori_SALATALIK (MUHTELİF)",
    "Urun_Kategori_SALATALIK (SİLOR)",
    "Urun_Kategori_SARIMSAK (KURU)",
    "Urun_Kategori_SEMİZ OTU (BAĞ)",
    "Urun_Kategori_SOGAN (KURU)",
    "Urun_Kategori_SOGAN (YEŞİL)",
    "Urun_Kategori_TERE",
    "Don_Olayi",
    "Asiri_Sicak_Soku",
    "Sicaklik_Trendi",
    "Onceki_Seans_Degisim_Yuzdesi",
    "Fiyat_Oynakligi_L3",
    "Normalize_Fiyat",
    "ay",
    "ay_sin",
    "ay_cos",
]

# ================================================================
#  KULLANICININ GİRMEDİĞİ ÖZELLİKLER İÇİN VARSAYıLAN DEĞERLER
#  (2021-2026 eğitim setinin ortalamaları)
# ================================================================
FEATURE_DEFAULTS = {
    "Onceki_Seans_En_Dusuk_Fiyat"     : 16.0934,
    "Maks_Sicaklik_C"                  : 19.1693,
    "Min_Sicaklik_C"                   : 7.5446,
    "Ort_Sicaklik_C"                   : 13.1683,
    "Toplam_Yagis_mm"                  : 0.752,
    "Ort_Sicaklik_C_7_gun_once"        : 13.1735,
    "Ort_Sicaklik_C_14_gun_once"       : 13.3282,
    "Ort_Sicaklik_C_30_gun_once"       : 13.6177,
    "Toplam_Yagis_mm_7_gun_once"       : 0.7788,
    "Toplam_Yagis_mm_14_gun_once"      : 0.7449,
    "Toplam_Yagis_mm_30_gun_once"      : 0.8188,
    "Son_30_Gun_Ort_Yagis"             : 0.9178,
    "Urun_Frekans_Kodu"                : 0.0187,
    "Urun_Kategori_BROKOLİ"            : 0.0,
    "Urun_Kategori_BİBER (CİN)"        : 0.0,
    "Urun_Kategori_BİBER (DOLMA)"      : 0.0,
    "Urun_Kategori_BİBER (KAPYA)"      : 0.0,
    "Urun_Kategori_BİBER (KIL)"        : 0.0,
    "Urun_Kategori_BİBER (KÖY)"        : 0.0,
    "Urun_Kategori_BİBER (SİVRİ)"      : 0.0,
    "Urun_Kategori_BİBER (ÇARLİ)"      : 0.0,
    "Urun_Kategori_DERE OTU"           : 0.0,
    "Urun_Kategori_DIGER"              : 1.0,   # varsayılan ürün = DIGER
    "Urun_Kategori_DOMATES (KÖY)"      : 0.0,
    "Urun_Kategori_DOMATES (MUHTELİF)" : 0.0,
    "Urun_Kategori_DOMATES (SALKIM)"   : 0.0,
    "Urun_Kategori_DOMATES (Çeri)"     : 0.0,
    "Urun_Kategori_DOMATES (İKİNCİ)"   : 0.0,
    "Urun_Kategori_EKŞİ OT (BAĞ)"     : 0.0,
    "Urun_Kategori_FASULYE (MUHTELİF)" : 0.0,
    "Urun_Kategori_FESLEĞEN (REYHAN)"  : 0.0,
    "Urun_Kategori_HAVUÇ"              : 0.0,
    "Urun_Kategori_ISPANAK"            : 0.0,
    "Urun_Kategori_KABAK (BEYAZ)"      : 0.0,
    "Urun_Kategori_KABAK (KARA)"       : 0.0,
    "Urun_Kategori_KARNABAHAR"         : 0.0,
    "Urun_Kategori_LAHANA (BEYAZ)"     : 0.0,
    "Urun_Kategori_LAHANA (MOR)"       : 0.0,
    "Urun_Kategori_MANTAR (KÜLTÜR)"    : 0.0,
    "Urun_Kategori_MARUL (AYSBERK)"    : 0.0,
    "Urun_Kategori_MARUL (MOR)"        : 0.0,
    "Urun_Kategori_MARUL (MUHTELİF)"   : 0.0,
    "Urun_Kategori_MAYDONOZ"           : 0.0,
    "Urun_Kategori_NANE (BAĞ)"         : 0.0,
    "Urun_Kategori_PATATES"            : 0.0,
    "Urun_Kategori_PATLICAN (KEMER)"   : 0.0,
    "Urun_Kategori_PATLICAN (TOPAK)"   : 0.0,
    "Urun_Kategori_ROKA (BAĞ)"         : 0.0,
    "Urun_Kategori_SALATALIK (MUHTELİF)": 0.0,
    "Urun_Kategori_SALATALIK (SİLOR)"  : 0.0,
    "Urun_Kategori_SARIMSAK (KURU)"    : 0.0,
    "Urun_Kategori_SEMİZ OTU (BAĞ)"    : 0.0,
    "Urun_Kategori_SOGAN (KURU)"       : 0.0,
    "Urun_Kategori_SOGAN (YEŞİL)"      : 0.0,
    "Urun_Kategori_TERE"               : 0.0,
    "Don_Olayi"                        : 0.0,
    "Asiri_Sicak_Soku"                 : 0.0,
    "Sicaklik_Trendi"                  : -0.0052,
    "Onceki_Seans_Degisim_Yuzdesi"     : 0.915,
    "Fiyat_Oynakligi_L3"               : 5.4206,
    "Normalize_Fiyat"                  : 0.0059,
    "ay"                               : 6.0,
    "ay_sin"                           : 0.0,    # ay'dan otomatik hesaplanır
    "ay_cos"                           : 0.0,    # ay'dan otomatik hesaplanır
}

# ================================================================
#  END OF CONFIG
# ================================================================

# Sınıflandırma için etiketler (bu projede kullanılmıyor)
CLASS_LABELS = {}

print(f"\n[demo_server] Model yükleniyor: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print(f"[demo_server] Model yüklendi: {type(model).__name__}")
except FileNotFoundError:
    print(f"[demo_server] HATA: '{MODEL_PATH}' bulunamadı.")
    print("  Modeli notebook'ta kaydet:  joblib.dump(best_model, 'model.joblib')")
    model = None

scaler = None


@app.route("/")
def index():
    return send_from_directory(".", "demo_ui.html")


@app.route("/config")
def get_config():
    return jsonify({
        "model_type"    : MODEL_TYPE,
        "feature_names" : FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "target_label"  : TARGET_LABEL,
        "target_unit"   : TARGET_UNIT,
        "target_min"    : TARGET_MIN,
        "target_max"    : TARGET_MAX,
        "class_labels"  : CLASS_LABELS,
        "model_info"    : MODEL_INFO,
        "model_ready"   : model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model yüklenmedi. '{MODEL_PATH}' dosyasını kontrol et."}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "İstek 'features' dict içermelidir."}), 400

    raw = data["features"]

    # Kullanıcı girdilerini doğrula
    try:
        user_values = {name: float(raw[name]) for name in FEATURE_NAMES}
    except KeyError as e:
        return jsonify({"error": f"Eksik özellik: {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Geçersiz değer: {e}"}), 400

    # Tam 64-boyutlu vektörü oluştur: varsayılanlar + kullanıcı girdileri
    full = dict(FEATURE_DEFAULTS)
    full.update(user_values)

    # ay'dan ay_sin ve ay_cos'u otomatik hesapla
    ay = full["ay"]
    full["ay_sin"] = float(np.sin(2 * np.pi * ay / 12))
    full["ay_cos"] = float(np.cos(2 * np.pi * ay / 12))

    # Eğitimdeki sıraya göre vektörü oluştur
    try:
        values = [full[name] for name in ALL_FEATURE_NAMES]
    except KeyError as e:
        return jsonify({"error": f"Eksik varsayılan değer: {e}"}), 500

    X = np.array(values, dtype=float).reshape(1, -1)

    try:
        pred = float(model.predict(X)[0])
        result = {
            "prediction"  : round(pred, 2),
            "target_label": TARGET_LABEL,
            "target_unit" : TARGET_UNIT,
            "target_min"  : TARGET_MIN,
            "target_max"  : TARGET_MAX,
        }
    except Exception as e:
        return jsonify({"error": f"Tahmin başarısız: {str(e)}"}), 500

    # Feature importance — UI'da gösterilecek 8 özellik için
    if hasattr(model, "feature_importances_"):
        all_imp = model.feature_importances_
        name_to_imp = dict(zip(ALL_FEATURE_NAMES, all_imp))
        result["feature_importances"] = {
            name: round(float(name_to_imp.get(name, 0.0)), 4)
            for name in FEATURE_NAMES
        }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Konya Tarım Fiyat Tahmini — Demo Sunucusu")
    print("  Tarayıcıda aç: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
