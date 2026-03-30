import pandas as pd
import numpy as np
import os

# 1. Dosya Yolları (File Paths)
base_path = "C:/Users/ismet/Desktop/Projects/Local_Agricultural_Price_and_Climate_Predictor"
hal_csv = os.path.join(base_path, "konya_hal_fiyatlari_temiz_5yillik.csv")
hava_csv = os.path.join(base_path, "konya_hava_durumu_5yillik.csv")

print("Veriler okunuyor ve özellik mühendisliği (Feature Engineering) başlıyor...")

df_hal = pd.read_csv(hal_csv)
df_hava = pd.read_csv(hava_csv)

# 2. Tarih Formatları (Date Formatting)
df_hal['Tarih'] = pd.to_datetime(df_hal['Tarih'], format='%d.%m.%Y')
df_hava['Tarih'] = pd.to_datetime(df_hava['Tarih'], format='%d.%m.%Y')

# 3. VERİ TEMİZLİĞİ: Fiyat Sütununu Sayısala Çevirme (Data Cleaning: Convert Price to Numeric)
# Colab'de yaptığımız virgül/nokta temizliğini doğrudan kaynağında yapıyoruz
df_hal['En Düşük Fiyat'] = df_hal['En Düşük Fiyat'].astype(str).str.replace('.', '', regex=False) # Varsa binlik ayracını sil
df_hal['En Düşük Fiyat'] = df_hal['En Düşük Fiyat'].str.replace(',', '.', regex=False) # Kuruş ayracını noktaya çevir
df_hal['En Düşük Fiyat'] = pd.to_numeric(df_hal['En Düşük Fiyat'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

# 4. FİYAT GECİKMESİ VE HEDEF DEĞİŞKEN (Price Lag & Target Variable)
df_hal = df_hal.sort_values(['Ürün Adı', 'Tarih'])
df_hal['Onceki_Seans_En_Dusuk_Fiyat'] = df_hal.groupby('Ürün Adı')['En Düşük Fiyat'].shift(1)

# ENFLASYONDAN ARINDIRMA: Hedef değişken artık "Yüzdelik Değişim" (% Change)
df_hal['Fiyat_Degisim_Yuzdesi'] = ((df_hal['En Düşük Fiyat'] - df_hal['Onceki_Seans_En_Dusuk_Fiyat']) / df_hal['Onceki_Seans_En_Dusuk_Fiyat']) * 100

# Matematiksel hataları (Sıfıra bölünme sonucu çıkan sonsuz değerleri) temizleme
df_hal['Fiyat_Degisim_Yuzdesi'].replace([np.inf, -np.inf], np.nan, inplace=True)

# 5. İKLİM GECİKMELERİ (Climate Lags)
df_hava = df_hava.sort_values('Tarih')
for col in ['Ort_Sicaklik_C', 'Toplam_Yagis_mm']:
    df_hava[f'{col}_7_gun_once'] = df_hava[col].shift(7)
    df_hava[f'{col}_14_gun_once'] = df_hava[col].shift(14)
    df_hava[f'{col}_30_gun_once'] = df_hava[col].shift(30)

df_hava['Son_30_Gun_Ort_Yagis'] = df_hava['Toplam_Yagis_mm'].rolling(window=30).mean()

# 6. VERİ BİRLEŞTİRME (Merging)
df_final = pd.merge(df_hal, df_hava, on='Tarih', how='left')

# 7. EKSİK VE HATALI VERİLERİ DÜŞÜRME (Drop NaNs and Outliers)
# İlk günlerin lag verileri boş olacağı için o satırları düşüyoruz
df_final.dropna(subset=['Fiyat_Degisim_Yuzdesi', 'Ort_Sicaklik_C_30_gun_once'], inplace=True)

# Veri giriş hatalarını (örneğin bir günde fiyat %500 artmış yazıldıysa) filtreleme
# Tarımda bir günde %100'den fazla artış veya %90'dan fazla düşüş genelde belediye memuru hatasıdır
df_final = df_final[(df_final['Fiyat_Degisim_Yuzdesi'] > -90) & (df_final['Fiyat_Degisim_Yuzdesi'] < 100)]

# 8. KAYDETME (Saving)
df_final['Tarih'] = df_final['Tarih'].dt.strftime('%d.%m.%Y')
output_file = os.path.join(base_path, "konya_tarim_final_engineered_dataset.csv")
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"HARİKA! Enflasyon düzeltmesi yapıldı ve hedef değişken eklendi.")
print(f"Tertemiz yeni satır sayısı: {len(df_final)}")