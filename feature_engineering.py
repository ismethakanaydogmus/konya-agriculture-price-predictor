import pandas as pd
import os

# 1. Dosya Yolları
base_path = r"C:\Users\ismet\Desktop\Projects\Local_Agricultural_Price_and_Climate_Predictor"
hal_csv = os.path.join(base_path, "konya_hal_fiyatlari_temiz_5yillik.csv")
hava_csv = os.path.join(base_path, "konya_hava_durumu_5yillik.csv")

# 2. Verileri Yükle
df_hal = pd.read_csv(hal_csv)
df_hava = pd.read_csv(hava_csv)

# Tarih formatlarını datetime nesnesine çevirelim (Sıralama ve kaydırma için şart)
df_hal['Tarih'] = pd.to_datetime(df_hal['Tarih'], format='%d.%m.%Y')
df_hava['Tarih'] = pd.to_datetime(df_hava['Tarih'], format='%d.%m.%Y')

# Verileri tarihe göre sıralayalım
df_hava = df_hava.sort_values('Tarih')

# 3. Hava Durumu İçin Gecikmeli Veriler (Lags) Oluşturma
# Sıcaklık ve yağış için 7, 14 ve 30 günlük geçmiş verileri ekleyelim
for col in ['Ort_Sicaklik_C', 'Toplam_Yagis_mm']:
    df_hava[f'{col}_7_gun_once'] = df_hava[col].shift(7)
    df_hava[f'{col}_14_gun_once'] = df_hava[col].shift(14)
    df_hava[f'{col}_30_gun_once'] = df_hava[col].shift(30)

# Hareketli Ortalama (Rolling Average): Son 30 günlük ortalama yağış
# Bu, kuraklık veya aşırı yağış dönemlerini anlamak için çok kritiktir
df_hava['Son_30_Gun_Ort_Yagis'] = df_hava['Toplam_Yagis_mm'].rolling(window=30).mean()

# 4. Fiyat Verisi İçin Gecikmeli Veri (Önceki Seans Fiyatı)
# Önce ürün bazında gruplayıp sonra kaydırma yapmalıyız
df_hal = df_hal.sort_values(['Ürün Adı', 'Tarih'])
df_hal['Onceki_Seans_En_Dusuk_Fiyat'] = df_hal.groupby('Ürün Adı')['En Düşük Fiyat'].shift(1)

# 5. Zenginleştirilmiş Verileri Birleştirme
df_final = pd.merge(df_hal, df_hava, on='Tarih', how='left')

# 6. Kaydetme
# Tekrar okumayı kolaylaştırmak için tarihi metin formatına geri çevirebiliriz
df_final['Tarih'] = df_final['Tarih'].dt.strftime('%d.%m.%Y')

output_file = os.path.join(base_path, "konya_tarim_final_engineered_dataset.csv")
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Özellik mühendisliği tamamlandı! Yeni sütunlar eklendi.")
print(f"Yeni satır sayısı: {len(df_final)}")
print(f"Dosya kaydedildi: {output_file}")