import pandas as pd
import os

# 1. Dosya Yollarını Tanımla
base_path = r"C:\Users\ismet\Desktop\Projects\Local_Agricultural_Price_and_Climate_Predictor"
hal_csv = os.path.join(base_path, "konya_hal_fiyatlari_temiz_5yillik.csv")
hava_csv = os.path.join(base_path, "konya_hava_durumu_5yillik.csv")

print("Veriler okunuyor...")

# 2. Veri Setlerini Yükle
df_hal = pd.read_csv(hal_csv)
df_hava = pd.read_csv(hava_csv)

# 3. Veri Birleştirme (Merging)
# 'Tarih' sütunu üzerinden 'left' join yapıyoruz. 
# Böylece tüm hal fiyatlarını koruyoruz ve o tarihlere denk gelen hava durumunu yanına ekliyoruz.
df_final = pd.merge(df_hal, df_hava, on='Tarih', how='left')

# 4. Eksik Veri Kontrolü
# Eğer hal verisinde olup hava durumu verisinde olmayan bir tarih varsa orası boş (NaN) kalacaktır.
eksik_gunler = df_final['Ort_Sicaklik_C'].isnull().sum()

# 5. Sonuç Dosyasını Kaydet
output_file = os.path.join(base_path, "konya_tarim_ve_iklim_final_dataset.csv")
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"İşlem Başarıyla Tamamlandı!")
print(f"Birleştirilmiş Veri Satır Sayısı: {len(df_final)}")
if eksik_gunler > 0:
    print(f"Uyarı: {eksik_gunler} adet satırda hava durumu verisi eşleşmedi.")
print(f"Final veri setiniz kaydedildi: {output_file}")