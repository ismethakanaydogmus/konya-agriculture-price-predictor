import pandas as pd
import os

# 1. Dosya Yolları (File Paths)
base_path = "C:/Users/ismet/Desktop/Projects/Local_Agricultural_Price_and_Climate_Predictor"
raw_csv_file = os.path.join(base_path, "konya_hal_fiyatlari_5yillik.csv")
clean_csv_file = os.path.join(base_path, "konya_hal_fiyatlari_temiz_5yillik.csv")

print("Veri temizleme işlemi başlıyor...")

# 2. Ham Veriyi Oku (Read the Raw Data)
df = pd.read_csv(raw_csv_file)
ilk_satir_sayisi = len(df)

# 3. Temizleme (Cleaning)
# 'Ürün Adı' sütununda 'Ürün' kelimesi geçen (büyük/küçük harf duyarsız) satırları bul ve DIŞARIDA bırak (~)
df_clean = df[~df['Ürün Adı'].str.contains('Ürün', na=False, case=False)]

# İstersen boş olan satırları (NaN) da bu aşamada silebiliriz
df_clean = df_clean.dropna(subset=['Ürün Adı', 'En Düşük Fiyat'])

# 4. Temiz Veriyi Kaydet (Save the Clean Data)
df_clean.to_csv(clean_csv_file, index=False, encoding='utf-8-sig')
son_satir_sayisi = len(df_clean)

# 5. Rapor (Report)
print(f"İşlem Tamamlandı!")
print(f"Temizlenmeden Önceki Satır Sayısı : {ilk_satir_sayisi}")
print(f"Temizlendikten Sonraki Satır Sayısı : {son_satir_sayisi}")
print(f"Toplam {ilk_satir_sayisi - son_satir_sayisi} adet hatalı/başlık satırı silindi.")
print(f"Temiz dosyanız şu adrese kaydedildi: {clean_csv_file}")