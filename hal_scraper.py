import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# 1. Proje klasörüne erişim ve çalışma dizini ayarı
base_path = r"C:\Users\ismet\Desktop\Projects\Local_Agricultural_Price_and_Climate_Predictor"
if not os.path.exists(base_path):
    os.makedirs(base_path)

# 2. Tarihlerin Oluşturulması (Son 5 Yıl - Pazartesi ve Perşembe)
# Bugünden geriye doğru 5 yıl hesaplayalım
end_date = datetime.now() # Kodu ne zaman çalıştırırsan o günün tarihini alır
start_date = end_date - timedelta(days=5*365)

target_dates = []
current = start_date
while current <= end_date:
    # 0: Pazartesi, 3: Perşembe
    if current.weekday() in [0, 3]:
        target_dates.append(current)
    current += timedelta(days=1)

print(f"Toplam {len(target_dates)} adet tarih sorgulanacak...")

# 3. Scraping Ayarları
all_data = []
errors = []
BASE_URL = "https://www.konya.bel.tr/hal-fiyatlari" 

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 4. Kazıma Döngüsü
for date_obj in target_dates:
    # İŞTE BURASI DEĞİŞTİ: Site YYYY-MM-DD istiyor (Örn: 2026-03-30)
    date_str_url = date_obj.strftime("%Y-%m-%d") 
    
    # Bizim CSV dosyamızda temiz görünmesi için klasik format
    date_str_display = date_obj.strftime("%d.%m.%Y") 
    day_name = "Pazartesi" if date_obj.weekday() == 0 else "Perşembe"
    
    try:
        # parametre olarak 'tarih' kullanıyoruz ve URL formatındaki tarihi gönderiyoruz
        response = requests.get(BASE_URL, params={'tarih': date_str_url}, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tabloyu bul
            table = soup.find('table') 
            
            if table:
                rows = table.find_all('tr')
                
                # Tabloda veri satırı var mı kontrol et (başlık satırı hariç)
                if len(rows) <= 1:
                    errors.append(f"{date_str_display} {day_name} gününün verilerine ulaşılamadı (Tatil veya veri girilmemiş).")
                    continue
                
                # İlk satır genelde başlıktır (Ürün, Birim, En Düşük, En Yüksek vb.), onu atlıyoruz [1:]
                for row in rows[1:]:
                    cols = row.find_all(['td', 'th']) # Bazen td yerine th kullanabilirler
                    cols = [ele.text.strip() for ele in cols]
                    
                    if len(cols) >= 3: # En az 3 sütun varsa (Ürün, Birim, Fiyat) işlem yap
                        all_data.append({
                            'Tarih': date_str_display,
                            'Gün': day_name,
                            'Ürün Adı': cols[0] if len(cols) > 0 else "",
                            'Birim': cols[1] if len(cols) > 1 else "",
                            'En Düşük Fiyat': cols[2] if len(cols) > 2 else "",
                            'En Yüksek Fiyat': cols[3] if len(cols) > 3 else ""
                        })
            else:
                errors.append(f"{date_str_display} {day_name} sayfasında tablo yapısı bulunamadı.")
        else:
            errors.append(f"{date_str_display} {day_name} günü için sunucu hatası: {response.status_code}")
            
    except Exception as e:
        errors.append(f"{date_str_display} {day_name} gününde teknik hata oluştu: {str(e)}")
    
    # Sunucuyu yormamak için kısa bir bekleme (Çok önemli!)
    time.sleep(0.5)

# 5. Verileri Kaydetme
if all_data:
    df = pd.DataFrame(all_data)
    csv_file = os.path.join(base_path, "konya_hal_fiyatlari_5yillik.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\nMÜKEMMEL! İşlem tamamlandı. {len(all_data)} satır veri başarıyla çekildi ve CSV'ye kaydedildi.")
else:
    print("\nMaalesef hala hiç veri çekilemedi. HTML yapısını incelememiz gerekecek.")

# 6. Hataları Raporlama
log_file = os.path.join(base_path, "scraping_hatalari.txt")
with open(log_file, "w", encoding="utf-8") as f:
    for error in errors:
        f.write(error + "\n")

print(f"Ulaşılamayan tarihlerin listesi '{log_file}' dosyasına kaydedildi.")