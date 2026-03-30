import requests
import pandas as pd
import os

# 1. Proje Klasörü (Project Folder)
base_path = r"C:\Users\ismet\Desktop\Projects\Local_Agricultural_Price_and_Climate_Predictor"

# 2. Konya'nın Koordinatları ve Tarih Aralığı (Konya's Coordinates and Date Range)
# Konya Merkez: Enlem (Latitude) ~37.87, Boylam (Longitude) ~32.49
LATITUDE = 37.8746
LONGITUDE = 32.4932

# Hal fiyatlarını 30.03.2021 ile 30.03.2026 arası çektik. Aynı aralığı veriyoruz.
START_DATE = "2021-03-30"
END_DATE = "2026-03-30"

print(f"Konya için {START_DATE} ile {END_DATE} arası hava durumu verileri çekiliyor...")

# 3. Open-Meteo API URL'sini Oluşturma (Building the Open-Meteo API URL)
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}&"
    f"start_date={START_DATE}&end_date={END_DATE}&"
    f"daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum&"
    f"timezone=Europe%2FIstanbul"
)

try:
    # 4. API'ye İstek Atma (Making the API Request)
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # JSON'dan 'daily' (günlük) verilerini alıp DataFrame'e çeviriyoruz
        daily_data = data['daily']
        df_weather = pd.DataFrame({
            'Tarih': daily_data['time'],
            'Maks_Sicaklik_C': daily_data['temperature_2m_max'],
            'Min_Sicaklik_C': daily_data['temperature_2m_min'],
            'Ort_Sicaklik_C': daily_data['temperature_2m_mean'],
            'Toplam_Yagis_mm': daily_data['precipitation_sum']
        })
        
        # Tarih formatını hal fiyatları dosyamızla eşleşmesi için gün.ay.yıl formatına çeviriyoruz
        df_weather['Tarih'] = pd.to_datetime(df_weather['Tarih']).dt.strftime('%d.%m.%Y')
        
        # 5. CSV Olarak Kaydetme (Saving as CSV)
        weather_csv_file = os.path.join(base_path, "konya_hava_durumu_5yillik.csv")
        df_weather.to_csv(weather_csv_file, index=False, encoding='utf-8-sig')
        
        print(f"MÜKEMMEL! İşlem Tamamlandı. {len(df_weather)} günlük iklim verisi başarıyla kaydedildi.")
        print(f"Dosya konumu: {weather_csv_file}")
        
    else:
        print(f"API Hatası: Sunucu {response.status_code} kodunu döndürdü.")

except Exception as e:
    print(f"Veri çekilirken bir hata oluştu: {e}")