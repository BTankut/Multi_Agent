#!/bin/bash

# Multi-Agent sistemi için test betiği

echo "==== Multi-Agent Sistemi Test Betiği ===="
echo ""

# Proje dizinine git
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo "Proje dizini: $PROJECT_DIR"

# Gerekli dosyaların varlığını kontrol et
if [ ! -f "app.py" ]; then
    echo "HATA: app.py bulunamadı!"
    exit 1
fi

# .env dosyasını kontrol et
if [ ! -f ".env" ]; then
    echo "UYARI: .env dosyası bulunamadı! API anahtarı tanımlı olmayabilir."
    echo "API_KEY=... şeklinde bir .env dosyası oluşturmanız önerilir."
fi

# Streamlit uygulamasının çalışıp çalışmadığını kontrol et
STREAMLIT_PID=$(pgrep -f "streamlit run app.py")
if [ -z "$STREAMLIT_PID" ]; then
    echo "Streamlit uygulaması çalışmıyor. Başlatılıyor..."
    nohup streamlit run app.py > /dev/null 2>&1 &
    STREAMLIT_PID=$!
    echo "Streamlit uygulaması başlatıldı (PID: $STREAMLIT_PID)"
    echo "Uygulamanın başlaması için 5 saniye bekleniyor..."
    sleep 5
else
    echo "Streamlit uygulaması zaten çalışıyor (PID: $STREAMLIT_PID)"
fi

# Manuel test talimatlarını göster
echo ""
echo "Manuel test talimatları gösteriliyor..."
python3 tests/test_multi_agent.py

echo ""
echo "Test tamamlandı. Sonuçları değerlendirmek için Streamlit uygulamasını kullanın."
echo "URL: http://localhost:8501"
echo ""
echo "ÖZET DÜZELTMELER:"
echo "1. Model seçimi sorunu düzeltildi - 'Free models only' seçeneği artık SADECE ücretsiz modelleri seçiyor"
echo "   - Üç aşamalı ücretsiz model tespit mekanizması eklendi:"
echo "     a) Model ID veya isimde ':free' ifadesi aranır (OpenRouter API format)"
echo "     b) model_labels.json dosyasındaki 'free' etiketli modeller tespit edilir"
echo "     c) Fiyatlandırma bilgisine bakılır (en son öncelikli)"
echo "2. Etiketleme sorunu düzeltildi - mantık soruları artık 'reasoning_expert' VE 'math_expert' etiketlerini alıyor"
echo "3. Agent seçim algoritması büyük ölçüde iyileştirildi - önce modeller filtreleniyor, sonra etiketlere göre seçiliyor"
echo "4. Acil durum fallback mekanizması iyileştirildi - seçenek türlerine göre uygun fallback davranışı"
echo ""
echo "Aşağıdaki sorunlar çözüldü:"
echo "- Ücretsiz model konfigürasyonu sorunu: Free models only seçildiği halde ücretli modeller seçiliyordu"
echo "- Etiketleme sorunu: Mantık soruları için math_expert etiketi atanmıyordu"