import requests
import json
import time
import sys
import os
from pathlib import Path

# Ana proje dizinini ekleyin
sys.path.append(str(Path(__file__).parent.parent))

# Test senaryoları - Önceden tanımlanmış farklı sorgu türleri
TEST_SCENARIOS = [
    {
        "name": "Mantık/Akıl Yürütme Sorusu",
        "query": """
        Üç kız kardeş var: Ayşe, Fatma ve Zeynep. Birisi kırmızı, birisi mavi, birisi yeşil elbise giyiyor. 
        Kırmızı elbise giyen kız kardeş, Ayşe ile kardeş değildir. 
        Fatma, mavi elbise giymiyor. 
        Zeynep en büyük kardeştir.
        Hangi kardeş hangi renk elbise giyiyor?
        """,
        "expected_labels": ["reasoning_expert"]
    },
    {
        "name": "Matematik Sorusu",
        "query": """
        Bir fabrika saatte 120 parça üretiyor. Fabrikanın günlük çalışma süresi 8 saattir. 
        Her ayın ilk haftası bakım çalışmaları nedeniyle üretim %30 düşmektedir. 
        Bu fabrika 30 günlük bir ayda toplam kaç parça üretir?
        """,
        "expected_labels": ["math_expert"]
    },
    {
        "name": "Kodlama Sorusu",
        "query": """
        Python'da bir liste içindeki en sık tekrar eden elemanı bulan bir fonksiyon yazabilir misin? 
        Adım adım açıkla ve örnek bir kullanım göster.
        """,
        "expected_labels": ["code_expert"]
    },
    {
        "name": "Genel Bilgi Sorusu",
        "query": """
        Kapadokya bölgesi hakkında bilgi verir misin? Nerede bulunur, hangi şehirleri kapsar 
        ve turistler tarafından neden ziyaret edilir?
        """,
        "expected_labels": ["general_assistant"]
    },
    {
        "name": "Çoklu Etiketli Soru",
        "query": """
        Bir yazılım şirketi için yapay zeka destekli müşteri hizmetleri sistemi tasarlarken, 
        hangi matematiksel modeller kullanılmalıdır ve bu sistemin kodlaması için en uygun 
        programlama dili hangisidir? Cevabını matematiksel algoritmaları ve kod örneklerini 
        içerecek şekilde detaylandır.
        """,
        "expected_labels": ["math_expert", "code_expert"]
    }
]

# Test seçenekleri
TEST_OPTIONS = [
    "Free models only",
    "Paid models only", 
    "Optimized mix of free and paid models"
]

def test_multi_agent_manually():
    """
    Manuel test için talimatları yazdır
    """
    print("\n==== Multi-Agent Sistemi Manuel Test Talimatları ====\n")
    
    print("Bu test dosyası, Multi-Agent sisteminin farklı sorgu türleri ve seçeneklerle")
    print("doğru çalışıp çalışmadığını manuel olarak test etmek için bir kılavuzdur.")
    print("\nÖNCE MUTLAKA UYGULAMAYI ÇALIŞTIRIN: streamlit run app.py\n")
    
    print("TEST ADIMLARI:")
    print("1. Streamlit uygulamasını açın")
    print("2. 'Fetch OpenRouter Models' butonuna tıklayın")
    print("3. Aşağıdaki seçenekleri ve sorguları test edin:")
    
    for option in TEST_OPTIONS:
        print(f"\n--- {option} ---")
        for scenario in TEST_SCENARIOS:
            print(f"\n* {scenario['name']}")
            print(f"  Sorgu: {scenario['query'].strip()}")
            print(f"  Beklenen Etiketler: {scenario['expected_labels']}")
            
    print("\nHER TEST İÇİN KONTROL EDİLECEKLER:")
    print("- Query Analysis sekmesinde doğru etiketlerin gösterilmesi")
    print("- Doğru sayıda ajanın seçilmesi (matematik/kodlama için en az 2)")
    print("- Ajan derecelendirmelerinin ve grafiklerin doğru görüntülenmesi")
    print("- Seçilen ajanın doğru etiketlere sahip olması")
    print("- Sonuç ekranında yeterli bilginin gösterilmesi")
    
    print("\nNOT: Bu testler manuel olarak yapılmalıdır, otomatik test için")
    print("Streamlit'in test API'si veya Selenium gibi bir browser otomasyon")
    print("aracı kullanılabilir.")

if __name__ == "__main__":
    test_multi_agent_manually()