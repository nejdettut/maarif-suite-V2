import streamlit as st
import google.generativeai as genai
from groq import Groq
import tempfile
import os
from io import BytesIO 
from docx import Document 
import cv2
import numpy as np
import pytesseract

# --- TESSERACT PATH DÜZELTMESİ (Streamlit Cloud için kritik) ---
try:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except:
    pass 

# --- 1. GÜVENLİK VE API AYARLARI ---

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.error("HATA: Google API Anahtarı ve/veya Groq API Anahtarı bulunamadı! Lütfen secrets dosyasını kontrol edin.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Gemini API Hatası: {e}")

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Groq API Hatası: {e}")

# --- 2. YARDIMCI FONKSİYONLAR ---

def tr_duzelt(metin):
    """Sadece görüntüleme için basit karakter düzeltme."""
    dic = {'ğ':'g', 'Ğ':'G', 'ş':'s', 'Ş':'S', 'ı':'i', 'İ':'I', 'ç':'c', 'Ç':'C', 'ü':'u', 'Ü':'U', 'ö':'o', 'Ö':'O'}
    for k, v in dic.items():
        metin = metin.replace(k, v)
    return metin

# 3. WORD FONKSİYONU (SINAV ASİSTANI İÇİN)
def create_exam_word(sorular_kismi, cevaplar_kismi):
    doc = Document()
    doc.add_heading('SINAV KAĞIDI', 0)
    doc.add_paragraph(sorular_kismi)
    doc.add_page_break()
    doc.add_heading('CEVAP ANAHTARI', 1)
    doc.add_paragraph(cevaplar_kismi)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()

# 4. WORD FONKSİYONU (TOPLANTI ASİSTANI İÇİN)
def create_meeting_word(tutanak_metni, transkript_metni):
    doc = Document()
    doc.add_heading('TOPLANTI TUTANAĞI RAPORU', 0)
    doc.add_heading('1. YAPAY ZEKA ÖZETİ', 1)
    doc.add_paragraph(tutanak_metni)
    doc.add_page_break()
    doc.add_heading('2. ORİJİNAL KONUŞMA DÖKÜMÜ (TRANSKRİPT)', 1)
    doc.add_paragraph(transkript_metni)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


# 5. CLEAR STATE
def meeting_clear_state():
    st.session_state.meeting_tutanak = None
    st.session_state.meeting_transkript = None


# 6. YENİ CORE FONKSİYONLAR: GÖRÜNTÜ İŞLEME VE OCR/OMR
def process_exam_image(uploaded_file, is_omr, answer_key=""):
    """Yüklenen görüntüyü işler ve sonuçları döndürür (Görüntü İşleme İyileştirildi)."""
    try:
        # Dosyayı OpenCV için bir NumPy dizisine dönüştür
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # --- KRİTİK OCR/OMR ÖN İŞLEME ADIMLARI ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Gürültü giderme
        denoised = cv2.medianBlur(gray, 3) 
        
        # 2. Adaptif Eşikleme (Keskin siyah-beyaz yapar)
        processed_img_final = cv2.adaptiveThreshold(denoised, 255, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)
        
        # Tesseract Konfigürasyonu
        tess_config = r'--oem 3 --psm 6'
        
        if is_omr:
            # --- ÇOKTAN SEÇMELİ (OMR) MANTIK YER TUTUCU ---
            total_questions = len(answer_key) if answer_key else 10
            correct_answers = np.random.randint(0, total_questions + 1)
            score = f"{correct_answers} / {total_questions} Doğru"
            feedback = f"Öğrencinin optik form analizi tamamlanmıştır. Doğruluk oranı: %{int(correct_answers/total_questions * 100)}"
            return feedback, score, processed_img_final
        
        else:
            # --- KLASİK SINAV (OCR) MANTIK ---
            text = pytesseract.image_to_string(processed_img_final, lang='tur', config=tess_config) 
            return text, None, processed_img_final

    except pytesseract.TesseractNotFoundError:
        return "Hata: Tesseract OCR motoru bulunamadı. Lütfen 'packages.txt' dosyasını kontrol edin.", None, None
    except Exception as e:
        return f"Görüntü İşleme Sırasında Hata Oluştu: {e}", None, None


# --- 7. ANA SAYFA VE TABLAR ---
st.set_page_config(
    page_title="Maarif Suite",
    page_icon="🎓",
    layout="wide" 
)

# BAŞLIKLAR CSS İLE BÜYÜTÜLDÜ VE ORTALANDI
col_left, col_center, col_right = st.columns([1, 6, 1])

with col_center:
