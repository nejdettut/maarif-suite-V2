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
def create_meeting_word(tutanak_metni, transkript
