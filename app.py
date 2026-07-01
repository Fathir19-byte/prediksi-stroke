import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Risiko Stroke - AI",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }

    .result-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-safe {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #34d399;
    }
    .result-safe h2 { color: #065f46; }
    .result-safe p { color: #047857; }

    .result-danger {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #f87171;
    }
    .result-danger h2 { color: #991b1b; }
    .result-danger p { color: #b91c1c; }

    .info-card {
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border: 1px solid #c4b5fd;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .info-card h4 { color: #5b21b6; margin-bottom: 0.5rem; }
    .info-card p { color: #6d28d9; font-size: 0.9rem; }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 Prediksi Risiko Stroke</h1>
    <p>Aplikasi Kecerdasan Buatan untuk memprediksi risiko penyakit stroke berdasarkan data medis pasien</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & FEATURE COLUMNS
# ============================================================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_stroke_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_feature_columns():
    cols_path = os.path.join(os.path.dirname(__file__), 'feature_columns.pkl')
    with open(cols_path, 'rb') as f:
        cols = pickle.load(f)
    return cols

try:
    model = load_model()
    feature_columns = load_feature_columns()
except FileNotFoundError:
    st.error("⚠️ File model (`best_stroke_model.pkl`) atau `feature_columns.pkl` tidak ditemukan. "
             "Pastikan kamu sudah menjalankan notebook.ipynb terlebih dahulu untuk melatih dan menyimpan model.")
    st.stop()

# ============================================================
# FORM INPUT DATA PASIEN
# ============================================================
st.markdown("### 📋 Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.slider("Usia (Tahun)", min_value=1, max_value=100, value=45)
    hypertension = st.selectbox("Riwayat Hipertensi?", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Riwayat Penyakit Jantung?", ["Tidak", "Ya"])
    ever_married = st.selectbox("Status Pernikahan", ["Belum Menikah", "Sudah Menikah"])

with col2:
    work_type = st.selectbox("Jenis Pekerjaan", ["Swasta", "Wiraswasta", "Pemerintahan", "Anak-anak", "Belum Bekerja"])
    residence = st.selectbox("Tipe Tempat Tinggal", ["Perkotaan", "Pedesaan"])
    avg_glucose = st.number_input("Rata-rata Kadar Glukosa (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    smoking_status = st.selectbox("Status Merokok", ["Tidak Pernah", "Pernah Merokok", "Merokok Aktif", "Tidak Diketahui"])

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def prepare_input():
    """Menyiapkan data input agar sesuai dengan format fitur model."""
    data = {
        'age': age,
        'hypertension': 1 if hypertension == "Ya" else 0,
        'heart_disease': 1 if heart_disease == "Ya" else 0,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        # One-Hot Encoded columns (drop_first=True)
        'gender_Male': 1 if gender == "Laki-laki" else 0,
        'ever_married_Yes': 1 if ever_married == "Sudah Menikah" else 0,
        'work_type_Never_worked': 1 if work_type == "Belum Bekerja" else 0,
        'work_type_Private': 1 if work_type == "Swasta" else 0,
        'work_type_Self-employed': 1 if work_type == "Wiraswasta" else 0,
        'work_type_children': 1 if work_type == "Anak-anak" else 0,
        'Residence_type_Urban': 1 if residence == "Perkotaan" else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == "Pernah Merokok" else 0,
        'smoking_status_never smoked': 1 if smoking_status == "Tidak Pernah" else 0,
        'smoking_status_smokes': 1 if smoking_status == "Merokok Aktif" else 0,
    }

    # Buat DataFrame dan pastikan kolomnya sesuai urutan model
    input_df = pd.DataFrame([data])

    # Tambahkan kolom yang mungkin tidak ada (isi 0) dan urutkan sesuai fitur model
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    return input_df

# ============================================================
# TOMBOL PREDIKSI & HASIL
# ============================================================
st.markdown("---")

if st.button("🔍 Prediksi Risiko Stroke"):
    input_data = prepare_input()
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    prob_stroke = probability[1] * 100
    prob_safe = probability[0] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box result-danger">
            <h2>⚠️ RISIKO TINGGI TERKENA STROKE</h2>
            <p style="font-size: 1.5rem; font-weight: 700;">{prob_stroke:.1f}% kemungkinan stroke</p>
            <p>Pasien disarankan untuk segera berkonsultasi dengan dokter spesialis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-safe">
            <h2>✅ RISIKO RENDAH TERKENA STROKE</h2>
            <p style="font-size: 1.5rem; font-weight: 700;">{prob_safe:.1f}% kemungkinan sehat</p>
            <p>Pasien memiliki risiko rendah, tetapi tetap jaga pola hidup sehat.</p>
        </div>
        """, unsafe_allow_html=True)

    # Menampilkan detail probabilitas
    st.markdown("""
    <div class="info-card">
        <h4>📊 Detail Probabilitas Model</h4>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    col_r1.metric("Probabilitas Sehat", f"{prob_safe:.1f}%")
    col_r2.metric("Probabilitas Stroke", f"{prob_stroke:.1f}%")

# ============================================================
# INFORMASI TAMBAHAN
# ============================================================
st.markdown("""
<div class="info-card">
    <h4>ℹ️ Tentang Aplikasi Ini</h4>
    <p>
        Aplikasi ini menggunakan algoritma <b>Random Forest Classifier</b> yang dilatih menggunakan dataset
        <i>Healthcare Dataset - Stroke Data</i> dari Kaggle. Model ini dibangun mengikuti kerangka kerja
        <b>CRISP-DM</b> sebagai tugas mata kuliah Kecerdasan Buatan.
    </p>
    <p style="margin-top: 0.5rem;">
        <b>⚠️ Disclaimer:</b> Hasil prediksi ini hanya bersifat edukatif dan BUKAN diagnosis medis.
        Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis profesional.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Tugas Tubes Individu — Mata Kuliah Kecerdasan Buatan | CRISP-DM Framework</p>
</div>
""", unsafe_allow_html=True)
