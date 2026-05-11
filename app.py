import streamlit as st
import joblib
import pandas as pd

try:
    model = joblib.load("model.pkl")
except:
    class DummyModel:
        def predict(self, x): return ["negative"]
    model = DummyModel()

st.set_page_config(page_title="AI Text Classifier", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #FDFCF0 0%, #3B82F6 50%, #1E3A8A 100%) !important;
        background-attachment: fixed;
    }
    .custom-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border-radius: 30px; 
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15); 
        margin-bottom: 25px;
    }
    div.stButton > button {
        background-color: white !important;
        border: none !important;
        border-radius: 15px !important;
        height: 50px !important;
        padding: 0 30px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out !important; 
    }
    div.stButton > button:hover {
        transform: translateY(-3px); 
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    div.stButton > button p {
        color: #1E3A8A !important; 
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    .stTextArea textarea {
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        padding: 15px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        border: 2px dashed rgba(255,255,255,0.3) !important;
    }
    h3 { color: white !important; font-weight: 600 !important; }
    .glow-text {
        font-size: 3.5rem;
        text-align: center;
        color: #1E3A8A;
        font-weight: 800;
    }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="glow-text">SENTIMENT ANALYZER</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #1E3A8A !important; font-weight: 500; margin-bottom: 40px;'>Klasifikasi Teks Berbasis Kecerdasan Buatan</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="custom-card"><h3>📝 Analisis Tunggal</h3><p style="color: rgba(255,255,255,0.8);">Prediksi sentimen kalimat secara instan.</p></div>', unsafe_allow_html=True)

    text_input = st.text_area("Input", placeholder="Ketik pesan Anda di sini...", height=150, label_visibility="collapsed")

    if st.button("ANALISIS SEKARANG"):
        if text_input.strip():
            prediction = model.predict([text_input.lower()])
            res_text = prediction[0]

            st.markdown(f"""
                <div style="background: white; padding: 25px; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-top: 20px;">
                    <small style="color: #64748b; font-weight: bold; letter-spacing: 1px;">HASIL SENTIMEN</small>
                    <h2 style="margin: 5px 0 0 0; color: #1E3A8A; font-weight: 800; font-size: 2rem;">{str(res_text).upper()}</h2>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

with col2:
    st.markdown('<div class="custom-card"><h3>📂 Analisis Batch</h3><p style="color: rgba(255,255,255,0.8);">Proses banyak data sekaligus via file CSV.</p></div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")

    if file:
        df = pd.read_csv(file)

        if "text" in df.columns:

            with st.spinner('Menganalisis...'):
                df["hasil"] = model.predict(df["text"].astype(str).str.lower())

            st.success("Analisis Selesai!")

            st.write("📄 Preview Data")
            st.dataframe(df.head())

            st.write("📊 Distribusi Hasil")
            st.bar_chart(df["hasil"].value_counts())

            st.write("📈 Persentase Sentimen")

            total = len(df)
            counts = df["hasil"].value_counts()

            for label, count in counts.items():
                st.write(f"**{label}** : {(count/total)*100:.2f}%")

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇️ Download Hasil",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )

        else:
            st.error("Kolom 'text' tidak ditemukan.")