import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

from core.preprocessing import clean_text

MODEL_PATH = os.path.join('saved_models', 'svm_sentimen_3kelas.pkl')

st.set_page_config(
    page_title="Klasifikasi Sentimen Ulasan Google Maps",
    page_icon="🗺️",
    layout="wide"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data['model'], data['vectorizer']

THRESHOLD = 0.60

model, vectorizer = load_model()

if model is None:
    st.warning("Model belum dilatih! Jalankan train_local.py terlebih dahulu")
    st.stop()

def predict_with_netral(text_vec):
    proba = model.predict_proba(text_vec)[0]
    confidence = proba.max()
    if confidence < THRESHOLD:
        return 'NETRAL', confidence
    pred = model.classes_[proba.argmax()]
    return pred, confidence

st.markdown("""
    <style>
    .sentimen-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .positif {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .negatif {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .netral {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Klasifikasi Sentimen Ulasan Wisatawan")
st.markdown("**TF-IDF + Support Vector Machine (SVM)** — Destinasi Wisata di Pacitan, Jawa Timur")
st.divider()

page = st.sidebar.radio("Menu", ["Analisis Teks Manual", "Analisis File CSV & Dashboard", "Data Scraping"])

sentimen_color = {'POSITIF': 'positif', 'NEGATIF': 'negatif', 'NETRAL': 'netral'}

if page == "Analisis Teks Manual":
    st.header("✍️ Analisis Teks Manual")
    st.markdown("Masukkan ulasan wisatawan secara manual untuk mengetahui sentimennya.")

    user_input = st.text_area(
        "Teks ulasan:",
        placeholder="Contoh: Tempatnya sangat indah dan nyaman untuk liburan keluarga...",
        height=150
    )

    if st.button("🔍 Analisis Sentimen", type="primary"):
        if user_input.strip():
            with st.spinner("Menganalisis sentimen..."):
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                pred, confidence = predict_with_netral(vec)
                proba = model.predict_proba(vec)[0]
                classes = list(model.classes_) + (['NETRAL'] if 'NETRAL' not in model.classes_ else [])
                score = confidence * 100

            st.subheader("Hasil Analisis")
            css_class = sentimen_color[pred]
            st.markdown(
                f'<div class="sentimen-box {css_class}">Sentimen: {pred} '
                f'<br><span style="font-size:0.9rem;">Keyakinan: {score:.2f}%</span></div>',
                unsafe_allow_html=True
            )

            st.markdown("**Probabilitas per kelas:**")
            all_probs = {c: 0.0 for c in classes}
            for i, c in enumerate(model.classes_):
                all_probs[c] = proba[i]
            prob_positif = all_probs.get('POSITIF', 0)
            prob_negatif = all_probs.get('NEGATIF', 0)
            prob_netral = 1 - prob_positif - prob_negatif
            all_probs['NETRAL'] = prob_netral

            cols = st.columns(3)
            for col, cls in zip(cols, ['POSITIF', 'NEGATIF', 'NETRAL']):
                with col:
                    st.metric(label=cls, value=f"{all_probs[cls]*100:.2f}%")
        else:
            st.error("Silakan masukkan teks ulasan terlebih dahulu.")

elif page == "Analisis File CSV & Dashboard":
    st.header("📁 Analisis File CSV & Dashboard")
    st.markdown("Unggah file CSV (dengan kolom teks ulasan) untuk memprediksi sentimen secara massal.")

    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df_input = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')

        st.subheader("📄 Pratinjau Data")
        st.dataframe(df_input.head(), use_container_width=True)

        df_input.columns = df_input.columns.str.strip()
        possible_text_cols = ['Review', 'ulasan', 'review', 'text', 'content', 'komentar', 'Ulasan']
        text_col = None
        for col in possible_text_cols:
            if col in df_input.columns:
                text_col = col
                break

        if text_col is None:
            text_col = df_input.columns[0]
            st.info(f"Kolom teks otomatis: **{text_col}**")

        if st.button("🚀 Prediksi Semua", type="primary"):
            with st.spinner(f"Memproses {len(df_input)} ulasan..."):
                texts = df_input[text_col].astype(str).apply(clean_text)
                X_vec = vectorizer.transform(texts)
                results = [predict_with_netral(X_vec[i:i+1]) for i in range(X_vec.shape[0])]
                predictions, confidences = zip(*results)
                df_input['Sentimen'] = predictions
                df_input['Keyakinan (%)'] = (np.array(confidences) * 100).round(2)

            st.subheader("✅ Hasil Prediksi")
            st.dataframe(df_input, use_container_width=True)

            csv_download = df_input.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil CSV",
                data=csv_download,
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Proporsi Sentimen")
                sentimen_counts = df_input['Sentimen'].value_counts()
                colors = {'POSITIF': '#28a745', 'NEGATIF': '#dc3545', 'NETRAL': '#ffc107'}
                pie_colors = [colors.get(s, '#6c757d') for s in sentimen_counts.index]

                fig1, ax1 = plt.subplots(figsize=(6, 6))
                wedges, texts, autotexts = ax1.pie(
                    sentimen_counts,
                    labels=sentimen_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=pie_colors,
                    textprops={'fontsize': 12, 'weight': 'bold'}
                )
                ax1.axis('equal')
                st.pyplot(fig1)

            with col2:
                st.subheader("☁️ Word Cloud")
                all_text = ' '.join(map(str, texts))
                if all_text.strip():
                    wc = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(all_text)

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.imshow(wc, interpolation='bilinear')
                    ax2.axis('off')
                    st.pyplot(fig2)
                else:
                    st.info("Tidak ada teks untuk ditampilkan di Word Cloud.")
    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

elif page == "Data Scraping":
    st.header("🕷️ Data Scraping Google Maps")
    st.markdown("Hasil scraping review destinasi wisata Pacitan menggunakan SerpAPI.")

    scraped_path = os.path.join('dataset', 'GMaps_Review_Scraped.csv')

    if not os.path.exists(scraped_path):
        st.warning("File dataset/scraping belum tersedia. Jalankan `python scrape_reviews.py` terlebih dahulu.")
        st.stop()

    df_scraped = pd.read_csv(scraped_path)
    st.success(f"Menampilkan {len(df_scraped)} review dari hasil scraping.")

    with st.spinner("Memprediksi sentimen..."):
        texts = df_scraped['Review'].astype(str).apply(clean_text)
        X_vec = vectorizer.transform(texts)
        results = [predict_with_netral(X_vec[i:i+1]) for i in range(X_vec.shape[0])]
        predictions, confidences = zip(*results)
        df_scraped['Sentimen'] = predictions
        df_scraped['Keyakinan (%)'] = (np.array(confidences) * 100).round(2)

    st.subheader("📋 Data Hasil Scraping + Prediksi")
    st.dataframe(df_scraped, use_container_width=True)

    csv_download = df_scraped.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Hasil CSV",
        data=csv_download,
        file_name="GMaps_Review_Scraped_Labeled.csv",
        mime="text/csv"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Proporsi Sentimen")
        sentimen_counts = df_scraped['Sentimen'].value_counts()
        colors = {'POSITIF': '#28a745', 'NEGATIF': '#dc3545', 'NETRAL': '#ffc107'}
        pie_colors = [colors.get(s, '#6c757d') for s in sentimen_counts.index]

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax1.pie(
            sentimen_counts,
            labels=sentimen_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        st.subheader("☁️ Word Cloud")
        all_text = ' '.join(map(str, texts))
        if all_text.strip():
            wc = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(all_text)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        else:
            st.info("Tidak ada teks untuk ditampilkan di Word Cloud.")

    st.divider()
    st.subheader("🏷️ Distribusi per Destinasi")
    pivot = df_scraped.groupby(['Nama Tempat', 'Sentimen']).size().unstack(fill_value=0)
    st.dataframe(pivot, use_container_width=True)
