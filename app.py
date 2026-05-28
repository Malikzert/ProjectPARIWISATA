import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

from core.preprocessing import clean_text

MODEL_PATH = os.path.join('saved_models', 'svm_sentimen_3kelas.pkl')
DATASET_PRE_PATH = os.path.join('dataset', 'GMaps_Review_Preprocessed.csv')
DATASET_LABELED_PATH = os.path.join('dataset', 'GMaps_Review_Labeled.csv')

st.set_page_config(
    page_title="Klasifikasi Sentimen Ulasan Google Maps",
    page_icon="🗺️",
    layout="wide"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

BG_COLOR = '#1e1e1e' if st.session_state.dark_mode else 'white'
TEXT_COLOR = '#f0f0f0' if st.session_state.dark_mode else '#333'
CARD_BG = '#2d2d2d' if st.session_state.dark_mode else '#f8f9fa'

dark_css = """
<style>
.stApp {
    background-color: #1e1e1e;
    color: #f0f0f0;
}
.stSidebar, .css-1d391kg, .css-163ttbj, .css-1wrcr25, .css-1v3fvcr {
    background-color: #252525 !important;
    color: #f0f0f0 !important;
}
.stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, span {
    color: #f0f0f0 !important;
}
.st-bb, .st-at, .st-ae, .st-af, .st-ag {
    background-color: #2d2d2d !important;
}
.css-1n76uvr, .css-17z71as, .css-1cpxqw2, .css-15zrgzn {
    color: #f0f0f0 !important;
}
.stTextInput > div > div > input {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
.stTextArea > div > div > textarea {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
.stSelectbox > div > div {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #f0f0f0 !important;
}
.stDataFrame {
    background-color: #2d2d2d !important;
}
div[data-testid="stDataFrame"] th {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
div[data-testid="stDataFrame"] td {
    background-color: #2d2d2d !important;
    color: #f0f0f0 !important;
}
div.stAlert {
    background-color: #333 !important;
    color: #f0f0f0 !important;
}
.st-download-button {
    background-color: #333 !important;
}
.sentimen-box.positif {
    background-color: #1e3a2a !important;
    color: #8fdf9f !important;
    border: 2px solid #28a745 !important;
}
.sentimen-box.negatif {
    background-color: #3a1e1e !important;
    color: #f8a0a0 !important;
    border: 2px solid #dc3545 !important;
}
.sentimen-box.netral {
    background-color: #3a3a1e !important;
    color: #f0e68c !important;
    border: 2px solid #ffc107 !important;
}
.st-bb {
    background-color: #2d2d2d !important;
}
hr {
    border-color: #444 !important;
}
[data-testid="stSidebarNav"] {
    background-color: #252525 !important;
}
.st-bq {
    background-color: #333 !important;
}
.st-c0 {
    background-color: #333 !important;
}
</style>
"""

light_css = """
<style>
.sentimen-box.positif {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #28a745;
}
.sentimen-box.negatif {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #dc3545;
}
.sentimen-box.netral {
    background-color: #fff3cd;
    color: #856404;
    border: 2px solid #ffc107;
}
</style>
"""

if st.session_state.dark_mode:
    st.markdown(dark_css, unsafe_allow_html=True)
    plt.style.use('dark_background')
else:
    st.markdown(light_css, unsafe_allow_html=True)
    plt.style.use('default')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data['model'], data['vectorizer']

@st.cache_data
def load_preprocessed_dataset():
    if not os.path.exists(DATASET_PRE_PATH):
        return None
    return pd.read_csv(DATASET_PRE_PATH)

@st.cache_data
def load_labeled_dataset():
    if not os.path.exists(DATASET_LABELED_PATH):
        return None
    return pd.read_csv(DATASET_LABELED_PATH)

THRESHOLD = 0.70

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
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Klasifikasi Sentimen Ulasan Wisatawan")
st.markdown("**TF-IDF + Support Vector Machine (SVM)** — Destinasi Wisata di Pacitan, Jawa Timur")
st.divider()

page = st.sidebar.radio("Menu", [
    "Analisis Teks Manual",
    "Analisis File CSV & Dashboard",
    "Data Scraping",
    "Distribusi Dataset"
])

st.sidebar.divider()

dark_mode_toggle = st.sidebar.toggle("🌙 Mode Gelap", value=st.session_state.dark_mode)
if dark_mode_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode_toggle
    st.rerun()

st.sidebar.divider()

if 'df_result' in st.session_state and st.session_state.df_result is not None:
    st.sidebar.markdown("### 📊 Distribusi Hasil")
    df_side = st.session_state.df_result
    total = len(df_side)
    st.sidebar.metric("Total Review", total)
    sentimen_counts = df_side['Sentimen'].value_counts()
    for s in ['POSITIF', 'NEGATIF', 'NETRAL']:
        cnt = sentimen_counts.get(s, 0)
        pct = cnt / total * 100
        st.sidebar.markdown(f"- **{s}**: {cnt} ({pct:.1f}%)")
else:
    st.sidebar.info("Belum ada data diproses.")

st.sidebar.divider()

sentimen_color = {'POSITIF': 'positif', 'NEGATIF': 'negatif', 'NETRAL': 'netral'}

def plot_sentimen_pie(counts, title="Proporsi Sentimen", figsize=(4, 4)):
    colors = {'POSITIF': '#28a745', 'NEGATIF': '#dc3545', 'NETRAL': '#ffc107'}
    pie_colors = [colors.get(s, '#6c757d') for s in counts.index]
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct='%1.1f%%',
        startangle=90, colors=pie_colors,
        textprops={'fontsize': 9, 'weight': 'bold', 'color': TEXT_COLOR if st.session_state.dark_mode else '#333'}
    )
    for t in autotexts:
        t.set_color('white')
    ax.axis('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT_COLOR if st.session_state.dark_mode else '#333')
    return fig

def plot_wordcloud(texts, title="Word Cloud", figsize=(4, 3)):
    all_text = ' '.join(map(str, texts))
    if not all_text.strip():
        return None
    wc = WordCloud(
        width=800, height=400,
        background_color=BG_COLOR,
        colormap='viridis',
        max_words=100
    ).generate(all_text)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT_COLOR if st.session_state.dark_mode else '#333')
    return fig

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
                df_input['text_clean'] = texts
                st.session_state.df_result = df_input

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
                fig1 = plot_sentimen_pie(sentimen_counts, figsize=(6, 6))
                st.pyplot(fig1)

            with col2:
                st.subheader("☁️ Word Cloud")
                fig2 = plot_wordcloud(texts, figsize=(8, 4))
                if fig2:
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
        df_scraped['text_clean'] = texts
        st.session_state.df_result = df_scraped

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
        fig1 = plot_sentimen_pie(sentimen_counts, figsize=(6, 6))
        st.pyplot(fig1)

    with col2:
        st.subheader("☁️ Word Cloud")
        fig2 = plot_wordcloud(texts, figsize=(8, 4))
        if fig2:
            st.pyplot(fig2)
        else:
            st.info("Tidak ada teks untuk ditampilkan di Word Cloud.")

    st.divider()
    st.subheader("🏷️ Distribusi per Destinasi")
    pivot = df_scraped.groupby(['Nama Tempat', 'Sentimen']).size().unstack(fill_value=0)
    st.dataframe(pivot, use_container_width=True)

elif page == "Distribusi Dataset":
    st.header("📊 Distribusi Dataset")
    st.markdown("Ringkasan dataset ulasan wisata Pacitan hasil preprocessing dan prediksi sentimen.")

    df_pre = load_preprocessed_dataset()
    df_label = load_labeled_dataset()

    if df_pre is None:
        st.warning("Dataset preprocessing belum tersedia. Jalankan `python preprocess_dataset.py` terlebih dahulu.")
        st.stop()

    st.subheader("📋 Dataset Preprocessing")
    st.dataframe(df_pre.head(10), use_container_width=True)

    totals_col1, totals_col2, totals_col3 = st.columns(3)
    with totals_col1:
        st.metric("Total Review", len(df_pre))
    with totals_col2:
        tempat_count = df_pre['Nama Tempat'].nunique() if 'Nama Tempat' in df_pre.columns else '-'
        st.metric("Total Destinasi", tempat_count)
    with totals_col3:
        if 'Wisatawan Lokal' in df_pre.columns:
            lokal_count = df_pre['Wisatawan Lokal'].sum()
            st.metric("Wisatawan Lokal", f"{lokal_count} ({lokal_count/len(df_pre)*100:.1f}%)")

    st.divider()

    st.subheader("☁️ Word Cloud Seluruh Review")
    text_col = 'text_clean' if 'text_clean' in df_pre.columns else 'Review'
    fig_wc = plot_wordcloud(df_pre[text_col], title="Distribusi Kata pada Dataset", figsize=(10, 5))
    if fig_wc:
        st.pyplot(fig_wc)

    st.divider()
    st.subheader("🗂️ Distribusi per Destinasi")
    if 'Nama Tempat' in df_pre.columns:
        tempat_dist = df_pre['Nama Tempat'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(tempat_dist)))
        ax.barh(tempat_dist.index, tempat_dist.values, color=colors)
        ax.set_xlabel('Jumlah Review', color=TEXT_COLOR if st.session_state.dark_mode else '#333')
        ax.set_ylabel('Destinasi', color=TEXT_COLOR if st.session_state.dark_mode else '#333')
        ax.set_title('Distribusi Review per Destinasi', fontsize=13, fontweight='bold',
                      color=TEXT_COLOR if st.session_state.dark_mode else '#333')
        ax.tick_params(colors=TEXT_COLOR if st.session_state.dark_mode else '#333')
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR if st.session_state.dark_mode else '#333')
        st.pyplot(fig)

    st.divider()

    if df_label is not None:
        st.subheader("🎯 Hasil Prediksi Sentimen")
        st.markdown("Dataset telah diprediksi menggunakan model SVM dengan threshold **0.70**.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Distribusi Sentimen**")
            sentimen_counts = df_label['sentimen'].value_counts()
            fig_pie = plot_sentimen_pie(sentimen_counts, figsize=(6, 6))
            st.pyplot(fig_pie)

        with col2:
            st.markdown("**Word Cloud per Sentimen**")
            sentimen_tabs = st.tabs(['POSITIF', 'NEGATIF', 'NETRAL'])
            for tab, sent in zip(sentimen_tabs, ['POSITIF', 'NEGATIF', 'NETRAL']):
                with tab:
                    mask = df_label['sentimen'] == sent
                    if mask.any():
                        texts_sent = df_label.loc[mask, 'text_clean']
                        fig_wc_sent = plot_wordcloud(texts_sent, title=sent, figsize=(6, 3))
                        if fig_wc_sent:
                            st.pyplot(fig_wc_sent)
                    else:
                        st.info(f"Tidak ada data untuk sentimen {sent}.")

        st.divider()
        st.markdown("**Stat Lengkap Dataset**")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("POSITIF", f"{sentimen_counts.get('POSITIF', 0)} ({sentimen_counts.get('POSITIF', 0)/len(df_label)*100:.1f}%)")
        with col_stats2:
            st.metric("NEGATIF", f"{sentimen_counts.get('NEGATIF', 0)} ({sentimen_counts.get('NEGATIF', 0)/len(df_label)*100:.1f}%)")
        with col_stats3:
            st.metric("NETRAL", f"{sentimen_counts.get('NETRAL', 0)} ({sentimen_counts.get('NETRAL', 0)/len(df_label)*100:.1f}%)")

        st.subheader("📋 Data Hasil Prediksi")
        st.dataframe(df_label.head(20), use_container_width=True)

        csv_download = df_label.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi CSV",
            data=csv_download,
            file_name="GMaps_Review_Labeled.csv",
            mime="text/csv"
        )
    else:
        st.info("Dataset hasil prediksi belum tersedia. Jalankan `python label_sentimen.py` terlebih dahulu.")
