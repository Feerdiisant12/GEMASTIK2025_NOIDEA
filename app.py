# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from helpers import (
    create_risk_map, create_cluster_plot, generate_pdf_report,
    create_feature_importance_plot, create_cluster_comparison_plot
)
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Prediksi Krisis Air Indonesia", page_icon="💧", layout="wide")

@st.cache_resource
def load_all_resources():
    try:
        scaler = joblib.load('minmax_scaler.joblib'); pca = joblib.load('pca_model.joblib'); kmeans = joblib.load('kmeans_model.joblib'); classifier = joblib.load('risk_classifier_model.joblib')
        labeled_data = pd.read_csv('labeled_station_profiles.csv'); df_station = pd.read_csv('station_detail.csv'); df_province = pd.read_csv('province_detail.csv')
        df_full_climate = pd.read_csv('climate_data.csv')
        df_full_merged = pd.merge(pd.merge(df_full_climate, df_station, on='station_id'), df_province, on='province_id')
        df_full_merged['date'] = pd.to_datetime(df_full_merged['date'], format='%d-%m-%Y')
        return scaler, pca, kmeans, classifier, labeled_data, df_station, df_province, df_full_merged
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file: {e}."); return [None] * 8
scaler, pca, kmeans, classifier, labeled_data, df_station, df_province, df_full = load_all_resources()
if scaler is None: st.stop()

def predict_risk(input_df, scaler, pca, classifier):
    feature_order = ['max_consecutive_dry_days', 'max_consecutive_hot_days']; input_df_ordered = input_df[feature_order]
    input_scaled = scaler.transform(input_df_ordered); input_pca = pca.transform(input_scaled)
    prediction = classifier.predict(input_df_ordered); probability = classifier.predict_proba(input_df_ordered)
    return prediction, probability, input_pca

st.title("💧 Dashboard Prediksi Wilayah Rawan Krisis Air")

# --- SIDEBAR ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    try: genai.configure(api_key=gemini_api_key)
    except Exception as e: st.sidebar.error(f"Error konfigurasi Gemini: {e}")
else:
    st.sidebar.warning("GEMINI_API_KEY tidak ditemukan. Fitur insight AI tidak akan aktif.", icon="⚠️")
st.sidebar.header("Pilih Metode Input"); input_method = st.sidebar.radio("Metode Input", ["Input Manual", "Upload File CSV"], key="input_method_radio")
if 'run_prediction_clicked' not in st.session_state:
    st.session_state.run_prediction_clicked = False
input_df = None; selected_province_manual = None
if input_method == "Input Manual":
    st.sidebar.subheader("Masukkan Karakteristik Wilayah")
    provinsi_list = sorted(df_province['province_name'].unique())
    selected_province_manual = st.sidebar.selectbox("Pilih Provinsi", provinsi_list, index=provinsi_list.index("Nusa Tenggara Timur") if "Nusa Tenggara Timur" in provinsi_list else 0, key='manual_provinsi_select')
    consecutive_dry = st.sidebar.slider("Periode Kering Terpanjang (Hari)", 1, 200, 50)
    consecutive_hot = st.sidebar.slider("Periode Panas (>35°C) Terpanjang (Hari)", 0, 100, 10)
    input_data = {'max_consecutive_dry_days': consecutive_dry, 'max_consecutive_hot_days': consecutive_hot}
    input_df = pd.DataFrame([input_data])
else:
    st.sidebar.subheader("Upload File CSV Anda")
    uploaded_file = st.sidebar.file_uploader("Contoh: ...dry_days,...hot_days,provinsi", type="csv")
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            required_cols = {'max_consecutive_dry_days', 'max_consecutive_hot_days'}
            if not required_cols.issubset(input_df.columns):
                st.error(f"File CSV minimal harus memiliki kolom: {', '.join(required_cols)}"); input_df = None
            elif 'provinsi' in input_df.columns:
                st.session_state.csv_provinces = sorted(input_df['provinsi'].unique())
        except Exception as e:
            st.error(f"Gagal membaca file: {e}"); input_df = None
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Jalankan Prediksi", type="primary"):
    st.session_state.run_prediction_clicked = True

# --- AREA HASIL ---
if st.session_state.run_prediction_clicked and input_df is not None:
    is_batch = len(input_df) > 1 and input_method == 'Upload File CSV'
    with st.spinner("Menganalisis data dan menghasilkan laporan..."):
        if not is_batch:
            prediction, probability, input_pca = predict_risk(input_df, scaler, pca, classifier)
            result_text = "Rawan Krisis Air" if prediction[0] == 1 else "Tidak Rawan Krisis Air"
            risk_probability = probability[0][1]
            st.header("Hasil Analisis Prediksi")
            col1, col2 = st.columns(2)
            with col1: st.metric(label="**Tingkat Kerawanan**", value=result_text)
            with col2: st.metric(label="**Probabilitas Rawan Krisis**", value=f"{risk_probability*100:.2f}%")
            st.markdown("---")
            insights_text = "..." # Default
            if gemini_api_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-pro-latest')
                    
                    # Mengambil nilai input
                    input_dry_days = input_df['max_consecutive_dry_days'].iloc[0]
                    input_hot_days = input_df['max_consecutive_hot_days'].iloc[0]

                    # Konteks dan Persona
                    base_prompt = (
                        "Anda adalah seorang penasihat ahli di bidang kebijakan publik dan manajemen risiko bencana, dengan spesialisasi pada adaptasi perubahan iklim dan ketahanan air. "
                        f"Analisis data dari model machine learning yang andal telah menetapkan bahwa sebuah wilayah {selected_province_manual} dengan karakteristik berikut:\n"
                        f"- Periode kering terpanjang: {input_dry_days} hari\n"
                        f"- Periode panas terpanjang: {input_hot_days} hari\n"
                        f"Hasilnya, wilayah ini dikategorikan sebagai **{result_text}**."
                    )

                    # Instruksi Dinamis berdasarkan Hasil Prediksi
                    if result_text == "Tidak Rawan Krisis Air":
                        dynamic_instructions = (
                            f"\nMeskipun dikategorikan 'Tidak Rawan', probabilitas risiko tetap ada sebesar {risk_probability*100:.2f}%. "
                            f"Berikan insight strategis dalam format markdown berikut:\n\n"
                            f"###  Interpretasi Hasil\n"
                            f"Jelaskan secara singkat mengapa kombinasi {input_dry_days} hari kering dan {input_hot_days} hari panas menunjukkan resiliensi yang baik terhadap krisis air.\n\n"
                            f"### Potensi Risiko Tersembunyi\n"
                            f"Sebutkan 2-3 potensi risiko di masa depan yang harus tetap diwaspadai oleh wilayah ini (contoh: dampak jangka panjang perubahan iklim, peningkatan populasi, perubahan tata guna lahan).\n\n"
                            f"### Rekomendasi Proaktif untuk Menjaga Ketahanan\n"
                            f"Berikan 3 poin rekomendasi utama yang bersifat preventif untuk memastikan wilayah ini tetap aman dari krisis air di masa depan. Gunakan bahasa yang jelas, positif, dan dapat ditindaklanjuti."
                        )
                    else: # Jika Rawan Krisis Air
                        dynamic_instructions = (
                            f"\nProbabilitas kerawanan yang tinggi ({risk_probability*100:.2f}%) menandakan situasi yang memerlukan perhatian serius. "
                            f"Berikan insight darurat dalam format markdown berikut:\n\n"
                            f"### Interpretasi Tingkat Kerawanan\n"
                            f"Jelaskan secara singkat mengapa kombinasi {input_dry_days} hari kering dan {input_hot_days} hari panas menjadi pemicu utama status 'Rawan Krisis Air'.\n\n"
                            f"### Dampak Kaskade yang Perlu Diantisipasi\n"
                            f"Sebutkan 2-3 dampak turunan (kaskade) yang paling signifikan jika krisis air terjadi (contoh: gagal panen yang memicu inflasi, masalah sanitasi dan kesehatan, konflik sosial).\n\n"
                            f"### Rekomendasi Aksi Cepat (Prioritas Utama)\n"
                            f"Berikan 3 poin rekomendasi paling mendesak yang harus segera dilakukan untuk mitigasi dalam jangka pendek. Gunakan bahasa yang tegas, jelas, dan dapat ditindaklanjuti."
                        )


                    final_prompt = base_prompt + dynamic_instructions; response = model.generate_content(final_prompt); insights_text = response.text
                except Exception as e:
                    insights_text = f"Gagal menghasilkan insight dari Gemini. Error: {e}"
            st.subheader("💡 Insight dan Rekomendasi dari AI")
            st.markdown(insights_text, unsafe_allow_html=True)
            
            pdf_params = { "is_batch": False, "input_data": input_df.iloc[0].to_dict(), "result": result_text, "probability": risk_probability }
        else:
            st.header("Hasil Analisis Prediksi Batch")
            predictions, probabilities = [], []
            for _, row in input_df.iterrows():
                pred, prob, _ = predict_risk(pd.DataFrame([row]), scaler, pca, classifier); predictions.append("Rawan" if pred[0] == 1 else "Tidak Rawan"); probabilities.append(f"{prob[0][1]*100:.2f}%")
            results_df = input_df.copy(); results_df['Prediksi'] = predictions; results_df['Probabilitas Rawan'] = probabilities
            st.dataframe(results_df)
            
            # --- PERBAIKAN DI SINI: Logika Prompt Batch yang Lebih Cerdas ---
            insights_text = "Fitur Insight AI tidak aktif."
            if gemini_api_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-pro-latest')
                    summary_rawan = results_df['Prediksi'].value_counts().get('Rawan', 0)
                    total_data = len(results_df)
                    
                    # Logika prompt dinamis untuk batch
                    if summary_rawan > 0:
                        avg_dry_rawan = results_df[results_df['Prediksi'] == 'Rawan']['max_consecutive_dry_days'].mean()
                        prompt = (
                            f"Anda adalah seorang analis data senior. Anda baru saja menganalisis {total_data} wilayah.\n"
                            f"Hasilnya, ditemukan **{summary_rawan} wilayah** teridentifikasi **Rawan Krisis Air**. "
                            f"Rata-rata periode kering terpanjang untuk wilayah-wilayah rawan tersebut adalah **{avg_dry_rawan:.0f} hari**.\n\n"
                            f"Berikan **analisis ringkasan** dari temuan ini dan **tiga rekomendasi kebijakan strategis** yang paling berdampak berdasarkan data agregat ini. Fokus pada tindakan yang bisa diambil oleh pemerintah daerah atau lembaga terkait."
                        )
                    else:
                        prompt = (
                            f"Anda adalah seorang analis data senior. Anda baru saja menganalisis {total_data} wilayah, dan hasilnya sangat positif: **tidak ada satu pun** yang teridentifikasi sebagai **Rawan Krisis Air**.\n\n"
                            f"Berikan **analisis singkat** yang menjelaskan mengapa hasil ini menggembirakan. Kemudian, berikan **tiga rekomendasi kebijakan proaktif** untuk memastikan wilayah-wilayah ini dapat mempertahankan ketahanan air mereka di masa depan menghadapi tantangan perubahan iklim."
                        )
                    response = model.generate_content(prompt)
                    insights_text = response.text
                except Exception as e:
                    insights_text = f"Gagal menghasilkan insight: {e}"
            
            st.subheader("💡 Ringkasan Insight dari AI")
            st.markdown(insights_text, unsafe_allow_html=True)
            pdf_params = { "is_batch": True, "results_df": results_df }

        st.markdown("---")
        st.subheader("🔬 Analisis Data & Model Tambahan")
        comparison_plot_path = create_cluster_comparison_plot(labeled_data)
        feature_names = ['max_consecutive_dry_days', 'max_consecutive_hot_days']
        importance_plot_path = create_feature_importance_plot(classifier, feature_names)

        with st.expander("Lihat Detail Analisis", expanded=True):
            st.markdown("#### Tren Iklim Historis per Provinsi")
            if selected_province_manual:
                selected_province_for_trend = selected_province_manual
            elif 'csv_provinces' in st.session_state:
                selected_province_for_trend = st.selectbox("Pilih Provinsi dari file CSV Anda", st.session_state.csv_provinces, key='csv_trend_select')
            else:
                provinsi_options = sorted(df_full['province_name'].unique())
                selected_province_for_trend = st.selectbox("Pilih Provinsi", provinsi_options, key='default_trend_select')
            
            prov_data = df_full[df_full['province_name'] == selected_province_for_trend].copy()
            if prov_data.empty:
                st.warning(f"Tidak ada data historis yang tersedia untuk {selected_province_for_trend}.")
            else:
                prov_data.set_index('date', inplace=True)
                rolling_avg_temp = prov_data['Tavg'].rolling(window=30, min_periods=1).mean()
                rolling_sum_rain = prov_data['RR'].rolling(window=30, min_periods=1).sum()
                chart_data = pd.DataFrame({'Suhu Rata-rata (°C) (30-hari)': rolling_avg_temp, 'Total Hujan (mm) (30-hari)': rolling_sum_rain})
                st.line_chart(chart_data)
            
            st.markdown("#### Distribusi Karakteristik Klaster"); st.image(comparison_plot_path)
            st.markdown("#### Faktor Paling Berpengaruh dalam Model"); st.image(importance_plot_path)

        st.markdown("---")
        st.subheader("📊 Visualisasi Prediksi & Peta Risiko")
        risk_map_path = create_risk_map(labeled_data, df_station, df_province)
        if risk_map_path: st.image(risk_map_path)
        if not is_batch:
            _, _, input_pca = predict_risk(input_df, scaler, pca, classifier)
            features_pca = pca.transform(scaler.transform(labeled_data[feature_names]))
            cluster_map_path = create_cluster_plot(features_pca, input_pca, kmeans)
            st.image(cluster_map_path)
        else: cluster_map_path = None
        st.markdown("---")

        st.subheader("📄 Unduh Laporan Lengkap")
        pdf_params.update({"insights": insights_text, "map_path": risk_map_path, "cluster_path": cluster_map_path, "comparison_path": comparison_plot_path, "importance_path": importance_plot_path})
        pdf_path = generate_pdf_report(**pdf_params)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="Unduh Laporan PDF", data=pdf_file, file_name="Laporan_Prediksi_Krisis_Air.pdf", mime="application/octet-stream")
else:
    st.info("Silakan masukkan data di sidebar kiri dan klik 'Jalankan Prediksi' untuk memulai analisis.")