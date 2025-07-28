# helpers.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from fpdf import FPDF
from datetime import datetime

# (Fungsi standardize_name, create_risk_map, create_cluster_plot, create_feature_importance_plot, create_cluster_comparison_plot, dan generate_pdf_report tetap sama persis seperti versi sebelumnya)
# Pastikan semua fungsi tersebut ada di sini.
# Fungsi create_trend_chart telah dihapus dari file ini.

def standardize_name(name):
    name = name.lower(); name = re.sub(r'\.', '', name); name = re.sub(r'\s+', ' ', name).strip(); return name
def create_risk_map(station_profile, df_station, df_province):
    province_risk = pd.merge(station_profile, df_station[['station_id', 'province_id']], on='station_id')
    province_risk = pd.merge(province_risk, df_province, on='province_id')
    province_risk_summary = province_risk.groupby('province_name')['rawan_krisis'].mean().reset_index()
    province_risk_summary.rename(columns={'rawan_krisis': 'tingkat_risiko'}, inplace=True)
    try: gdf = gpd.read_file("https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson")
    except Exception: return None
    special_mapping = {'nanggroe aceh darussalam': 'di aceh', 'kep bangka belitung': 'bangka belitung', 'di yogyakarta': 'daerah istimewa yogyakarta', 'dki jakarta': 'dki jakarta', 'kep riau': 'kepulauan riau', 'nusa tenggara barat': 'nusatenggara barat'}
    province_risk_summary['join_key'] = province_risk_summary['province_name'].apply(standardize_name).replace(special_mapping)
    gdf['join_key'] = gdf['Propinsi'].apply(standardize_name).replace(special_mapping)
    merged_gdf = gdf.merge(province_risk_summary, on='join_key', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Peta Tingkat Risiko Krisis Air Bersih per Provinsi', fontdict={'fontsize': '16', 'fontweight': '3'}); ax.axis('off')
    provinces_without_data = merged_gdf[merged_gdf['tingkat_risiko'].isna()]
    provinces_with_data = merged_gdf.dropna(subset=['tingkat_risiko'])
    if not provinces_without_data.empty: provinces_without_data.plot(color='lightgrey', linewidth=0.8, ax=ax, edgecolor='0.8')
    if not provinces_with_data.empty: provinces_with_data.plot(column='tingkat_risiko', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, legend_kwds={'label': "Tingkat Risiko", 'orientation': "horizontal", 'pad': 0.01})
    map_filename = "risk_map.png"; plt.savefig(map_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    return map_filename
def create_cluster_plot(features_pca, new_data_pca, kmeans):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=kmeans.labels_, palette='viridis', s=80, alpha=0.6, ax=ax, legend='full')
    if new_data_pca is not None:
        ax.scatter(new_data_pca[:, 0], new_data_pca[:, 1], marker='*', s=300, c='red', edgecolor='black', label='Input Anda')
    ax.set_title('Posisi Data dalam Klaster Iklim', fontsize=16)
    ax.set_xlabel('Principal Component 1'); ax.set_ylabel('Principal Component 2')
    ax.legend(title='Klaster'); ax.grid(True)
    cluster_plot_filename = "cluster_plot.png"; plt.savefig(cluster_plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    return cluster_plot_filename
def create_feature_importance_plot(model, feature_names):
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='rocket', ax=ax)
    ax.set_title('Faktor Paling Berpengaruh Menurut Model', fontsize=16)
    ax.set_xlabel('Tingkat Kepentingan'); ax.set_ylabel('Fitur')
    importance_plot_filename = "feature_importance.png"; plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    return importance_plot_filename
def create_cluster_comparison_plot(labeled_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = labeled_data.copy()
    plot_data['Status Risiko'] = plot_data['rawan_krisis'].apply(lambda x: 'Rawan' if x == 1 else 'Tidak Rawan')
    sns.violinplot(x='Status Risiko', y='max_consecutive_dry_days', data=plot_data, palette='coolwarm', ax=ax, inner='quartile')
    ax.set_title('Distribusi Periode Kering Terpanjang antar Klaster', fontsize=16)
    ax.set_xlabel('Status Risiko Hasil Clustering'); ax.set_ylabel('Maksimum Hari Kering Beruntun')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    comparison_plot_filename = "cluster_comparison.png"; plt.savefig(comparison_plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    return comparison_plot_filename
class PDF(FPDF):
    def header(self): self.set_font('Arial', 'B', 12); self.cell(0, 10, 'Laporan Analisis Prediksi Krisis Air', 0, 1, 'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')
def generate_pdf_report(is_batch, **kwargs):
    pdf = PDF(); pdf.add_page()
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Ringkasan Analisis', 0, 1)
    pdf.set_font('Arial', '', 11); pdf.cell(0, 8, f"Tanggal Laporan: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}", 0, 1); pdf.ln(5)
    if not is_batch:
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '1. Data Stasiun yang Dianalisis', 0, 1)
        pdf.set_font('Arial', '', 11)
        for key, value in kwargs['input_data'].items(): pdf.cell(0, 8, f"- {key.replace('_', ' ').title()}: {value:.2f}", 0, 1)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '2. Hasil Prediksi Risiko', 0, 1)
        pdf.set_font('Arial', 'B', 14)
        if kwargs['result'] == 'Rawan Krisis Air': pdf.set_text_color(220, 50, 50)
        else: pdf.set_text_color(50, 150, 50)
        pdf.cell(0, 10, kwargs['result'], 0, 1); pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 11); pdf.cell(0, 8, f"Probabilitas Rawan: {kwargs['probability']*100:.2f}%", 0, 1); pdf.ln(5)
    else:
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '1. Ringkasan Hasil Prediksi Batch', 0, 1)
        pdf.set_font('Arial', '', 10)
        results_df = kwargs['results_df']
        header = list(results_df.columns)
        col_widths = [40, 40, 30, 30, 30] if 'provinsi' in [h.lower() for h in header] else [45, 45, 45, 45]
        pdf.set_fill_color(200, 220, 255)
        for i, h in enumerate(header): pdf.cell(col_widths[i], 7, h, 1, 0, 'C', 1)
        pdf.ln()
        for _, row in results_df.iterrows():
            for i, item in enumerate(row): pdf.cell(col_widths[i], 6, str(item), 1)
            pdf.ln()
        pdf.ln(5)
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '2. Insight dan Rekomendasi (AI-Generated)', 0, 1)
    pdf.set_font('Arial', '', 11); pdf.multi_cell(0, 8, kwargs['insights']); pdf.ln(10)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '3. Visualisasi Analisis', 0, 1); pdf.ln(5)
    if not is_batch and kwargs.get('cluster_path'):
        pdf.set_font('Arial', 'B', 11); pdf.cell(0, 8, '3.1 Posisi Data dalam Klaster Iklim', 0, 1); pdf.image(kwargs['cluster_path'], x=None, y=None, w=180); pdf.ln(5)
    if kwargs.get('comparison_path'):
        pdf.set_font('Arial', 'B', 11); pdf.cell(0, 8, '3.2 Perbandingan Karakteristik Klaster', 0, 1); pdf.image(kwargs['comparison_path'], x=None, y=None, w=180); pdf.ln(5)
    if kwargs.get('importance_path'):
        pdf.set_font('Arial', 'B', 11); pdf.cell(0, 8, '3.3 Faktor Paling Berpengaruh', 0, 1); pdf.image(kwargs['importance_path'], x=None, y=None, w=180); pdf.ln(5)
    if kwargs.get('map_path'):
        pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '4. Peta Risiko Nasional', 0, 1); pdf.ln(5)
        pdf.image(kwargs['map_path'], x=None, y=None, w=180)
    report_filename = "Laporan_Prediksi_Krisis_Air.pdf"; pdf.output(report_filename); return report_filename