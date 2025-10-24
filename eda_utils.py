import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64


def clean_text(text):
    """Bozuk Unicode karakterleri temizle"""
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text


def run_eda(df):
    # DataFrame'i temizle
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_text)
    
    html = "<div style='padding:20px;'>"

    # Genel bilgiler
    html += f"<h2>Veri Ozeti</h2>"
    html += f"<p><strong>Gozlem Sayisi:</strong> {df.shape[0]}</p>"
    html += f"<p><strong>Ozellik Sayisi:</strong> {df.shape[1]}</p>"

    # Eksik veriler
    nulls = df.isnull().sum()
    null_html = nulls[nulls > 0].to_frame(name="Eksik Deger Sayisi").to_html(classes='table table-bordered table-sm text-light')
    null_html = clean_text(null_html)
    
    if nulls.sum() > 0:
        html += f"<h3>Eksik Veriler</h3>{null_html}"
    else:
        html += f"<p><strong>Eksik veri bulunmamaktadir.</strong></p>"

    # Sayısal istatistikler
    html += f"<h3>Sayisal Degisken Istatistikleri</h3>"
    desc = df.describe().T
    desc_html = desc.to_html(classes='table table-striped table-sm text-light')
    desc_html = clean_text(desc_html)
    html += desc_html

    # Kategorik değişkenlerin dağılımı (ilk 3)
    cat_cols = df.select_dtypes(include='object').columns[:3]
    for col in cat_cols:
        col_clean = clean_text(str(col))
        html += f"<h4>{col_clean} Deger Dagilimi</h4>"
        val_counts = df[col].value_counts().to_frame().reset_index()
        val_counts.columns = [col_clean, 'Frekans']
        val_html = val_counts.to_html(classes='table table-hover table-sm text-light', index=False)
        val_html = clean_text(val_html)
        html += val_html

    # Korelasyon Isı Haritası
    try:
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        html += f"<h3>Korelasyon Isi Haritasi</h3><img src='data:image/png;base64,{img_base64}' class='img-fluid'/>"
    except:
        html += "<p><strong>Korelasyon hesaplanamadi.</strong></p>"

    html += "</div>"
    html = clean_text(html)
    return html