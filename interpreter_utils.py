import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from markupsafe import Markup
from typing import Any

# interpreter_utils.py
# ... diğer importlar
import matplotlib
matplotlib.use('Agg') 
# YENİ EKLENTİ: Force Plot'un düzgün çalışması için 'module://ipykernel.pylab.backend_inline' gibi
# tarayıcı tabanlı back-end'leri devre dışı bırakır.
plt.rcParams.update({'figure.max_open_warning': 0}) # Ekstra uyarıları kapatır

def fig_to_base64(fig):
    """Matplotlib figürünü Base64 stringe dönüştür"""
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def clean_text(text):
    """Bozuk Unicode karakterleri temizle"""
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def interpret_churn_risk(model: Any, X_test: pd.DataFrame, sample_id: int) -> Markup:
    
    # Kapsam hatasını gidermek için yerel import
    import shap 

    if sample_id >= len(X_test):
         return Markup(f"<p style='color:red;'>Seçilen müşteri indeksi ({sample_id}) mevcut değil. Maksimum indeks: {len(X_test) - 1}</p>")
         
    sample = X_test.iloc[[sample_id]]
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        sample_shap_values = explainer.shap_values(sample)

        try:
            churn_proba = model.predict_proba(sample)[:, 1][0]
        except AttributeError: 
             churn_proba = 0.5 
             
        base_value = explainer.expected_value
        if isinstance(base_value, list) and len(base_value) == 2:
             base_value = base_value[1] 
             sample_shap_values = sample_shap_values[1]

        html = f"<div style='padding:20px;'>"
        html += f"<h3 style='color:#dc3545;'>🔥 Müşteri Terk Riski Yorumlaması (Örnek ID: {sample_id})</h3>"
        
        risk_color = '#dc3545' if churn_proba > 0.5 else '#198754'
        html += f"<p><strong>Tahmini Terk Olasılığı:</strong> <span style='color:{risk_color}; font-size:1.2em;'>%{churn_proba * 100:.2f}</span></p>"

       # interpreter_utils.py içindeki interpret_churn_risk fonksiyonu
# ...

        # 5. SHAP Force Plot (Statik PNG Çözümü)
        
        # Matplotlib'i zorla
        shap.force_plot(
             base_value, 
             sample_shap_values[0], 
             sample.iloc[0], 
             show=False, 
             matplotlib=True 
        )
        
        # OLUŞAN FIGURE'Ü YAKALA VE BASE64'E ÇEVİR
        # plt.gcf() ile mevcut figürü yakalama.
        try:
             fig = plt.gcf()
        except:
             # Eğer plt.gcf() başarısız olursa, yeni bir figür açıp onu boş bırakıyoruz
             # ve devam ediyoruz. Bu, sadece hata yakalama amaçlıdır.
             fig = plt.figure() 

        buf = BytesIO()
        plt.tight_layout()
        
        # Eğer figür boşsa kaydetme hatası verir, bu yüzden try/except ekleyelim.
        try:
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig) 
            buf.seek(0)
            img_base64_force = base64.b64encode(buf.read()).decode('ascii')
            
            html += f"<h4>Tahmin Mekanizması (Kırmızı Terk Riskini Artırır)</h4>"
            html += f"<img src='data:image/png;base64,{img_base64_force}' class='img-fluid mt-2 mb-4' style='max-width:100%; border-radius:8px; background:white; padding:10px;'/>"
            
        except Exception as save_e:
            html += f"<h4>Tahmin Mekanizması (Görselleştirme Hatası)</h4>"
            html += f"<p style='color:red;'>Görsel oluşturulamadı. (Hata: {str(save_e)})</p>"
        
        # ... (Geri kalan Bar Plot kısmı aynı kalır)
        
        # Oluşan figure objesini yakala ve Base64'e çevir
        fig = plt.gcf()
        
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig) 
        buf.seek(0)
        img_base64_force = base64.b64encode(buf.read()).decode('ascii')
        
        html += f"<h4>Tahmin Mekanizması (Kırmızı Terk Riskini Artırır)</h4>"
        # Statik görseli HTML'e Base64 olarak ekle (arka planı beyaz yapıldı)
        html += f"<img src='data:image/png;base64,{img_base64_force}' class='img-fluid mt-2 mb-4' style='max-width:100%; border-radius:8px; background:white; padding:10px;'/>"
        
        
        # 6. SHAP Bar Plot (Genel Özellik Önem Sıralaması)
        if isinstance(shap_values, list):
             mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
             mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importance.plot(kind='barh', color='#ffc107', ax=ax)
        ax.set_title("Genel Özellik Önem Sıralaması (Ortalama Mutlak SHAP Değeri)")
        ax.set_xlabel("Ortalama |SHAP Değeri|")
        ax.set_ylabel("Özellik")
        img_base64_bar = fig_to_base64(fig)
        
        html += f"<hr><h4>Modelin Terk Kararındaki En Önemli Özellikler (Genel)</h4>"
        html += f"<img src='data:image/png;base64,{img_base64_bar}' class='img-fluid mt-2 mb-4' style='max-width:100%; border-radius:8px;'/>"
        
        html += "</div>"
        return Markup(clean_text(html))

    except Exception as e:
        import traceback
        err = clean_text(str(e))
        tb = clean_text(traceback.format_exc())
        return Markup(f"<p style='color:red;'>Yorumlama Modülünde Hata oluştu: {err}</p><pre>{tb}</pre>")