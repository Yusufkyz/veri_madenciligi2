import sys
import locale
import io
import werkzeug.wrappers.response as wz_resp
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ==============================
# 🔧 UTF-8 / SURROGATE FIX PATCH (Basitleştirilmiş Hata Çözümü)
# ==============================

def ignore_surrogate_errors(text):
    """UTF-8 kodlama sırasında surrogate hatalarını yok sayar (TypeError çözümü)"""
    if isinstance(text, str):
        # Yalnızca encode ve decode metotları kullanılıyor
        return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')
    return text

# Flask Response objesini patch et
_old_set_data = wz_resp.Response.set_data

def _patched_set_data(self, value):
    if isinstance(value, str):
        value = ignore_surrogate_errors(value)
    return _old_set_data(self, value)

try:
    wz_resp.Response.set_data = _patched_set_data
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
    locale.setlocale(locale.LC_ALL, '')
    sys.getdefaultencoding = lambda: 'utf-8'
except:
    pass 

# ==============================

from eda_utils import run_eda
from model_utils import train_model
from interpreter_utils import interpret_churn_risk 

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# KRİTİK: Flask uygulamasını başlatma ve yapılandırma!
app = Flask(__name__)
app.secret_key = 'churn_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_AS_ASCII'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global değişkenler
df = None
uploaded_filename = None
target_column = None
best_model = None         
X_test_global = None      
y_test_global = None      


@app.after_request
def set_charset(response):
    """Tarayıcıya UTF-8 header ekle"""
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


@app.route('/')
def index():
    global df, uploaded_filename, target_column, best_model
    columns = df.columns.tolist() if df is not None else []
    
    model_trained = best_model is not None 
    
    return render_template(
        'index.html',
        filename=uploaded_filename,
        columns=columns,
        selected_target=target_column,
        result=None,
        model_trained=model_trained 
    )


@app.route('/upload', methods=['POST'])
def upload():
    global df, uploaded_filename, target_column, best_model, X_test_global, y_test_global
    file = request.files.get('file')

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Düzeltme: 'errors' argümanı kaldırıldı
            df = pd.read_csv(filepath, encoding='utf-8')
        except:
            try:
                # Düzeltme: 'errors' argümanı kaldırıldı
                df = pd.read_csv(filepath, encoding='latin-1')
            except:
                # Düzeltme: 'errors' argümanı kaldırıldı
                try:
                    df = pd.read_csv(filepath, encoding='cp1252')
                except Exception as e:
                    flash(f"Dosya okuma başarısız oldu: {str(e)}", "error")
                    return redirect(url_for('index'))

        df.columns = df.columns.str.strip()
        uploaded_filename = file.filename
        target_column = None
        best_model = None
        X_test_global = None
        y_test_global = None
        flash('Veri başarıyla yüklendi! Lütfen hedef değişkeni seçin.', 'success')
    else:
        flash('Lütfen bir CSV dosyası yükleyin.', 'error')

    return redirect(url_for('index'))


@app.route('/set_target', methods=['POST'])
def set_target():
    global target_column
    target = request.form.get('target_column')
    if target:
        target_column = target
        flash(f'Hedef değişken olarak "{target}" seçildi. Artık modeli eğitebilirsiniz.', 'success')
    else:
        flash('Hedef değişken seçimi başarısız oldu.', 'error')
    return redirect(url_for('index'))


@app.route('/eda')
def eda():
    global df
    if df is not None:
        report_html = run_eda(df)
        return render_template('eda.html', report_html=report_html)
    else:
        flash('Önce veri yükleyin!', 'error')
        return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def train():
    global df, target_column, uploaded_filename, best_model, X_test_global, y_test_global

    if df is None or not target_column:
        flash("Önce veri yükleyin ve hedef değişkeni seçin!", "error")
        return redirect(url_for('index'))

    try:
        # train_model 3 değer döndürüyor: html, model, X_test
        result_html, model, X_test = train_model(df, target_column)

        if model is None:
             flash("Model eğitiminde kritik bir hata oluştu. Detaylar için raporu kontrol edin.", "error")
             model_trained = False
        else:
             # BAŞARILI DURUM: Global değişkenleri güncelle
             best_model = model
             X_test_global = X_test
             flash("Model başarıyla eğitildi! Artık tekil müşteri terk riskini yorumlayabilirsiniz.", "success")
             model_trained = True


        columns = df.columns.tolist()
        return render_template(
            'index.html',
            filename=uploaded_filename,
            columns=columns,
            selected_target=target_column,
            result=result_html,
            model_trained=model_trained 
        )

    except Exception as e:
        import traceback
        error_msg = f"Hata oluştu: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        flash(f"Model eğitiminde KRİTİK bir hata oluştu: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/interpret', methods=['GET'])
def interpret():
    global best_model, X_test_global
    
    if best_model is None or X_test_global is None:
        flash("Önce modeli eğitin!", "error")
        return redirect(url_for('index'))

    import numpy as np
    
    # Veri setinin boş olmaması için kontrol
    if X_test_global.empty:
        flash("Yorumlama için uygun test verisi bulunamadı.", "error")
        return redirect(url_for('index'))
        
    # Rastgele bir müşteri seçelim.
    sample_id = np.random.randint(0, len(X_test_global) - 1)
    
    # Yorumlama Modülünü Çalıştır
    interpretation_html = interpret_churn_risk(best_model, X_test_global, sample_id)
    
    return render_template(
        'interpret.html',
        interpretation_html=interpretation_html,
        sample_id=sample_id
    )


if __name__ == '__main__':
    print("🚀 Flask başlatılıyor... (UTF-8 + surrogate fix aktif)")
    app.run(debug=True, host='0.0.0.0', port=5000)