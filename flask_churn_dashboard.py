import os
import sys
import locale
import io
import codecs
import werkzeug.wrappers.response as wz_resp
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ==============================
# 🔧 UTF-8 / SURROGATE FIX PATCH
# ==============================

def ignore_surrogate_errors(text):
    """UTF-8 kodlama sırasında surrogate hatalarını yok sayar"""
    if isinstance(text, str):
        return codecs.encode(text, 'utf-8', 'surrogatepass').decode('utf-8', 'ignore')
    return text

# Flask Response objesini patch et
_old_set_data = wz_resp.Response.set_data

def _patched_set_data(self, value):
    if isinstance(value, str):
        value = ignore_surrogate_errors(value)
    return _old_set_data(self, value)

wz_resp.Response.set_data = _patched_set_data

# stdout / stderr akışlarını da UTF-8'e yönlendir
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
locale.setlocale(locale.LC_ALL, '')
sys.getdefaultencoding = lambda: 'utf-8'

# ==============================

from eda_utils import run_eda
from model_utils import train_model

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'churn_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_AS_ASCII'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.after_request
def set_charset(response):
    """Tarayıcıya UTF-8 header ekle"""
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


# Global değişkenler
df = None
uploaded_filename = None
target_column = None


@app.route('/')
def index():
    global df, uploaded_filename, target_column
    columns = df.columns.tolist() if df is not None else []
    return render_template(
        'index.html',
        filename=uploaded_filename,
        columns=columns,
        selected_target=target_column,
        result=None
    )


@app.route('/upload', methods=['POST'])
def upload():
    global df, uploaded_filename, target_column
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
            except:
                df = pd.read_csv(filepath, encoding='cp1252', errors='ignore')

        df.columns = df.columns.str.strip()
        uploaded_filename = file.filename
        target_column = None
        flash('Veri başarıyla yüklendi!', 'success')
    else:
        flash('Lütfen bir CSV dosyası yükleyin.', 'error')

    return redirect(url_for('index'))


@app.route('/set_target', methods=['POST'])
def set_target():
    global target_column
    target = request.form.get('target_column')
    if target:
        target_column = target
        flash(f'Hedef değişken olarak "{target}" seçildi.', 'success')
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


@app.route('/train', methods=['GET', 'POST'])
def train():
    global df, target_column, uploaded_filename

    if request.method == 'GET':
        flash("Model eğitimi buton üzerinden başlatılmalıdır.", "info")
        return redirect(url_for('index'))

    if df is None:
        flash("Önce veri yükleyin!", "error")
        return redirect(url_for('index'))

    if not target_column:
        flash("Hedef değişken seçilmedi!", "error")
        return redirect(url_for('index'))

    try:
        result = train_model(df, target_column)
        columns = df.columns.tolist()
        flash("Model başarıyla eğitildi!", "success")

        return render_template(
            'index.html',
            filename=uploaded_filename,
            columns=columns,
            selected_target=target_column,
            result=result
        )

    except Exception as e:
        import traceback
        error_msg = f"Hata oluştu: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        flash(f"Hata oluştu: {str(e)}", "error")
        return redirect(url_for('index'))


if __name__ == '__main__':
    print("🚀 Flask başlatılıyor... (UTF-8 + surrogate fix aktif)")
    app.run(debug=True, host='0.0.0.0', port=5000)
