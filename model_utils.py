import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# YENİ İTHALATLAR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
# --------------------
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from markupsafe import Markup
from typing import Tuple, Any, Dict # Tip ipuçları için kritik import

def clean_text(text):
    """Bozuk Unicode karakterleri temizle"""
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def fig_to_base64(fig):
    """Matplotlib figürünü Base64 stringe dönüştür"""
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

# FONKSİYON İMZASI: (HTML, En İyi Model, X_test) döndürülüyor
def train_model(df: pd.DataFrame, target_column: str) -> Tuple[Any, Any, pd.DataFrame]:
    
    best_model = None
    X_test_result = pd.DataFrame() 
    
    if target_column not in df.columns:
        return Markup("<p style='color:red;'>Seçilen hedef sütun veride bulunamadı.</p>"), None, X_test_result

    if df[target_column].nunique() < 2:
        return Markup("<p style='color:red;'>Hedef değişkende yalnızca tek bir sınıf var! Eğitim yapılamaz.</p>"), None, X_test_result

    try:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(clean_text)

        y = df[target_column].copy()

        if y.dtype == 'O':
            y = y.astype(str).str.strip().str.lower()
            mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 'churn': 1, 'no churn': 0}
            y = y.map(mapping)
        
        if y.isnull().any() or y.nunique() < 2 or any(y.value_counts() < 2):
             return Markup("<p style='color:red;'>Hedef sütun dönüştürülemedi, eksik değer içeriyor veya yeterli sınıf örneği yok.</p>"), None, X_test_result


        X = df.drop(columns=[target_column])
        X = X.select_dtypes(include=['number', 'object'])
        
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        if not numerical_cols.empty:
            imputer = SimpleImputer(strategy='mean')
            X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
        X = X.fillna(0)
        
        # XGBoost uyumluluğu için sütun başlıklarını temizleme
        X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
        
        X_train, X_test_result, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        models: Dict[str, Any] = {
            # max_iter 2000'e çıkarıldı
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42), 
            "Random Forest (Dengeli)": RandomForestClassifier(
                random_state=42, 
                n_estimators=150, 
                class_weight='balanced'
            ),
            "XGBoost Classifier": xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42, 
                n_estimators=100,
                scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
            )
        }

        results: Dict[str, Dict[str, Any]] = {}
        model_details = []
        best_f1 = -1 

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_result)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            results[name] = {
                "Model": name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-Score": round(f1, 3)
            }

            report = classification_report(y_test, y_pred, digits=3)
            report = clean_text(report)

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek Değer")
            img_base64 = fig_to_base64(fig)

            model_details.append({
                "name": name,
                "report": report,
                "img": img_base64
            })

        # --- HTML OLUŞTUR ---
        html = "<div style='padding:20px;'>"
        html += "<h3 style='color:#ffc107;'>Model Performans Karşılaştırması (Geliştirilmiş)</h3>"
        
        best_model_name = "Bilinmiyor"
        if best_model and best_f1 > -1:
            # En iyi model adını bul
            for name, res in results.items():
                if res['F1-Score'] == best_f1:
                    best_model_name = name
                    break
            html += f"<p class='text-success'>**💡 En Yüksek F1-Skoru ({best_f1:.3f}) ile En İyi Model: {best_model_name}**</p>"
        
        html += "<table class='table table-dark table-striped table-bordered mt-3'>"
        html += "<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead><tbody>"
        for r in results.values():
            html += f"<tr><td>{r['Model']}</td><td>{r['Accuracy']}</td><td>{r['Precision']}</td><td>{r['Recall']}</td><td>{r['F1-Score']}</td></tr>"
        html += "</tbody></table>"

        for det in model_details:
            html += f"<hr><h4 style='color:#0dcaf0;'>{det['name']} Detaylı Rapor</h4>"
            html += f"<pre style='background:#1e1e1e; color:#d4d4d4; padding:15px; border-radius:5px;'>{det['report']}</pre>"
            html += f"<img src='data:image/png;base64,{det['img']}' class='img-fluid mt-2 mb-4' style='max-width:100%; border-radius:8px;'/>"

        html += "</div>"
        
        return Markup(clean_text(html)), best_model, X_test_result

    except Exception as e:
        import traceback
        err = clean_text(str(e))
        tb = clean_text(traceback.format_exc())
        return Markup(f"<p style='color:red;'>Hata oluştu: {err}</p><pre>{tb}</pre>"), None, X_test_result