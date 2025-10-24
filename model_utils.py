import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from markupsafe import Markup

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

def train_model(df: pd.DataFrame, target_column: str) -> Markup:
    if target_column not in df.columns:
        return Markup("<p style='color:red;'>Seçilen hedef sütun veride bulunamadı.</p>")

    if df[target_column].nunique() < 2:
        return Markup("<p style='color:red;'>Hedef değişkende yalnızca tek bir sınıf var! Eğitim yapılamaz.</p>")

    try:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(clean_text)

        y = df[target_column].copy()

        if y.dtype == 'O':
            y = y.astype(str).str.strip().str.lower()
            mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
            y = y.map(mapping)

        if y.isnull().any():
            return Markup("<p style='color:red;'>Hedef sütun dönüştürülemedi veya eksik değer içeriyor.</p>")

        if y.nunique() < 2:
            return Markup("<p style='color:red;'>Hedef değişkende yalnızca tek bir sınıf var!</p>")

        class_counts = y.value_counts()
        if any(class_counts < 2):
            return Markup("<p style='color:red;'>Hedef değişkende bazı sınıflarda 2'den az örnek var. Eğitim yapılamaz.</p>")

        X = df.drop(columns=[target_column])
        X = X.select_dtypes(include=['number', 'object'])
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # MODELLER
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM (Support Vector Machine)": SVC(kernel='rbf', probability=True, random_state=42)
        }

        results = []
        model_details = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-Score": round(f1, 3)
            })

            # Detaylı rapor
            report = classification_report(y_test, y_pred, digits=3)
            report = clean_text(report)

            # Confusion matrix çiz
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
        html += "<h3 style='color:#ffc107;'>Model Performans Karşılaştırması</h3>"

        html += "<table class='table table-dark table-striped table-bordered mt-3'>"
        html += "<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead><tbody>"
        for r in results:
            html += f"<tr><td>{r['Model']}</td><td>{r['Accuracy']}</td><td>{r['Precision']}</td><td>{r['Recall']}</td><td>{r['F1-Score']}</td></tr>"
        html += "</tbody></table>"

        # Her modelin detay raporu ve CM
        for det in model_details:
            html += f"<hr><h4 style='color:#0dcaf0;'>{det['name']} Detaylı Rapor</h4>"
            html += f"<pre style='background:#1e1e1e; color:#d4d4d4; padding:15px; border-radius:5px;'>{det['report']}</pre>"
            html += f"<img src='data:image/png;base64,{det['img']}' class='img-fluid mt-2 mb-4' style='max-width:100%; border-radius:8px;'/>"

        html += "</div>"

        return Markup(clean_text(html))

    except Exception as e:
        import traceback
        err = clean_text(str(e))
        tb = clean_text(traceback.format_exc())
        return Markup(f"<p style='color:red;'>Hata oluştu: {err}</p><pre>{tb}</pre>")
