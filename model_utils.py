import pandas as pd
from typing import Dict, Any
from io import BytesIO
import base64

import matplotlib.pyplot as plt
import seaborn as sns

from markupsafe import Markup

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------

def clean_text(text):
    if isinstance(text, str):
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --------------------------------------------------
# ANA MODEL EĞİTİM FONKSİYONU
# --------------------------------------------------

def train_model(df: pd.DataFrame, target_column: str) -> Markup:

    # ---------- KONTROLLER ----------
    if target_column not in df.columns:
        return Markup("<p style='color:red;'>Hedef sütun veride bulunamadı.</p>")

    if df[target_column].nunique() < 2:
        return Markup("<p style='color:red;'>Hedef değişkende yalnızca tek sınıf var.</p>")

    try:
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(clean_text)

        # ---------- HEDEF ----------
        y = df[target_column].astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "churn": 1, "no churn": 0
        }
        y = y.map(mapping)

        if y.isnull().any() or y.nunique() < 2:
            return Markup("<p style='color:red;'>Hedef değişken dönüştürülemedi.</p>")

        # ---------- ÖZELLİKLER ----------
        X = df.drop(columns=[target_column])
        X = X.select_dtypes(include=["number", "object"])

        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(include="object").columns

        if len(num_cols) > 0:
            X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
            X[num_cols] = StandardScaler().fit_transform(X[num_cols])

        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        X.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in X.columns]

        # ---------- TRAIN / TEST ----------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

        # ---------- MODELLER ----------
        models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=42
            ),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42)
        }

        results = {}
        model_details = []

        best_model = None
        best_model_name = ""
        best_f1 = -1

        # ---------- MODEL EĞİTİM + CV ----------
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

            results[name] = {
                "Model": name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-Score": round(f1, 3),
                "CV F1 Mean": round(cv_scores.mean(), 3),
                "CV F1 Std": round(cv_scores.std(), 3)
            }

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

            report = classification_report(y_test, y_pred, digits=3)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")

            model_details.append({
                "name": name,
                "report": report,
                "img": fig_to_base64(fig)
            })

        # ---------- GRIDSEARCH (SADECE RF) ----------
        if best_model_name == "Random Forest":
            rf_param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

            grid = GridSearchCV(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                rf_param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            tuned_model = grid.best_estimator_

            y_pred = tuned_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            cv_scores = cross_val_score(
                tuned_model, X_train, y_train, cv=5, scoring="f1"
            )

            results["Random Forest (Tuned)"] = {
                "Model": "Random Forest (Tuned)",
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-Score": round(f1, 3),
                "CV F1 Mean": round(cv_scores.mean(), 3),
                "CV F1 Std": round(cv_scores.std(), 3)
            }

        # ---------- HTML ----------
        html = "<div style='padding:20px;'>"
        html += "<h3 style='color:#ffc107;'>Model Performans Karşılaştırması</h3>"
        html += "<table class='table table-dark table-striped table-bordered'>"
        html += (
            "<thead><tr>"
            "<th>Model</th><th>Accuracy</th><th>Precision</th>"
            "<th>Recall</th><th>F1</th><th>CV F1 Mean</th><th>CV Std</th>"
            "</tr></thead><tbody>"
        )

        for r in results.values():
            html += (
                f"<tr><td>{r['Model']}</td>"
                f"<td>{r['Accuracy']}</td>"
                f"<td>{r['Precision']}</td>"
                f"<td>{r['Recall']}</td>"
                f"<td>{r['F1-Score']}</td>"
                f"<td>{r['CV F1 Mean']}</td>"
                f"<td>{r['CV F1 Std']}</td></tr>"
            )

        html += "</tbody></table>"

        for det in model_details:
            html += f"<hr><h4>{det['name']} Detaylı Rapor</h4>"
            html += f"<pre>{det['report']}</pre>"
            html += f"<img src='data:image/png;base64,{det['img']}' style='max-width:100%;'/>"

        html += "</div>"

        return Markup(clean_text(html))

    except Exception as e:
        import traceback
        return Markup(f"<pre>{traceback.format_exc()}</pre>")
