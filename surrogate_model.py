import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Данные ───────────────────────────────────────────────────────────────────
doe     = pd.read_csv("doe.csv")
results = pd.read_csv("results.csv")

X = doe.drop(columns=["points"])
targets = {
    "global_max":               "Глобальная макс. деформация",
    "glotis_max":               "Голосовая щель",
    "larynx_max":               "Гортань",
    "upper_trachea_bottom_max": "Трахея (нижняя)",
    "epiglotis_max":            "Надгортанник",
}

print("=" * 65)
print("СУРРОГАТНАЯ МОДЕЛЬ — Random Forest")
print("Оценка качества: R² (кросс-валидация 5-fold)")
print("R² = 1.0 → идеально   R² = 0.0 → модель бесполезна")
print("=" * 65)

models = {}
scores = {}

for col, label in targets.items():
    y = results[col]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    cv    = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s   = cross_val_score(model, X, y, cv=cv, scoring="r2")
    model.fit(X, y)
    models[col] = model
    scores[col] = r2s.mean()
    bar = "█" * int(max(r2s.mean(), 0) * 20)
    print(f"  {label:35s}  R²={r2s.mean():.3f}  {bar}")

print()
print("=" * 65)
print("ВАЖНОСТЬ ПРИЗНАКОВ — топ-8 для глобальной деформации")
print("=" * 65)
model_gmax = models["global_max"]
imp = pd.Series(model_gmax.feature_importances_, index=X.columns)
imp_sorted = imp.sort_values(ascending=False).head(8)
for feat, val in imp_sorted.items():
    bar = "█" * int(val * 200)
    print(f"  {feat:30s}  {val:.4f}  {bar}")

# ── Пример предсказания ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("ПРИМЕР: предсказание для нового пациента")
print("Меняем A_glotis с среднего (157) на минимальный (86)")
print("=" * 65)

mean_patient = X.mean().to_frame().T
small_glotis = mean_patient.copy()
small_glotis["A_glotis"] = X["A_glotis"].min()

for col, label in targets.items():
    pred_mean  = models[col].predict(mean_patient)[0]
    pred_small = models[col].predict(small_glotis)[0]
    diff = pred_small - pred_mean
    arrow = "▲" if diff > 0 else "▼"
    print(f"  {label:35s}  {pred_mean:.2f} → {pred_small:.2f} мм  {arrow}{abs(diff):.2f}")

# ── Сохраняем предсказания для всех 100 пациентов ────────────────────────────
preds = doe[["points"]].copy()
for col, label in targets.items():
    preds[col + "_predicted"] = models[col].predict(X).round(3)
    preds[col + "_actual"]    = results[col].values

preds.to_csv("surrogate_predictions.csv", index=False)
print()
print("Предсказания для всех 100 пациентов → surrogate_predictions.csv")