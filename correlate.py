import pandas as pd
import numpy as np

# Загружаем данные
doe = pd.read_csv("doe.csv")
results = pd.read_csv("results.csv")

# Убираем столбец points из doe, оставляем только параметры
params = doe.drop(columns=["points"])

# Берём ключевые метрики деформации из results
metrics = results[["global_max", "global_mean", "glotis_max", 
                    "larynx_max", "upper_trachea_bottom_max",
                    "gl_max", "gr_max", "epiglotis_max"]]

# Считаем корреляцию Пирсона каждого параметра с каждой метрикой
corr = pd.DataFrame(index=params.columns, columns=metrics.columns, dtype=float)

for param in params.columns:
    for metric in metrics.columns:
        corr.loc[param, metric] = params[param].corr(metrics[metric])

# Сортируем по влиянию на global_max
corr_sorted = corr.reindex(corr["global_max"].abs().sort_values(ascending=False).index)

print("=" * 70)
print("КОРРЕЛЯЦИЯ параметров анатомии с деформацией (Пирсон, -1 до +1)")
print("=" * 70)
print(corr_sorted.round(3).to_string())

print()
print("=" * 70)
print("ТОП-5 параметров влияющих на ГЛОБАЛЬНУЮ деформацию:")
print("=" * 70)
top5 = corr["global_max"].abs().sort_values(ascending=False).head(5)
for param, val in top5.items():
    direction = "↑ больше параметр → больше деформация" if corr.loc[param, "global_max"] > 0 else "↑ больше параметр → меньше деформация"
    print(f"  {param:30s}  r={corr.loc[param,'global_max']:+.3f}  {direction}")

print()
print("=" * 70)
print("ТОП-5 параметров влияющих на ГОЛОСОВУЮ ЩЕЛЬ (glotis_max):")
print("=" * 70)
top5g = corr["glotis_max"].abs().sort_values(ascending=False).head(5)
for param, val in top5g.items():
    direction = "↑" if corr.loc[param, "glotis_max"] > 0 else "↓"
    print(f"  {param:30s}  r={corr.loc[param,'glotis_max']:+.3f}  {direction}")

# Сохраняем полную таблицу
corr_sorted.round(3).to_csv("correlation_results.csv")
print()
print("Полная таблица сохранена → correlation_results.csv")