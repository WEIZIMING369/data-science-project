"""Reproduce the Iris exploratory analysis and paired feature comparison."""
from pathlib import Path
import hashlib
import json
import platform
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(ROOT / 'iris.data', names=COLS)
assert df.shape == (150, 5) and not df.isna().any().any()
assert df.species.value_counts().to_dict() == {k: 50 for k in df.species.unique()}
for folder in ['figures', 'results']:
    (ROOT / folder).mkdir(exist_ok=True)
colors = ['#176b87', '#c56635', '#7553a6']
markers = ['o', 's', '^']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

for group in ['petal', 'sepal']:
    fig, ax = plt.subplots(figsize=(8, 5.2), layout='constrained')
    for (species, subset), color, marker in zip(df.groupby('species', sort=True), colors, markers):
        ax.scatter(subset[f'{group}_length'], subset[f'{group}_width'],
                   label=species.replace('Iris-', ''), color=color, marker=marker,
                   alpha=.8, s=45, edgecolors='white', linewidths=.4)
    ax.set(xlabel=f'{group.title()} length (cm)', ylabel=f'{group.title()} width (cm)',
           title=f'Iris species in {group} measurements')
    ax.legend(title='Species', frameon=False)
    ax.text(0, -.19, 'Source: UCI Iris (Fisher, 1936), CC BY 4.0. 150 observations; 50 per species.',
            transform=ax.transAxes, fontsize=9)
    fig.savefig(ROOT / f'{group}_scatter.png', dpi=180)
    plt.close(fig)

# Identical stratified splits permit a paired descriptive comparison.
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, df.species))
scores = {}
for group in ['sepal', 'petal']:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    values = cross_val_score(model, df[[f'{group}_length', f'{group}_width']], df.species, cv=cv)
    scores[group] = {'fold_accuracy': values.tolist(), 'mean_accuracy': float(values.mean()),
                     'fold_sd': float(values.std(ddof=1))}
df.groupby('species').mean(numeric_only=True).round(3).to_csv(ROOT / 'species_means.csv')
summary = {'rows': len(df), 'missing_cells': int(df.isna().sum().sum()),
           'duplicate_rows': int(df.duplicated().sum()),
           'duplicate_policy': 'Retained: identical rounded measurements may represent distinct flowers.',
           'data_sha256': hashlib.sha256((ROOT / 'iris.data').read_bytes()).hexdigest(),
           'evaluation': '5-fold stratified shuffled CV; seed 42; scaling fitted within each training fold',
           'scores': scores,
           'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                        'matplotlib': matplotlib.__version__, 'scikit-learn': sklearn.__version__}}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))

