"""Metrics and committed report figures."""
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from tickets_data import LABELS

matplotlib.use('Agg')


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0))}


def curve(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in results.groupby('method'):
        stats = rows.groupby('n').val_accuracy.agg(['mean', 'min', 'max'])
        x = stats.index.to_numpy()
        ax.plot(x, stats['mean'].to_numpy(), marker='o', label=name)
        ax.fill_between(x, stats['min'].to_numpy(), stats['max'].to_numpy(), alpha=0.15)
    ax.set(xlabel='Training examples', ylabel='Validation accuracy', ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion(y, pred, path):
    disp = ConfusionMatrixDisplay.from_predictions(y, pred, labels=LABELS, xticks_rotation=30)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=160)
    plt.close(disp.figure_)
