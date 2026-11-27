"""Run validation grid, then an explicitly requested final evaluation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from baselines import fit_baselines
from evaluate import confusion, curve, metrics
from prepare import SEEDS, SIZES, sample_train, split_tickets
from tickets_data import LABELS

CANDIDATES = {'head': [(0.001, 2), (0.0005, 3)],
              'full': [(0.00002, 2), (0.00005, 3)]}
REPORTS = Path('reports')


def tune(train, val):
    from finetune import train_model

    sample = sample_train(train, 100, SEEDS[0])
    trials, logs = [], []
    for method, candidates in CANDIDATES.items():
        for lr, epochs in candidates:
            model, _, history = train_model(sample, val, SEEDS[0], method, lr, epochs)
            trials.append(dict(method=method, n=100, seed=SEEDS[0], lr=lr,
                               epochs=epochs, val_accuracy=history[-1]['val_accuracy']))
            logs.extend(dict(method=method, n=100, seed=SEEDS[0], lr=lr,
                             planned_epochs=epochs, **entry) for entry in history)
            del model
    frame = pd.DataFrame(trials)
    frame.to_csv(REPORTS / 'tuning.csv', index=False)
    pd.DataFrame(logs).to_csv(REPORTS / 'tuning_epochs.csv', index=False)
    chosen = {}
    for name, group in frame.groupby('method'):
        row = group.loc[group.val_accuracy.idxmax()]
        chosen[name] = [float(row.lr), int(row.epochs)]
    (REPORTS / 'settings.json').write_text(json.dumps(chosen, indent=2))
    return chosen


def validate(sizes=SIZES, seeds=SEEDS, baseline_only=False, smoke=False):
    train, val, _ = split_tickets()
    rows, logs = [], []
    settings = {}
    if not baseline_only:
        from finetune import predict, train_model

        REPORTS.mkdir(exist_ok=True)
        settings = ({name: options[0] for name, options in CANDIDATES.items()}
                    if smoke else tune(train, val))
    for n in sizes:
        for seed in seeds:
            sample = sample_train(train, n, seed)
            for name, model in fit_baselines(sample, seed).items():
                score = metrics(val.label, model.predict(val.text))
                rows.append(dict(method=name, n=n, seed=seed, lr=0, epochs=0,
                                 val_accuracy=score['accuracy'], val_macro_f1=score['macro_f1']))
            if baseline_only:
                continue
            for name, (lr, epochs) in settings.items():
                epochs = 1 if smoke else epochs
                model, tok, history = train_model(sample, val, seed, name, lr, epochs)
                for entry in history:
                    logs.append(dict(method=name, n=n, seed=seed, lr=lr, **entry))
                score = metrics(val.label, predict(model, tok, val.text))
                rows.append(dict(method=name, n=n, seed=seed, lr=lr, epochs=epochs,
                                 val_accuracy=score['accuracy'], val_macro_f1=score['macro_f1']))
                del model
    folder = Path('scratch') if smoke or baseline_only else REPORTS
    folder.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(folder / ('baselines.csv' if baseline_only else 'results.csv'), index=False)
    if logs:
        pd.DataFrame(logs).to_csv(folder / 'epochs.csv', index=False)
    curve(results, folder / ('baseline_curve.png' if baseline_only else 'learning_curve.png'))
    summary = results.groupby(['method', 'n']).val_accuracy.agg(['mean', 'min', 'max'])
    summary.to_csv(folder / ('baseline_summary.csv' if baseline_only else 'summary.csv'))
    print(summary)


def final_check():
    from finetune import predict, train_model
    from share import save_model

    if (REPORTS / 'final.json').exists():
        raise SystemExit('Final scores already exist. Do not reuse test results for tuning.')
    own = pd.read_csv('data/own_messages.csv')
    if (list(own.columns) != ['text', 'label'] or len(own) != 40
            or own.text.isna().any() or own.text.str.strip().eq('').any()
            or own.text.duplicated().any()
            or own.label.value_counts().to_dict() != dict.fromkeys(LABELS, 10)):
        raise ValueError('Write 40 original unique messages, ten for each queue, first.')
    results = pd.read_csv(REPORTS / 'results.csv')
    if len(results) != 36 or set(results.method) != {'majority', 'tfidf', 'head', 'full'}:
        raise ValueError('Complete the full validation grid first.')
    n = max(SIZES)
    largest = results[results.n == n]
    means = largest.groupby('method').val_accuracy.mean()
    winner = str(means.idxmax())
    neural = str(means.loc[['head', 'full']].idxmax())
    seed = SEEDS[0]  # fixed in advance; not the best seed
    train, val, test = split_tickets()
    sample = sample_train(train, n, seed)
    settings = json.loads((REPORTS / 'settings.json').read_text())
    model, tok, history = train_model(sample, val, seed, neural, *settings[neural])
    pd.DataFrame(history).to_csv(REPORTS / 'final_epochs.csv', index=False)
    predictions = {}
    for name, baseline in fit_baselines(sample, seed).items():
        predictions[name] = [baseline.predict(test.text), baseline.predict(own.text)]
    predictions[neural] = [predict(model, tok, test.text), predict(model, tok, own.text)]
    final = dict(n=n, seed=seed, validation_winner=winner, published_variant=neural,
                 validation_means=means.to_dict(), settings=settings, scores={})
    for name, (p_test, p_own) in predictions.items():
        final['scores'][name] = {'test': metrics(test.label, p_test),
                                 'own': metrics(own.label, p_own)}
    confusion(test.label, predictions[winner][0], REPORTS / 'confusion_matrix.png')
    save_model(model, tok, final)
    (REPORTS / 'final.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'smoke', 'grid', 'final', 'all'], default='all')
    args = parser.parse_args()
    if args.mode in ('grid', 'all') and (REPORTS / 'final.json').exists():
        raise SystemExit('Final scores exist; do not tune after test evaluation.')
    if args.mode == 'baseline':
        validate(baseline_only=True)
    elif args.mode == 'smoke':
        validate(sizes=(16,), seeds=(42,), smoke=True)
    else:
        if args.mode in ('grid', 'all'):
            validate()
        if args.mode in ('final', 'all'):
            final_check()
