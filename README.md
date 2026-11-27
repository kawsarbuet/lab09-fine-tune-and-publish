# Lakeshore Ticket Sorter: fine-tuning against a baseline

Model: TODO (replace after publishing)
Space: TODO (replace after publishing)

## What it does
Classifies synthetic bike-share messages into billing, bike_fault, station and account. Compares majority, TF-IDF logistic regression, frozen DistilBERT head and full DistilBERT.

## How to run it
Run from the repository root using Anaconda Prompt or a terminal with Conda enabled.

```bash
conda env create -f environment.yml
conda activate lab09-fine-tune-and-publish
python -c "import torch, transformers, gradio; print(torch.__version__, torch.cuda.is_available())"
python prepare.py
python run_experiments.py --mode baseline
python run_experiments.py --mode smoke
python run_experiments.py --mode grid
# Write your own 40 messages BEFORE final evaluation.
python run_experiments.py --mode final
python app.py
python -m pytest
```

`python run_experiments.py` runs the grid and final evaluation in one command on a clean workspace, after the independent messages are ready. Model weights in model/ are ignored; this command recreates them. The script deliberately refuses to overwrite final results; for reproduction, use a fresh clone and move the committed final.json outside the clone first. Do not use repeated test results to choose settings. Smoke and baseline runs write only scratch/; grid runs write reports/. First neural run needs internet. Default device is CPU unless CUDA is available. For CPU-only PyTorch, install the matching CPU wheel from the official PyTorch wheel index if necessary. Pins are proposed, not verified on Python 3.12 here; confirm with pip freeze and update direct pins if needed.

## Data and split
Use unchanged tickets_data.py defaults: 1200 invented tickets. Label counts: billing 277, bike_fault 347, station 331, account 245. Split seed 42; 720/240/240 disjoint ticket IDs, stratified by label. Repeated text patterns may occur across distinct IDs; this is a limitation of the specified split. Training samples: 50, 100, 300; run seeds 42, 123, 2026.

## Methods
Majority and TF-IDF (unigrams/bigrams, logistic regression) are trained on the same sample as each DistilBERT variant. The body is frozen for head-only training; all weights are trainable for full fine-tuning. Dynamic batch padding, maximum length 96, batch size 16, AdamW. Validation-only pilot at 100 rows, seed 42, compares head (0.001/2 epochs vs 0.0005/3) and full (0.00002/2 vs 0.00005/3); choose the highest final validation accuracy per variant, tie by first candidate. The grid uses these settings for all sizes and seeds. Select highest mean validation accuracy at 300 rows; choose best neural variant separately for publishing. Final refit uses predeclared seed 42 and the same 300 rows for all final methods. Baselines and best neural model are evaluated once on held-out test and own messages. No test-driven selection.

## Results
Preliminary measured validation-only baseline results from package preparation (local environment, not the final pinned environment):

| Method | n | Mean | Min | Max |
|---|---:|---:|---:|---:|
| Majority | 50 | .2875 | .2875 | .2875 |
| Majority | 100 | .2875 | .2875 | .2875 |
| Majority | 300 | .2875 | .2875 | .2875 |
| TF-IDF | 50 | .6792 | .6667 | .6875 |
| TF-IDF | 100 | .7931 | .7542 | .8375 |
| TF-IDF | 300 | .8653 | .8625 | .8667 |

TODO replace with complete measured table from reports/summary.csv after grid. Tuning runs and logs are in tuning.csv and tuning_epochs.csv; grid runs in results.csv and epochs.csv. Add final test/own accuracy and macro F1 from final.json. Embed the generated learning curve and confusion matrix using Markdown image links. Do not claim results before execution.

## Your own 40 messages
Write 10 original messages per label in data/own_messages.csv, with header text,label. Include varied lengths and occasional spelling errors. Do not use the generator, AI-written examples, or the handout's sentences. Do not train or tune on these messages. TODO discuss the final independent results.

## Publishing
```bash
hf auth login
python publish.py --model-id YOUR_USERNAME/lakeshore-ticket-sorter --space-id YOUR_USERNAME/lakeshore-ticket-sorter
```
The script updates the local Space app with your model ID and uploads public model and Space repositories. Replace links at the top. Never store a token in the project. Open the Space in a private browser window, try a message and empty text.

## Tests
Run `python -m ruff format --check .`, `python -m ruff check .`, and `python -m pytest`. Unit tests use a tiny randomly initialized DistilBERT for fast offline checks; the actual experiment uses the pretrained base. Five data/baseline cases passed during preparation. Neural tests and Ruff are unverified here. Run formatter and linter locally; preserve the supplied generator unchanged.

## Limitations
Invented messages, repeated patterns, vague and mixed-topic tickets, small training samples, seed variability, and uncalibrated confidence scores. No real customer evaluation. Predictions should be reviewed by a human.
