# Lab 09 completion guide


- GitHub: https://github.com/kawsarbuet/lab09-fine-tune-and-publish
- Model: https://huggingface.co/kawsarbuet/lakeshore-ticket-sorter
- Space: https://huggingface.co/spaces/kawsarbuet/lakeshore-ticket-sorter

## 1. Repository and commits

Create an empty GitHub repository named `lab09-fine-tune-and-publish`.
Do not initialize the repository with a README, license or .gitignore.

The first commit must contain ONLY the original instructor-provided
`tickets_data.py` file. This file must remain unchanged throughout
the project.

### 1.1 Clone and configure the repository

```bash
git clone https://github.com/kawsarbuet/lab09-fine-tune-and-publish
cd lab09-fine-tune-and-publish
git branch -M main
git status
```

Extract the supplied project package into the cloned repository
and open the project folder in VS Code.

### 1.2 Commit the original dataset generator

```bash
git add tickets_data.py
git commit -m "Add unchanged instructor ticket generator"
git push
```

### 1.3 Commit data preparation, baselines and environment configuration

```bash
git add .gitignore prepare.py baselines.py environment.yml requirements.txt requirements-dev.txt pyproject.toml tests/test_data.py
git commit -m "Add stratified split, baseline models and environment"
git push
```

### 1.4 Commit fine-tuning and evaluation implementation

```bash
git add finetune.py evaluate.py run_experiments.py tests/test_model.py share.py
git commit -m "Implement validation tuning and DistilBERT comparison"
git push
```

### 1.5 Commit Gradio application and publication workflow

```bash
git add app.py publish.py space/
git commit -m "Add local demo and Hugging Face publication workflow"
git push
```

### 1.6 Commit experimental results and final evaluation

Complete the baseline experiments, hyperparameter tuning,
full grid experiment and final evaluation before committing
the generated reports.

Prepare the independent evaluation dataset according to
the lab requirements.

```bash
git add data/own_messages.csv reports/
git commit -m "Complete DistilBERT fine-tuning, final evaluation and experimental reports"
git push
```

### 1.7 Commit application verification and Hugging Face deployment

Test the local Gradio application across all four categories.
Publish the trained model and Gradio Space to Hugging Face.
Verify public accessibility and classification functionality.

Published model:

https://huggingface.co/kawsarbuet/lakeshore-ticket-sorter

Published Space:

https://huggingface.co/spaces/kawsarbuet/lakeshore-ticket-sorter

Apply Ruff formatting to the Space application after publication.

```bash
python -m ruff format space/app.py
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

Commit the updated application files if changes exist.

```bash
git add app.py publish.py space/
git commit -m "Finalize Gradio application and Hugging Face deployment"
git push
```

### 1.8 Commit final documentation

Update README.md and REPORT.md with measured experimental
results, independent evaluation findings, limitations and
Hugging Face deployment links.

Update SETUP_AND_SUBMISSION.md with reproducibility,
troubleshooting and submission instructions.

```bash
git add .
git commit -m "Finalize documentation, experimental findings and reproducibility steps"
git push
```

### 1.9 Final repository verification

Verify the repository status, commit history and remote URL.

```bash
git status
git log --oneline
git log --reverse --oneline
git remote -v
```

Verify that the original instructor file remains unchanged
relative to the first commit.

```bash
git diff FIRST_COMMIT_HASH HEAD -- tickets_data.py
```

Replace FIRST_COMMIT_HASH with the actual first commit hash.

Verify that model weights are excluded from Git tracking.

```bash
git check-ignore model/model.safetensors
git ls-files model/
```

Verify that the required experimental reports are tracked.

```bash
git ls-files reports/
```

Check for potential Hugging Face access-token exposure in
tracked files and available Git history.

```bash
git grep -l -E "hf_[A-Za-z0-9]{20,}"
git log --all -G "hf_[A-Za-z0-9]{20,}" --oneline
```

No access tokens, passwords, environment credentials or
trained model weights should be committed to GitHub.

These pattern checks are supplementary security checks and
do not guarantee that all types of credentials are detected.

The repository must contain at least five meaningful commits,
with the first commit containing only tickets_data.py.

If the remote repository is already configured, no additional
remote configuration is required.

The repository may be public or private and shared only with
the instructor specified on page 7 of the lab instructions.


## 2. Data exploration

The original `tickets_data.py` generator produces 1,200 synthetic
support tickets across four categories: billing, bike_fault,
station and account.

A stratified 60/20/20 split was performed using random seed 42:

| Dataset | Rows | Percentage |
|---|---:|---:|
| Training | 720 | 60% |
| Validation | 240 | 20% |
| Test | 240 | 20% |
| Total | 1,200 | 100% |

The experiments used training sample sizes of 50, 100 and 300,
with random seeds 42, 123 and 2026.

### Dataset exploration

```python
from tickets_data import make_tickets

frame = make_tickets()
print(frame.label.value_counts())
print(frame.groupby("label").head(12).to_string(index=False))
```
exit()
```
### Independent evaluation dataset

The independent evaluation dataset must contain 40 personally
written support messages, with 10 messages per category.

The dataset is stored in `data/own_messages.csv` using the
columns `text` and `label`. Messages should include varied
sentence lengths, natural wording and occasional spelling errors.

These messages are reserved exclusively for final evaluation
and must not be used for model training, validation or
hyperparameter selection.

Any AI-generated draft must be replaced with personally
authored messages before final submission.



## 3. Install and run

The following is the reproduction sequence, not a request to repeat completed training. Use Anaconda Prompt and the repository root. Create the environment only if it does not already exist.

```cmd
conda env create -f environment.yml
conda activate lab09-fine-tune-and-publish
python prepare.py
python run_experiments.py --mode baseline
python run_experiments.py --mode smoke
python run_experiments.py --mode grid
python run_experiments.py --mode final
python app.py
```

Prepare the independent messages before final evaluation. Stop Gradio with Ctrl+C after testing. The local URL is http://127.0.0.1:7860.

The completed grid comprises 36 comparison runs (18 baseline and 18 neural), plus four validation-only tuning trials and a final refit. Selected settings observed in the logs: head-only 0.001 / 2 epochs; full fine-tuning 0.00005 / 3 epochs. Full fine-tuning reached 90.42% mean validation accuracy at 300 samples versus 86.53% for TF-IDF. TF-IDF led at 50 and 100 samples.

The earlier `FileNotFoundError` for `reports/results.csv` occurred when final evaluation was invoked without completed grid results. The grid subsequently finished and final evaluation was reported complete. Do not fabricate or substitute missing results.

```cmd
dir reports
type reports\final.json
dir model
```

Expected grid outputs: `tuning.csv`, `tuning_epochs.csv`, `settings.json`, `results.csv`, `epochs.csv`, `summary.csv`, `learning_curve.png`. Final outputs: `final.json`, `final_epochs.csv`, `confusion_matrix.png`, and the ignored `model/` directory. Exact final metrics must be copied from the actual `final.json`, not inferred from validation accuracy.

## 4. Checks and fixes

Ruff and pytest completion was reported locally. The following corrections were made or recommended during troubleshooting:

- Format authored files without changing `tickets_data.py`.
- Resolve `I001` import ordering in `evaluate.py` using Ruff's automatic fix.
- Resolve `E501` in `share.py` by wrapping model-card prose and in `tests/test_model.py` by splitting the long comment. Preserve the 88-character limit.

```cmd
python -m ruff format . --exclude tickets_data.py
python -m ruff check . --fix --exclude tickets_data.py
python -m ruff format --check .
python -m ruff check .
python -m pytest
python -m pip freeze
```

The exact final pytest count and installed dependency versions have not been supplied. Verify the pins against the actual environment and retain the official check outputs. Do not alter the instructor generator or weaken lint rules to pass checks.

## 5. Publication

Publication completed on November 27, 2026. The terminal showed `model.safetensors` uploaded at 100% (268 MB in 40 seconds, reported 6.57 MB/s). Container logs confirmed successful application startup on port 7860.

For reproduction, authenticate with a write-capable Hugging Face token. The `hf` command was not recognized in the local environment; the Python API provides an alternative:

```cmd
python -c "from huggingface_hub import login; login()"
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
python publish.py --model-id kawsarbuet/lakeshore-ticket-sorter --space-id kawsarbuet/lakeshore-ticket-sorter
```

Do not include the token in source files, logs shared publicly, or commits. Do not republish unless needed. Open the public Space in a private browser window and test billing, bike fault, station and account inputs plus empty input. Confirm four scores and a clear empty-input message. `share=True` is unnecessary. Public prediction testing has not been independently confirmed.

If publication changed tracked Space files, review and commit those changes:

```cmd
git status
git add space/
git commit -m "Publish fine-tuned model and Gradio Space on Hugging Face"
git push
```

## 6. Fresh clone verification

Before the final push, verify that model weights are ignored and reports are tracked:

```cmd
git check-ignore model/model.safetensors
git ls-files model/
git ls-files reports/
```

`git ls-files model/` should be empty. Inspect staged files for secrets. After completing README and REPORT using actual final metrics, stage the intended files:

```cmd
git add README.md REPORT.md SETUP_AND_SUBMISSION.md reports/ data/own_messages.csv
git status
git commit -m "Finalize Lab 09 documentation and reproducibility steps"
git push
```

From the parent directory, clone into a separate, unused folder and run the mandatory checks:

```cmd
cd ..
git clone https://github.com/kawsarbuet/lab09-fine-tune-and-publish.git lab09-verification
cd lab09-verification
conda activate lab09-fine-tune-and-publish
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

To reproduce final outputs, move the committed `final.json` outside that new clone before running the full experiment, as required by the original guide. Compare regenerated outputs without tuning on the held-out data. Hardware differences may affect reproducibility. Do not delete the original results or retrain simply to improve test scores. Fresh-clone verification remains to be confirmed.

## 7. Submit

- [x] Complete comparison grid (terminal summary observed).
- [x] Complete final evaluation and local application testing (reported by user).
- [x] Complete local Ruff and pytest checks (reported by user).
- [x] Publish model and Space (upload and startup logs observed).
- [ ] Verify public Space predictions and accessibility.
- [ ] Confirm personally authored independent messages and actual final metrics.
- [ ] Complete README final metrics and REPORT.md (300–450 words).
- [ ] Verify pins, original generator, reports, Git history and absence of model weights or tokens.
- [ ] Push final changes and pass the three checks in a fresh clone.
- [ ] Submit GitHub and Space URLs to Blackboard before the posted deadline.

Submission links:

- GitHub: https://github.com/kawsarbuet/lab09-fine-tune-and-publish
- Space: https://huggingface.co/spaces/kawsarbuet/lakeshore-ticket-sorter

