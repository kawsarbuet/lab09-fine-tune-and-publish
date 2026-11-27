# Lab 09 completion guide (not a submission report)

## 1. Repository and commits
Create an empty GitHub repository named lab09-fine-tune-and-publish; do not initialize it with a README. Extract ZIP and open the project folder in VS Code. First commit must contain ONLY the supplied tickets_data.py.

```bash
git status
git clone https://github.com/kawsarbuet/lab09-fine-tune-and-publish
git branch -M main
cd lab09-fine-tune-and-publish
git add tickets_data.py
git commit -m "Add unchanged instructor ticket generator"
git push
git add .gitignore prepare.py baselines.py environment.yml requirements.txt requirements-dev.txt pyproject.toml tests/test_data.py
git commit -m "Add stratified split, baseline models and environment"
git push
git add finetune.py evaluate.py run_experiments.py tests/test_model.py share.py
git commit -m "Implement validation tuning and DistilBERT comparison"
git push
git add app.py publish.py space
git commit -m "Add local demo and Hub publication workflow"
git push
```
Write your own messages, run the steps in README, update reports and links, then:

```bash
git status
git add data/own_messages.csv reports README.md REPORT.md space/app.py
git commit -m "Record measured results and publish model and demo"
git push
git add .
git commit -m "Complete DistilBERT fine-tuning, final evaluation and experimental reports"
git push
git add SETUP_AND_SUBMISSION.md
git commit -m "Document reproducibility steps"
git branch -M main
git remote add origin https://github.com/kawsarbuet/lab09-fine-tune-and-publish
git push -u origin main
```
If the remote is already configured, skip remote add. Run `git status` and confirm no token or model weights are tracked. If you prefer a private repository, share only with the instructor email listed on page 7 of the PDF.

## 2. Data exploration
`python prepare.py` prints counts for each split. Use a Python console:
```python
from tickets_data import make_tickets
frame = make_tickets()
print(frame.label.value_counts())
print(frame.groupby('label').head(12).to_string(index=False))
```
Read examples, including vague messages. Write your independent CSV yourself, with 10 messages for each of the four labels. Quote fields containing commas using a CSV editor. Never use these rows for tuning.

## 3. Install and run
Follow README commands. If conda is not recognized use Anaconda Prompt. If a CPU run is slow, use the smoke mode first; do not substitute smoke results for the full grid. Full grid has 36 runs (18 baseline and 18 neural), plus 4 validation-only tuning trials and a final refit. Grid outputs tuning.csv, tuning_epochs.csv, settings.json, results.csv, epochs.csv, summary.csv, learning_curve.png. Final outputs final.json, final_epochs.csv and confusion_matrix.png, and the ignored model directory. Use measured outputs to complete README and REPORT.

## 4. Checks and fixes
Run `python -m ruff format .` and `python -m ruff check . --fix` on your authored modules (exclude tickets_data.py if a tool proposes changes to it). Then run all three official checks. Fix remaining lint errors manually, rerun tests. Package preparation did not have Ruff or Transformers installed; only data/baseline tests were executed. Confirm exact versions via `python -m pip freeze`. For Windows use the activated Conda interpreter in VS Code.

## 5. Publication
Create your own Hugging Face account and authenticate with `hf auth login` locally, using a write-capable token. Do not paste a token into code. Run publish.py as described in README. Check links publicly; Space must return all four scores and a clear blank-input message.

## 6. Fresh clone verification
Clone into a new folder, activate/recreate environment and run the three official checks. To reproduce final outputs, move final.json out of reports in that new clone before running the full experiment; compare the regenerated files against the committed ones without changing settings based on test performance. Model weights must not be committed. Exact GPU results can vary by hardware; disclose differences.

## 7. Submit
Submit GitHub URL and public Space URL to Blackboard, not just the ZIP. Verify all required reports are tracked. Keep the model repository public. Deadline is on Blackboard. Lab 09 is a separate repository; do not merge earlier lab code or commits into it. Earlier labs provide reusable practices: baseline-first, fixed splits, reproducibility, model saving/reloading, pytest/Ruff, Gradio and Hugging Face deployment.
