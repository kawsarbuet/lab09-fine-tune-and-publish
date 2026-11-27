"""Save trained weights and generate a model card from measured evidence."""
import json
from pathlib import Path

from finetune import BASE


def model_card(final):
    return f'''---
license: apache-2.0
base_model: {BASE}
pipeline_tag: text-classification
---
# Lakeshore Ticket Sorter

## Intended use
Suggest a support queue for bike-share tickets; human review is required.

## Training data
Invented messages; {final['n']} training examples from a fixed stratified split.

## Results
All listed methods used the same training sample and evaluation rows.

```json
{json.dumps(final, indent=2)}
```

## Limitations
Training text was invented. Vague and mixed-topic messages can be ambiguous.
Scores are not calibrated confidence estimates and do not establish real-world performance.
'''


def save_model(model, tokenizer, final, path='model'):
    folder = Path(path)
    folder.mkdir(exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    (folder / 'README.md').write_text(model_card(final), encoding='utf-8')
