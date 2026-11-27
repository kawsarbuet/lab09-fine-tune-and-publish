"""Publish using the credentials from local hf auth login."""

import argparse
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


def publish(model_id, space_id):
    api = HfApi()
    if not Path("model/README.md").exists():
        raise SystemExit("Run final evaluation first.")
    api.create_repo(model_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=model_id, repo_type="model", folder_path="model")
    source = Path("space/app.py")
    text = source.read_text()
    source.write_text(
        re.sub(r"^MODEL_ID = .*", f"MODEL_ID = {model_id!r}", text, flags=re.M)
    )
    with TemporaryDirectory() as temp:
        target = Path(temp) / "space"
        shutil.copytree("space", target)
        app = target / "app.py"
        app.write_text(app.read_text().replace("REPLACE_WITH_MODEL_ID", model_id))
        api.create_repo(
            space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(target))
    print(f"Model: https://huggingface.co/{model_id}")
    print(f"Space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--space-id", required=True)
    args = parser.parse_args()
    publish(args.model_id, args.space_id)
