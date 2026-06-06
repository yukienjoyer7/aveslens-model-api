import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload(
    onnx_path: str | Path = "checkpoints/model.onnx",
    labels_path: str | Path = "labels.json",
    repo_id: str = "",
) -> None:
    if not repo_id:
        raise ValueError("repo_id is required, e.g. your-username/aveslens-weights")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_file(path_or_fileobj=str(onnx_path), path_in_repo="model.onnx", repo_id=repo_id)
    print(f"uploaded {onnx_path} → {repo_id}/model.onnx")

    api.upload_file(path_or_fileobj=str(labels_path), path_in_repo="labels.json", repo_id=repo_id)
    print(f"uploaded {labels_path} → {repo_id}/labels.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF Hub repo, e.g. username/aveslens-weights")
    parser.add_argument("--onnx", default="checkpoints/model.onnx")
    parser.add_argument("--labels", default="labels.json")
    args = parser.parse_args()

    upload(onnx_path=args.onnx, labels_path=args.labels, repo_id=args.repo_id)
