import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_folder(folder_path, repo_id, repo_type="dataset", token=None):
    api = HfApi(token=token)
    
    # Ensure repo exists
    print(f"Creating/Checking repo {repo_id}...")
    try:
        url = create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token)
        print(f"Repo URL: {url}")
    except Exception as e:
        print(f"Note: Repo creation might have failed or needs auth: {e}")
    
    print(f"Uploading {folder_path} to {repo_id}...")
    
    # Upload folder
    # We use multi_commits to handle large number of files/size if needed, 
    # though upload_folder usually handles large files well via LFS if configured or just normal chunks.
    # For many tar files, standard upload is usually fine.
    
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=".", # Upload to root or subfolder
        allow_patterns=["*.tar", "*.json", "*.md"],
        token=token
    )
    print("Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to local folder containing shards")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--token", type=str, help="HF Auth Token (optional if logged in via CLI)")
    args = parser.parse_args()
    
    upload_folder(args.folder, args.repo_id, token=args.token)
