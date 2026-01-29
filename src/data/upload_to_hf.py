import argparse
import os
from huggingface_hub import HfApi, login

# You can hardcode your token here if you prefer
# HF_TOKEN = "hf_..."
HF_TOKEN = None

def upload_shards(repo_id, input_dir, repo_type="dataset", path_in_repo=None, num_workers=4, token=None):
    """
    Uploads shards from input_dir to the specified Hugging Face repository.
    
    Args:
        repo_id (str): The ID of the repository (e.g., "username/dataset_name").
        input_dir (str): Local path to the directory containing shards.
        repo_type (str): Type of repository ("dataset", "model", "space").
        path_in_repo (str): Path in the repository where files will be uploaded. 
                            If None, files are uploaded to root.
        num_workers (int): Number of workers for large folder upload.
        token (str): Hugging Face token.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Prioritize argument token, then global variable
    auth_token = token or HF_TOKEN
    
    if auth_token:
        print(f"Logging in with provided token...")
        login(token=auth_token)
        api = HfApi() # Uses the token from login()
    else:
        api = HfApi()

    # Check if we should use upload_folder or upload_large_folder
    # For many shards, upload_folder is generally fine and easier to manage for single commit.
    # But if it's huge, we might consider upload_large_folder.
    # Let's stick to upload_folder as it's the standard way for datasets unless extreme size.
    
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}...")
            api.upload_folder(
                folder_path=input_dir,
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=path_in_repo
            )
            print("Upload completed successfully!")
            break
        except Exception as e:
            print(f"Error during upload (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                print("Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print("Failed after multiple attempts.")
                print("Hint: Make sure you are logged in and have a stable internet connection.")
                raise e

def main():
    parser = argparse.ArgumentParser(description="Upload shards to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., username/dataset)")
    parser.add_argument("--input_dir", type=str, default="data/shards", help="Directory containing the shards")
    parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model", "space"], help="Repository type")
    parser.add_argument("--path_in_repo", type=str, default=None, help="Path within the repository to upload to")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional if logged in via CLI)")

    args = parser.parse_args()

    upload_shards(
        repo_id=args.repo_id,
        input_dir=args.input_dir,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        token=args.token
    )

if __name__ == "__main__":
    main()
