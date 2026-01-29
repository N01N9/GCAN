from huggingface_hub import list_repo_files
import argparse

def debug_repo(repo_id):
    print(f"Listing files in repo: {repo_id}")
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        print(f"Total files: {len(files)}")
        for f in files[:20]: # Print first 20 files
            print(f"  - {f}")
        if len(files) > 20:
            print("  ...")
            
        tar_files = [f for f in files if f.endswith(".tar")]
        print(f"\nFound {len(tar_files)} total .tar files.")
        
        train_tars = [f for f in tar_files if "train" in f]
        print(f"Found {len(train_tars)} .tar files containing 'train'.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()
    debug_repo(args.repo_id)
