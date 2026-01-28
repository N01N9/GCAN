import os
import zipfile
import glob

def backup_code(output_filename="code_backup.zip"):
    """
    Backs up code files to a zip archive.
    """
    
    files_to_backup = [
        "config.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    dirs_to_backup = [
        "src"
    ]
    
    print(f"Creating backup: {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add individual files
        for f in files_to_backup:
            if os.path.exists(f):
                print(f"Adding {f}")
                zipf.write(f)
            else:
                print(f"Warning: {f} not found.")
                
        # Add directories
        for d in dirs_to_backup:
            if not os.path.exists(d):
                print(f"Warning: Directory {d} not found.")
                continue
                
            for root, _, files in os.walk(d):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Ignore pycache and compiled python files
                    if "__pycache__" in file_path or file.endswith(".pyc") or file.endswith(".zip") or file.endswith(".tar"):
                        continue
                        
                    print(f"Adding {file_path}")
                    zipf.write(file_path)
                    
    print(f"Backup completed! Saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    # Ensure src/utils exists (it should if this script is here)
    backup_code()
