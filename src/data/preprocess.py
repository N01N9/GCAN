import os
import glob
import torch
import torchaudio
import yaml
import json
import tqdm
import numpy as np
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_audio_files(root_path):
    # VoxCeleb structure: id/video_id/wav
    # We want to group by speaker ID
    speaker_files = {}
    
    # Support both glob patterns and direct paths
    files = glob.glob(os.path.join(root_path, "**", "*.wav"), recursive=True) + \
            glob.glob(os.path.join(root_path, "**", "*.m4a"), recursive=True)
            
    print(f"Found {len(files)} files in {root_path}")
    
    for f in files:
        # Extract speaker ID. Assuming structure .../idXXXXX/...
        parts = f.split(os.sep)
        # Heuristic: find the part starting with 'id1' or 'id0'
        spk_id = None
        for p in parts:
            if p.startswith('id') and p[2:].isdigit() and len(p) >= 7:
                spk_id = p
                break
        
        if spk_id:
            if spk_id not in speaker_files:
                speaker_files[spk_id] = []
            speaker_files[spk_id].append(f)
            
    return speaker_files

def compute_embeddings(speaker_files, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
    except Exception as e:
        print(f"Failed to load SpeechBrain model: {e}")
        return

    embeddings = {}
    
    print("Computing embeddings...")
    for spk_id, files in tqdm.tqdm(speaker_files.items()):
        # Compute embedding for a few random files and average them to get a robust speaker embedding
        # Or just take the first few to save time
        
        selected_files = np.random.choice(files, min(len(files), 5), replace=False)
        spk_embs = []
        
        for f_path in selected_files:
            try:
                signal, fs = torchaudio.load(f_path)
                # Resample if needed? Model usually handles 16k
                # ECAPA expects [batch, time]
                emb = classifier.encode_batch(signal)
                spk_embs.append(emb.squeeze().cpu().numpy())
            except Exception as e:
                # print(f"Error processing {f_path}: {e}")
                pass
        
        if spk_embs:
            # Average embeddings
            avg_emb = np.mean(spk_embs, axis=0)
            embeddings[spk_id] = avg_emb.tolist() # Convert to list for JSON serialization

    # Convert to matrix for similarity search later
    # We will save as a dictionary map and a matrix
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'speaker_embeddings.json'), 'w') as f:
        json.dump(embeddings, f)
        
    print(f"Saved embeddings for {len(embeddings)} speakers to {output_path}")

def main():
    config = load_config()
    all_speakers = {}
    
    # Iterate over splits
    for split in ['train', 'test']:
        if split not in config['data']['paths']:
            print(f"No configuration for {split}. Skipping.")
            continue
            
        split_speakers = {}
        paths = config['data']['paths'][split]
        
        # Paths is now a dict {name: path}
        for name, path in paths.items():
            if not path or not os.path.exists(path):
                print(f"Warning: Path {path} for {name} ({split}) does not exist. Skipping.")
                continue
                
            print(f"Scanning {split}/{name} at {path}...")
            spk_files = get_audio_files(path)
            
            for spk, files in spk_files.items():
                if spk in split_speakers:
                    split_speakers[spk].extend(files)
                else:
                    split_speakers[spk] = files

        if not split_speakers:
            print(f"No speakers found for {split}.")
            continue

        # Save file list for this split
        os.makedirs(config['preprocess']['output_path'], exist_ok=True)
        out_file = os.path.join(config['preprocess']['output_path'], f'file_list_{split}.json')
        with open(out_file, 'w') as f:
            json.dump(split_speakers, f)
        print(f"Saved {split} list to {out_file}")

        # Compute Embeddings ONLY for TRAIN
        if split == 'train':
             compute_embeddings(split_speakers, config['preprocess']['output_path'])

if __name__ == "__main__":
    main()
