import torch
import torchaudio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import RefinedMultiStreamGCAN

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Default parameters if not in config
    num_slots = config.get('num_slots', 6)
    hidden_dim = config.get('hidden_dim', 256) # config might differ, strictly should follow train args default
    
    # If config doesn't have these, use defaults from train.py lookalike
    # But usually config comes from vars(args).
    
    print(f"Model Config: Num Slots={num_slots}, Hidden Dim={hidden_dim}")
    
    model = RefinedMultiStreamGCAN(num_speakers=num_slots, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_audio(audio_path, target_sr=16000, max_len_sec=20.0):
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # Chunking or Padding
    # For simplicity, let's take the first max_len_sec or pad
    num_samples = int(target_sr * max_len_sec)
    if wav.shape[1] > num_samples:
        print(f"Warning: Audio is {wav.shape[1]/target_sr:.2f}s, truncating to {max_len_sec}s")
        wav = wav[:, :num_samples]
    elif wav.shape[1] < num_samples:
        pad_amt = num_samples - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad_amt))
        
    return wav

def format_output_rttm(assignments, existence, audio_id="audio", threshold=0.5, hop_length=160, sr=16000):
    # assignments: (1, T, K) logits
    # existence: (1, K) logits (or existence prob) -> Actually model outputs aggregated existence?
    # Let's check model.py: existence = assignments.max(dim=1)[0] (logits)
    
    # Apply sigmoid
    probs = torch.sigmoid(assignments).squeeze(0).cpu().numpy() # (T, K)
    exist_probs = torch.sigmoid(existence).squeeze(0).cpu().numpy() # (K,)
    
    # Active speakers based on existence threshold
    active_indices = np.where(exist_probs > threshold)[0]
    
    predictions = [] # (start, end, speaker_id)
    
    frame_dur = hop_length / sr
    
    print(f"\nRaw Existence Probabilities: {np.round(exist_probs, 4)}")
    print(f"Detected {len(active_indices)} speakers (Threshold {threshold}): {active_indices.tolist()}")
    
    # Parameters for post-processing
    median_window = 11  # Must be odd
    min_duration_sec = 0.1 # Minimum segment duration
    
    from scipy.signal import medfilt
    
    for k in active_indices:
        spk_id = k.item()
        
        # 1. Get raw probabilities
        p = probs[:, k]
        
        # 2. binarize
        binary = p > threshold
        
        # 3. Median Filtering (Smoothing) to remove jitter
        # This fills small gaps and removes small spikes
        binary_smoothed = medfilt(binary.astype(float), kernel_size=median_window).astype(bool)
        
        # 4. Extract segments
        is_active = False
        start_t = 0.0
        
        segments = []
        
        for t, active in enumerate(binary_smoothed):
            if active and not is_active:
                is_active = True
                start_t = t * frame_dur
            elif not active and is_active:
                is_active = False
                end_t = t * frame_dur
                segments.append((start_t, end_t))
        
        if is_active:
            segments.append((start_t, len(binary_smoothed)*frame_dur))
            
        # 5. Filter by minimum duration
        for start, end in segments:
            if (end - start) >= min_duration_sec:
                predictions.append((start, end, spk_id))
            
    # Sort by start time
    predictions.sort(key=lambda x: x[0])
    
    return predictions

def visualize(wav, predictions, probs, active_indices, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # 1. Waveform & Segments
    ax1.plot(wav.t().numpy(), alpha=0.5, color='gray', label='Waveform')
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for start, end, spk_id in predictions:
        ax1.hlines(y=spk_id + 1, xmin=start*16000, xmax=end*16000, 
                   linewidth=4, color=colors[spk_id % 10], label=f'Spk {spk_id}')
        
    ax1.set_ylabel('Speaker ID')
    ax1.set_title('Diarization Result')
    
    # Deduplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # 2. Probability Curves
    time_axis = np.arange(probs.shape[0]) * 160 # frames to samples
    
    for k in active_indices:
        spk_id = k.item()
        ax2.plot(time_axis, probs[:, k], color=colors[spk_id % 10], label=f'Spk {spk_id}', alpha=0.8)
        
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Samples')
    ax2.set_title('Speaker Activity Probabilities')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="GCAN Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output', type=str, default='result.png', help='Path to save visualization')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = load_model(args.checkpoint, device)
    
    # Process Audio
    wav = preprocess_audio(args.audio)
    wav_in = wav.to(device) # (1, T)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # model expects (B, T), inputs are (1, n_samples)
        out = model(wav_in) 
        probs = torch.sigmoid(out['assignments']).squeeze(0).cpu().numpy()
        exist_probs = torch.sigmoid(out['existence']).squeeze(0).cpu().numpy()
        
    # Post-process
    # We pass pre-calculated probs to allow re-using them for viz
    # But format_output_rttm used to calculate them. Let's patch format_output_rttm too or just duplicate logic?
    # Better to patch format_output_rttm to accept probs.
    
    # Let's adjust format_output_rttm call logic inline here as tool replacement is chunks.
    # The previous replace replaced visualize and main. 
    # I need to make sure format_output_rttm signature matches what I call in main or modify it.
    
    # Wait, the user tool limits me to replacing chunks.
    # I will replace `visualize` and `main` fully, but I also need to update `format_output_rttm` signature OR
    # just adapt `main` to pass what `format_output_rttm` expects?
    # `format_output_rttm` expects `assignments` and `existence`.
    # I can keep `format_output_rttm` as is, and just extract `active_indices` from it?
    # No, `format_output_rttm` prints stuff and returns predictions. It doesn't return active_indices.
    
    # I'll just recalculate active_indices in main for visualization. It's cheap.
    
    preds = format_output_rttm(out['assignments'], out['existence'], threshold=args.threshold)
    
    # Recalculate active indices for viz
    active_indices = np.where(exist_probs > args.threshold)[0]
    
    # Print RTTM
    print("\n--- RTTM Output ---")
    for start, end, spk in preds:
        rec_id = Path(args.audio).stem
        # SPEAKE <REC_ID> 1 <START> <DUR> <NA> <NA> <SPK> <CONF>
        dur = end - start
        print(f"SPEAKER {rec_id} 1 {start:.3f} {dur:.3f} <NA> <NA> spk{spk} <NA>")
        
    # Visualize
    visualize(wav, preds, probs, active_indices, args.output)

if __name__ == "__main__":
    main()
