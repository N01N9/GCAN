import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class SpectralStream(nn.Module):
    """Enhanced Spectral Stream with deeper residual blocks"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv3x3 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(1, 16, kernel_size=7, padding=3)
        self.skip_proj = nn.Conv2d(48, 64, kernel_size=1)
        
        # Enhanced residual blocks (2 blocks instead of 1)
        self.res_block1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.proj = nn.Linear(64 * 80, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor):
        feat = torch.cat([self.conv3x3(mel), self.conv5x5(mel), self.conv7x7(mel)], dim=1)
        res1 = self.res_block1(feat)
        out1 = F.relu(self.skip_proj(feat) + res1)
        res2 = self.res_block2(out1)
        out = F.relu(out1 + res2)
        B, C, T, F_bins = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.dropout(self.proj(out))


class ProsodicStream(nn.Module):
    """Enhanced Prosodic Stream with 2-layer Bi-LSTM"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            4, hidden_dim // 2, 
            num_layers=2,  # Increased from 1 to 2
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio: torch.Tensor):
        frames = audio.unfold(1, 512, 160)
        e = torch.log(frames.pow(2).mean(-1) + 1e-6)
        z = (audio.sign().diff().abs() > 0).float().unfold(1, 512, 160).mean(-1)
        ed, zd = F.pad(e[:, 1:]-e[:, :-1], (1,0)), F.pad(z[:, 1:]-z[:, :-1], (1,0))
        out, _ = self.lstm(torch.stack([e, z, ed, zd], dim=-1))
        return self.layer_norm(out)


class RhythmStream(nn.Module):
    """Enhanced Rhythm Stream with deeper TCN"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, hidden_dim, 5, padding=2)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio: torch.Tensor):
        z = (audio.sign().diff().abs() > 0).float().unfold(1, 512, 160).mean(-1).unsqueeze(1)
        out = self.tcn(z).transpose(1, 2)
        return self.layer_norm(out)


class EnergyStream(nn.Module):
    """Enhanced Energy Stream with MLP"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio: torch.Tensor):
        s = torch.stft(audio, n_fft=512, hop_length=160, return_complex=True, 
                       window=torch.hann_window(512).to(audio.device)).abs()
        x = torch.stack([s[:,:8,:].mean(1), s[:,8:32,:].mean(1), 
                         s[:,32:64,:].mean(1), s[:,64:,:].mean(1)], dim=-1)
        return self.layer_norm(self.mlp(x))


class PhoneticStream(nn.Module):
    """Enhanced Phonetic Stream with multi-scale convolutions"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, 64, stride=160, padding=32)
        self.conv2 = nn.Conv1d(1, hidden_dim // 2, 128, stride=160, padding=64)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio: torch.Tensor):
        c1 = self.conv1(audio.unsqueeze(1)).transpose(1, 2)
        c2 = self.conv2(audio.unsqueeze(1)).transpose(1, 2)
        # Align sizes
        min_len = min(c1.size(1), c2.size(1))
        out = torch.cat([c1[:, :min_len], c2[:, :min_len]], dim=-1)
        return self.layer_norm(self.dropout(self.proj(out)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiLayerAttractorDecoder(nn.Module):
    """Enhanced Multi-Layer Attractor Decoder with iterative refinement"""
    def __init__(self, num_speakers, hidden_dim, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.query_embed = nn.Embedding(num_speakers, hidden_dim)
        
        # Multi-layer decoder
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm3_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, context_feat):
        B = context_feat.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        for i in range(self.num_layers):
            # Self-attention
            q2, _ = self.self_attn_layers[i](queries, queries, queries)
            queries = self.norm1_layers[i](queries + q2)
            
            # Cross-attention with context
            q3, _ = self.cross_attn_layers[i](queries, context_feat, context_feat)
            queries = self.norm2_layers[i](queries + q3)
            
            # FFN
            queries = self.norm3_layers[i](queries + self.ffn_layers[i](queries))
        
        return queries


class RefinedMultiStreamGCAN(nn.Module):
    """Enhanced GCAN with deeper architecture and improved components"""
    def __init__(self, num_speakers=6, hidden_dim=256, num_transformer_layers=6, dropout=0.1):
        super().__init__()
        self.num_speakers = num_speakers
        
        # 5 Enhanced Streams
        self.s1 = SpectralStream(hidden_dim, dropout)
        self.s2 = ProsodicStream(hidden_dim, dropout)
        self.s3 = RhythmStream(hidden_dim, dropout)
        self.s4 = EnergyStream(hidden_dim, dropout)
        self.s5 = PhoneticStream(hidden_dim, dropout)
        
        # Positional encoding with dropout
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Feature fusion with layer norm
        self.fusion = nn.Linear(hidden_dim * 5, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Enhanced Transformer Encoder (6 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Multi-layer Attractor Decoder
        self.attractor_decoder = MultiLayerAttractorDecoder(
            num_speakers, hidden_dim, num_layers=3, dropout=dropout
        )
        
        # Projection heads with layer norm
        self.spk_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.frm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0 initial
        
        # Mel spectrogram transform
        self.mel_trans = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

    def forward(self, audio: torch.Tensor):
        B = audio.shape[0]
        mel = self.mel_trans(audio).unsqueeze(1).transpose(2, 3)
        
        # Extract features from all streams
        f = [self.s1(mel), self.s2(audio), self.s3(audio), self.s4(audio), self.s5(audio)]
        T = f[1].size(1)
        
        # Align temporal dimensions
        f_aligned = [
            F.interpolate(x.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2) 
            for x in f
        ]
        
        # Fuse and encode
        fused = self.fusion_norm(self.fusion(torch.cat(f_aligned, dim=-1)))
        x = self.pos_encoder(fused)
        x = self.transformer(x)
        
        # Decode attractors
        attractors = self.attractor_decoder(x)
        
        # Project for similarity computation
        att_p = F.normalize(self.spk_proj(attractors), p=2, dim=-1, eps=1e-6)
        frm_p = F.normalize(self.frm_proj(x), p=2, dim=-1, eps=1e-6)
        
        # Learnable temperature (clamped for stability)
        temperature = torch.clamp(self.log_temperature.exp(), min=0.05, max=2.0)
        
        # Compute assignments
        assignments = torch.bmm(frm_p, att_p.transpose(1, 2)) / temperature
        assignments = torch.clamp(assignments, min=-15.0, max=15.0)
        
        # Existence logits
        existence = assignments.max(dim=1)[0]
        existence = torch.clamp(existence, min=-15.0, max=15.0)
        
        # Overlap detection (multi-speaker frames)
        probs = torch.sigmoid(assignments)
        overlap_logits = (probs.sum(dim=-1) - 1).clamp(min=0)  # >1 means overlap
        
        return {
            'assignments': assignments, 
            'existence': existence, 
            'attractors': attractors,
            'overlap_logits': overlap_logits,
            'temperature': temperature
        }