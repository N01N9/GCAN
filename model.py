import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from itertools import permutations

# ==============================================================================
# 1. Bidirectional Mamba Wrapper (New!)
# ==============================================================================

class BiMamba(nn.Module):
    """
    Mamba2를 양방향(Bidirectional)으로 동작하게 만드는 래퍼입니다.
    순방향(Forward)과 역방향(Backward)의 결과를 합쳐 문맥 정보를 강화합니다.
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        # Forward direction
        self.fwd_mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # Backward direction
        self.bwd_mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 결과를 합친 후 차원 축소 (Concat -> Linear) 대신
        # 여기서는 메모리 효율을 위해 단순히 더하는(Add) 방식을 사용하거나
        # Linear로 섞어줍니다. 여기서는 Linear Fusion을 사용합니다.
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        
        # 1. Forward Pass
        out_fwd = self.fwd_mamba(x)
        
        # 2. Backward Pass (Flip sequence -> Mamba -> Flip back)
        out_bwd = self.bwd_mamba(x.flip(dims=[1])).flip(dims=[1])
        
        # 3. Fusion
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        return self.fusion(out)

# ==============================================================================
# 2. Grid-Mamba Block (Improved)
# ==============================================================================

class GridMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dilation=1):
        super().__init__()
        
        # 1. Frequency Axis (Bi-Mamba)
        self.freq_mamba = BiMamba(d_model, d_state, d_conv, expand)
        self.freq_norm = nn.LayerNorm(d_model)
        self.dilation = dilation

        # 2. Time Axis (Bi-Mamba)
        self.time_mamba = BiMamba(d_model, d_state, d_conv, expand)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Input: [B, C, T, F]
        """
        B, C, T, F = x.shape
        
        # --- Path 1: Frequency Axis (Harmonic Scan with Dilation) ---
        # [B, C, T, F] -> [B*T, F, C]
        x_freq = x.permute(0, 2, 3, 1).contiguous().view(B*T, F, C)
        
        # Dilation Trick
        if self.dilation > 1:
            pad = (self.dilation - (F % self.dilation)) % self.dilation
            if pad > 0:
                x_freq = torch.nn.functional.pad(x_freq, (0, 0, 0, pad)) # F 차원 패딩
            F_pad = x_freq.shape[1]
            
            # Reshape for dilation: [BT, F//D, D, C] -> [BT*D, F//D, C]
            x_freq = x_freq.view(B*T, F_pad // self.dilation, self.dilation, C)
            x_freq = x_freq.permute(0, 2, 1, 3).contiguous().view(B*T*self.dilation, F_pad // self.dilation, C)
            
        x_freq_out = self.freq_mamba(self.freq_norm(x_freq))
        
        # Dilation Restore
        if self.dilation > 1:
            x_freq_out = x_freq_out.view(B*T, self.dilation, -1, C).permute(0, 2, 1, 3).contiguous()
            x_freq_out = x_freq_out.view(B*T, F_pad, C)
            x_freq_out = x_freq_out[:, :F, :] # Remove padding
            
        # Residual Connection
        # [BT, F, C] -> [B, T, F, C] -> [B, C, T, F]
        x = x + x_freq_out.view(B, T, F, C).permute(0, 3, 1, 2)

        # --- Path 2: Time Axis (Context Scan) ---
        # [B, C, T, F] -> [B*F, T, C]
        x_time = x.permute(0, 3, 2, 1).contiguous().view(B*F, T, C)
        x_time = self.time_mamba(self.time_norm(x_time))
        
        # Residual Connection
        x = x + x_time.view(B, F, T, C).permute(0, 3, 2, 1)

        return x

# ==============================================================================
# 3. Refined HR-GridMamba Main Model
# ==============================================================================

class HR_GridMamba(nn.Module):
    def __init__(
        self, 
        n_srcs=2,           # 화자 수
        n_fft=256,          # FFT 크기 (보통 256 or 512)
        stride=128,         # Hop length
        d_model=128,        # 내부 채널 수
        n_layers=6,         # 블록 수
        dropout=0.1
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.kernel_size = n_fft
        self.stride = stride
        self.d_model = d_model
        
        # 1. Complex Encoder (STFT -> Embed)
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, d_model, kernel_size=1),
            nn.GroupNorm(4, d_model),
            nn.PReLU()
        )

        # 2. Grid Blocks with Dilation Cycle
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            # Dilation: 1 -> 2 -> 4 -> 1 ...
            dilation = 2 ** (i % 3) 
            self.layers.append(
                GridMambaBlock(d_model=d_model, dilation=dilation)
            )

        # 3. Mask Generator (Complex Masking)
        # [B, C, T, F] -> [B, n_srcs * 2, T, F] (2 for Real/Imag)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.PReLU(),
            nn.Conv2d(d_model, n_srcs * 2, 1) 
        )

    def forward(self, x):
        """
        x: [Batch, Time] (Raw Waveform)
        """
        # 1. STFT
        # Return: [B, F, T] (Complex)
        X_stft = torch.stft(
            x, 
            n_fft=self.kernel_size, 
            hop_length=self.stride, 
            window=torch.hann_window(self.kernel_size).to(x.device),
            return_complex=True
        )
        
        # [B, F, T] -> [B, 2, F, T] (Real/Imag)
        X_stft_view = torch.view_as_real(X_stft).permute(0, 3, 2, 1) 
        
        # 2. Embed: [B, 2, F, T] -> [B, C, T, F]
        fea = self.input_conv(X_stft_view.permute(0, 1, 3, 2))
        
        # 3. Processing
        for layer in self.layers:
            fea = layer(fea)
            
        # 4. Mask Estimation
        # Output: [B, n_srcs*2, T, F]
        mask_out = self.mask_conv(fea)
        
        # Reshape: [B, n_srcs, 2, T, F]
        B, _, T, F = mask_out.shape
        mask_out = mask_out.view(B, self.n_srcs, 2, T, F)
        
        # --- [CRITICAL FIX] Complex Masking ---
        # 실수부(Real)와 허수부(Imag) 마스크를 각각 추출
        mask_real = mask_out[:, :, 0] # [B, K, T, F]
        mask_imag = mask_out[:, :, 1] # [B, K, T, F]
        
        # 복소수 마스크 생성 (Complex Tensor)
        # SOTA 모델들(DCCRN 등)에서 사용하는 Unbounded Complex Masking
        # 필요하다면 tanh로 범위를 제한할 수도 있음: torch.tanh(mask_real)
        complex_mask = torch.complex(mask_real, mask_imag) # [B, K, T, F]
        
        # Freq, Time 축 순서 원복: [B, K, F, T]
        complex_mask = complex_mask.permute(0, 1, 3, 2)
        
        # 5. Apply Mask & ISTFT
        separated_audios = []
        
        # Input STFT: [B, F, T] -> [B, 1, F, T] for broadcasting
        X_stft_expanded = X_stft.unsqueeze(1) 
        
        # Complex Multiplication: (Input) * (Mask)
        # [B, 1, F, T] * [B, K, F, T] -> [B, K, F, T]
        est_stft = X_stft_expanded * complex_mask
        
        for i in range(self.n_srcs):
            # iSTFT
            source_time = torch.istft(
                est_stft[:, i], 
                n_fft=self.kernel_size, 
                hop_length=self.stride, 
                window=torch.hann_window(self.kernel_size).to(x.device),
                length=x.shape[-1]
            )
            separated_audios.append(source_time)
            
        # [Batch, n_srcs, Time]
        return torch.stack(separated_audios, dim=1)

# ==============================================================================
# 4. Loss Functions (Unchanged but verified)
# ==============================================================================

def si_snr(preds, targets):
    eps = 1e-8
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    targets = targets - torch.mean(targets, dim=-1, keepdim=True)
    
    t_energy = torch.sum(targets ** 2, dim=-1, keepdim=True) + eps
    projection = torch.sum(targets * preds, dim=-1, keepdim=True) * targets / t_energy
    
    noise = preds - projection
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)

def pit_loss(preds, targets, n_srcs=2):
    """
    PIT Loss with automatic padding for fewer active speakers
    """
    B, _, T = preds.shape
    num_targets = targets.shape[1]
    
    if num_targets < n_srcs:
        pad = torch.zeros((B, n_srcs - num_targets, T), device=targets.device)
        targets = torch.cat([targets, pad], dim=1)
        
    perms = list(permutations(range(n_srcs)))
    loss_candidates = []
    
    for p in perms:
        snr = si_snr(preds[:, p, :].reshape(-1, T), targets.reshape(-1, T))
        loss_candidates.append(-torch.mean(snr))
        
    loss_candidates = torch.stack(loss_candidates)
    min_loss, _ = torch.min(loss_candidates, dim=0)
    return min_loss

# ==============================================================================
# Run Test
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    # 모델 생성
    model = HR_GridMamba(n_srcs=4, d_model=256, n_layers=6).to(device)
    
    # 1.6초 길이의 오디오 (16000sr * 1.6 = 25600)
    dummy_input = torch.randn(2, 25600).to(device)
    
    # Forward
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: [2, 2, 25600]
    
    # Loss Calculation
    dummy_target = torch.randn(2, 2, 25600).to(device)
    loss = pit_loss(output, dummy_target, n_srcs=2)
    print(f"PIT Loss: {loss.item()}")