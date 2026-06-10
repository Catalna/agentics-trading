import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
#  1.  Trading LSTM  (improved)
# ─────────────────────────────────────────────
class TradingLSTM(nn.Module):
    """
    Attention-LSTM with LayerNorm for stable training on financial data.
    Input shape: (batch, seq_len, features)
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 96,
                 num_layers: int = 2, dropout: float = 0.4):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False  # prevent look-ahead bias
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Additive attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)                        # (B, T, H)
        lstm_out = self.layer_norm(lstm_out)
        attn_w   = self.attention(lstm_out)                # (B, T, 1)
        context  = (lstm_out * attn_w).sum(dim=1)          # (B, H)
        return self.classifier(context)


# ─────────────────────────────────────────────
#  2.  Trading CNN1D  (residual connections)
# ─────────────────────────────────────────────
class _ResBlock(nn.Module):
    """1D Residual block with optional channel projection."""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return self.drop(self.act(self.conv(x) + self.skip(x)))


class TradingCNN1D(nn.Module):
    """
    1D ResNet-style CNN for detecting chart patterns.
    Input shape: (batch, features, seq_len)
    """
    def __init__(self, in_channels: int = 12, seq_len: int = 60):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.blocks = nn.Sequential(
            _ResBlock(64, 64,  kernel=3),
            nn.MaxPool1d(2),
            _ResBlock(64, 128, kernel=5),
            nn.MaxPool1d(2),
            _ResBlock(128, 128, kernel=5),
            _ResBlock(128, 256, kernel=7),
            nn.AdaptiveAvgPool1d(1),          # → (B, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  3.  Trading Transformer  (NEW)
# ─────────────────────────────────────────────
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):                              # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class TradingTransformer(nn.Module):
    """
    Lightweight Transformer encoder for financial time-series.
    - Multi-Head Self-Attention (causal mask optional, disabled for efficiency)
    - Input shape: (batch, seq_len, features)
    """
    def __init__(self, input_size: int = 12, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True         # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # aggregate over time dim

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):                       # x: (B, T, F)
        x = self.input_proj(x)                  # → (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                 # → (B, T, d_model)
        x = self.pool(x.permute(0, 2, 1))       # → (B, d_model, 1)
        x = x.squeeze(-1)                        # → (B, d_model)
        return self.classifier(x)
