import torch
import torch.nn as nn
import math
import torch.nn.functional as F # For GELU if needed, though TransformerEncoderLayer has it

# PositionalEncoding remains the same as before
class PositionalEncoding(nn.Module):
    # ... (Keep the previous implementation) ...
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
             # Handle odd d_model dimension if necessary, though usually even
             pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1] # Adjust size if odd
        else:
             pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = pe.permute(1, 0, 2) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # Add positional encoding scaled to the sequence length
        # Note: If using CLS token, x might be seq_len+1. Ensure pe is sliced correctly.
        # The PE is added starting from the first *actual* sequence token.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LateFusionTransformer(nn.Module):
    """
    Late Fusion Transformer using CLS token aggregation and concatenation fusion.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, embed_dim, nhead, num_encoder_layers,
                 dim_feedforward, hidden_dim_mlp, num_classes, max_len,
                 fusion_hidden_dim, dropout_rate=0.1):
        super().__init__()

        # ** NEW: Learnable CLS tokens **
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token_audio, std=0.02) # Initialize CLS tokens
        nn.init.normal_(self.cls_token_video, std=0.02)

        # Adjust max_len for positional encoding if CLS token is added
        # The PE module needs to handle max_len + 1 positions potentially
        self.pos_encoder = PositionalEncoding(embed_dim, dropout_rate, max_len + 1) # +1 for CLS

        # --- Input Projection ---
        self.audio_projection = nn.Linear(audio_dim, embed_dim)
        self.video_projection = nn.Linear(video_dim, embed_dim)

        # --- Transformer Encoder Layer (Use GELU) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            # *** CHANGE: Use GELU activation ***
            activation=F.gelu, # Or 'gelu' string if PyTorch version supports it
            batch_first=True,
            norm_first=False # Keep as Post-LN for now unless specifically trying Pre-LN
        )

        # --- Transformer Encoders ---
        self.audio_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.video_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- MLP for Personalized Features ---
        self.mlp_pers_layers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(), # Keep ReLU here for simplicity
            nn.Dropout(dropout_rate),
        )

        # --- Fusion MLP ---
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim + embed_dim + hidden_dim_mlp), # Add LayerNorm before fusion MLP
            nn.Linear(embed_dim + embed_dim + hidden_dim_mlp, fusion_hidden_dim),
            nn.GELU(), # Use GELU here too
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

        # --- Layer Normalization ---
        # LayerNorm is applied within TransformerEncoderLayer and before Fusion MLP
        # Keep these separate LayerNorms for input to transformer if desired (optional pre-norm)
        # self.layer_norm_audio_input = nn.LayerNorm(embed_dim) # Optional LN after projection
        # self.layer_norm_video_input = nn.LayerNorm(embed_dim)

    def forward(self, A_feat, V_feat, P_feat):
        batch_size = A_feat.size(0)

        # --- Audio Branch ---
        A_proj = self.audio_projection(A_feat) # [batch, seq_len, embed_dim]
        # Prepend CLS token
        cls_audio_expanded = self.cls_token_audio.expand(batch_size, -1, -1) # [batch, 1, embed_dim]
        A_with_cls = torch.cat((cls_audio_expanded, A_proj), dim=1) # [batch, seq_len+1, embed_dim]
        # Add positional encoding
        A_enc = self.pos_encoder(A_with_cls)
        # Pass through Transformer
        A_out = self.audio_transformer_encoder(A_enc) # [batch, seq_len+1, embed_dim]
        # ** NEW: Aggregate using CLS token output **
        A_agg = A_out[:, 0] # [batch, embed_dim] (Select output of first token)

        # --- Video Branch ---
        V_proj = self.video_projection(V_feat)
        # Prepend CLS token
        cls_video_expanded = self.cls_token_video.expand(batch_size, -1, -1)
        V_with_cls = torch.cat((cls_video_expanded, V_proj), dim=1)
        # Add positional encoding
        V_enc = self.pos_encoder(V_with_cls)
        # Pass through Transformer
        V_out = self.video_transformer_encoder(V_enc)
        # ** NEW: Aggregate using CLS token output **
        V_agg = V_out[:, 0] # [batch, embed_dim]

        # --- Personalized Branch ---
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")
        P_proc = self.mlp_pers_layers(P_feat) # [batch, hidden_dim_mlp]

        # --- Concatenation Fusion ---
        combined_features = torch.cat((A_agg, V_agg, P_proc), dim=1)
        # Expected shape: [batch, embed_dim + embed_dim + hidden_dim_mlp]

        # Pass through fusion MLP
        logits = self.fusion_mlp(combined_features) # [batch, num_classes]

        return logits