import torch
import torch.nn as nn
import math # For PositionalEncoding

# Helper: Positional Encoding
class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Shape: [max_len, 1, d_model] -> transpose to [1, max_len, d_model] for batch_first=True
        self.register_buffer('pe', pe.transpose(0, 1)) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x shape [B, L, D], self.pe shape [1, max_L, D]
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Main Model
class TransformerLateFusion(nn.Module):
    """
    A Late Fusion model using Transformer encoders for audio and video sequences.

    1. Processes audio and video sequences independently through Transformer encoders.
    2. Aggregates the outputs of the encoders (e.g., mean pooling).
    3. Concatenates the aggregated audio, aggregated video, and static personalized features.
    4. Passes the combined features through a final MLP classifier.
    """
    def __init__(self,
                 audio_dim: int,
                 video_dim: int,
                 pers_dim: int,
                 num_classes: int,
                 transformer_embed_dim: int = 128, # Internal dimension for transformers
                 transformer_nhead: int = 4,       # Number of attention heads
                 transformer_num_layers: int = 2,  # Number of encoder layers
                 transformer_dropout: float = 0.2, # Dropout within transformer blocks
                 mlp_hidden_dim: int = 256,        # Hidden dim for the final MLP
                 mlp_dropout: float = 0.5,         # Dropout for the final MLP
                 max_seq_len: int = 10             # Max sequence length for positional encoding
                 ):
        """
        Initializes the TransformerLateFusion model.

        Args:
            audio_dim (int): Input dimension of audio features.
            video_dim (int): Input dimension of video features.
            pers_dim (int): Input dimension of personalized features.
            num_classes (int): Number of output classes.
            transformer_embed_dim (int): Embedding dimension used inside transformers. Must be divisible by nhead.
            transformer_nhead (int): Number of attention heads in transformers.
            transformer_num_layers (int): Number of layers in each transformer encoder.
            transformer_dropout (float): Dropout rate within transformer encoder layers.
            mlp_hidden_dim (int): Hidden dimension of the final MLP classifier.
            mlp_dropout (float): Dropout rate for the final MLP classifier.
            max_seq_len (int): Maximum sequence length anticipated for positional encoding.
        """
        super().__init__()

        if transformer_embed_dim % transformer_nhead != 0:
             raise ValueError(f"transformer_embed_dim ({transformer_embed_dim}) must be divisible by "
                              f"transformer_nhead ({transformer_nhead})")

        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.pers_dim = pers_dim
        self.transformer_embed_dim = transformer_embed_dim

        # --- Input Projection Layers (Optional but recommended) ---
        self.audio_proj = nn.Linear(audio_dim, transformer_embed_dim)
        self.video_proj = nn.Linear(video_dim, transformer_embed_dim)

        # --- Positional Encodings ---
        self.pos_encoder = PositionalEncoding(transformer_embed_dim, transformer_dropout, max_seq_len)

        # --- Audio Transformer Encoder ---
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_embed_dim * 4, # Standard practice
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True # Important: expects [batch, seq, feature]
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            audio_encoder_layer,
            num_layers=transformer_num_layers
        )

        # --- Video Transformer Encoder ---
        video_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_embed_dim * 4,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True # Important: expects [batch, seq, feature]
        )
        self.video_transformer_encoder = nn.TransformerEncoder(
            video_encoder_layer,
            num_layers=transformer_num_layers
        )

        # --- Final MLP Classifier ---
        # Input dimension is aggregated audio_embed + aggregated video_embed + personalized_dim
        self.mlp_input_dim = transformer_embed_dim + transformer_embed_dim + pers_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.mlp_input_dim), # LayerNorm before MLP often helps
            nn.Linear(self.mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

        # Initialize weights (optional but can help)
        self._init_weights()

        print(f"Initialized TransformerLateFusion:")
        print(f"  - Input Dims: Audio={audio_dim}, Video={video_dim}, Pers={pers_dim}")
        print(f"  - Transformer: Embed={transformer_embed_dim}, Heads={transformer_nhead}, Layers={transformer_num_layers}, Dropout={transformer_dropout:.2f}")
        print(f"  - Final MLP: Hidden={mlp_hidden_dim}, Dropout={mlp_dropout:.2f}")
        print(f"  - Output Classes: {num_classes}")


    def _init_weights(self):
        # Initialize transformer and MLP weights
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def aggregate_features(self, features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        Aggregates features over the sequence length dimension (dim=1).
        Assumes input shape [batch_size, seq_len, feature_dim].

        Args:
            features (torch.Tensor): Features to aggregate.
            method (str): Aggregation method ('mean' or 'cls'). 'cls' assumes first token is CLS.

        Returns:
            torch.Tensor: Aggregated features of shape [batch_size, feature_dim].
        """
        if features.ndim != 3:
            raise ValueError(f"Unsupported feature dimension: {features.ndim}. Expected 3 ([B, L, D]).")

        if method == 'mean':
            return torch.mean(features, dim=1)
        elif method == 'cls':
            return features[:, 0, :] # Return the first token (assumed CLS)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


    def forward(self, A_feat: torch.Tensor, V_feat: torch.Tensor, P_feat: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            A_feat (torch.Tensor): Audio features (shape: [batch, max_len, audio_dim]).
            V_feat (torch.Tensor): Video features (shape: [batch, max_len, video_dim]).
            P_feat (torch.Tensor): Personalized features (shape: [batch, pers_dim]).

        Returns:
            torch.Tensor: Output logits (shape: [batch, num_classes]).
        """
        # 0. Input Shape Checks (Optional but helpful for debugging)
        # print(f"A_feat shape: {A_feat.shape}") # [B, L, Da]
        # print(f"V_feat shape: {V_feat.shape}") # [B, L, Dv]
        # print(f"P_feat shape: {P_feat.shape}") # [B, Dp]
        if A_feat.ndim != 3 or V_feat.ndim != 3:
             raise ValueError("Audio and Video features must be 3D [Batch, SeqLen, Dim]")
        if P_feat.ndim == 1: # Handle potential batch size 1 edge case
             P_feat = P_feat.unsqueeze(0)
        if P_feat.ndim != 2:
             raise ValueError("Personalized features must be 2D [Batch, Dim]")

        # 1. Project Audio and Video features to transformer embedding dimension
        A_proj = self.audio_proj(A_feat) # Shape: [B, L, E]
        V_proj = self.video_proj(V_feat) # Shape: [B, L, E]

        # 2. Add Positional Encoding
        A_in = self.pos_encoder(A_proj) # Shape: [B, L, E]
        V_in = self.pos_encoder(V_proj) # Shape: [B, L, E]

        # 3. Pass through Transformer Encoders
        # TransformerEncoder expects [B, L, E] if batch_first=True
        A_encoded_seq = self.audio_transformer_encoder(A_in) # Shape: [B, L, E]
        V_encoded_seq = self.video_transformer_encoder(V_in) # Shape: [B, L, E]

        # 4. Aggregate sequence outputs
        A_agg = self.aggregate_features(A_encoded_seq, method='mean') # Shape: [B, E]
        V_agg = self.aggregate_features(V_encoded_seq, method='mean') # Shape: [B, E]

        # 5. Concatenate aggregated features with personalized features
        # Ensure dimensions match expected values before concatenating
        if A_agg.shape[1] != self.transformer_embed_dim or V_agg.shape[1] != self.transformer_embed_dim:
            raise ValueError("Aggregated feature dimension mismatch.")
        if P_feat.shape[1] != self.pers_dim:
             raise ValueError(f"Personalized feature dim mismatch: expected {self.pers_dim}, got {P_feat.shape[1]}")

        combined_features = torch.cat((A_agg, V_agg, P_feat), dim=1) # Shape: [B, E + E + Dp]

        # 6. Pass through final MLP classifier
        logits = self.mlp_head(combined_features) # Shape: [B, num_classes]

        return logits