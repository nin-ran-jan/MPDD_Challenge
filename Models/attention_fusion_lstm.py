import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionLSTM(nn.Module):
    """
    A model using attention to fuse LSTM-encoded audio/video and MLP-encoded personalized features.
    """

    def __init__(self, audio_dim, video_dim, pers_dim, hidden_dim_lstm, hidden_dim_mlp, num_classes, lstm_layers=1, dropout_rate=0.1):
        super().__init__()

        self.hidden_dim_lstm = hidden_dim_lstm

        # Audio LSTM
        self.audio_lstm = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # Video LSTM
        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # MLP for Personalized Features
        self.mlp_pers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_mlp, hidden_dim_lstm)  # project to match LSTM output dim
        )

        # Attention module
        self.attention_query = nn.Parameter(torch.randn(hidden_dim_lstm))  # trainable query
        self.attn_linear = nn.Linear(hidden_dim_lstm, hidden_dim_lstm)
        self.output_proj = nn.Linear(hidden_dim_lstm, num_classes)

    def forward(self, A_feat, V_feat, P_feat):
        """
        Args:
            A_feat: [B, T, audio_dim]
            V_feat: [B, T, video_dim]
            P_feat: [B, pers_dim]
        Returns:
            logits: [B, num_classes]
        """
        batch_size = A_feat.size(0)

        # Audio LSTM encoding
        _, (A_hidden, _) = self.audio_lstm(A_feat)
        A_last = A_hidden[-1]  # [B, H]

        # Video LSTM encoding
        _, (V_hidden, _) = self.video_lstm(V_feat)
        V_last = V_hidden[-1]  # [B, H]

        # Personalized MLP
        if P_feat.ndim == 1:
            P_feat = P_feat.unsqueeze(0)
        P_proj = self.mlp_pers(P_feat)  # [B, H]

        # Stack all modality embeddings
        modality_stack = torch.stack([A_last, V_last, P_proj], dim=1)  # [B, 3, H]

        # Compute attention scores
        query = self.attention_query.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [B, 1, H]
        keys = self.attn_linear(modality_stack)  # [B, 3, H]
        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [B, 3]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 3]

        # Weighted sum
        fused = torch.bmm(attn_weights.unsqueeze(1), modality_stack).squeeze(1)  # [B, H]

        # Final classification
        logits = self.output_proj(fused)  # [B, num_classes]

        return logits
