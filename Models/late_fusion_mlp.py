import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionMLP(nn.Module):
    """
    A Late Fusion MLP model with separate branches for audio, video, and personalized features.
    Each branch processes its input and outputs logits, which are averaged for final prediction.
    """

    def __init__(self, audio_dim, video_dim, pers_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Initializes the LateFusionMLP model.

        Args:
            audio_dim (int): Dimensionality of audio features.
            video_dim (int): Dimensionality of video features.
            pers_dim (int): Dimensionality of personalized features.
            hidden_dim (int): Hidden layer size for each MLP branch.
            num_classes (int): Number of target classes.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self.video_branch = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self.pers_branch = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def aggregate_features(self, features):
        """
        Mean-pools over sequence length if needed.

        Args:
            features (Tensor): Input feature tensor.

        Returns:
            Tensor: Aggregated feature tensor.
        """
        if features.ndim == 3:
            return features.mean(dim=1)  # [B, D]
        elif features.ndim == 2:
            return features
        else:
            raise ValueError(f"Expected input of 2 or 3 dims, got {features.ndim}")

    def forward(self, A_feat, V_feat, P_feat):
        """
        Forward pass through all three modality-specific MLPs and performs late fusion.

        Args:
            A_feat (Tensor): Audio features [B, T, D] or [B, D].
            V_feat (Tensor): Video features [B, T, D] or [B, D].
            P_feat (Tensor): Personalized features [B, D].

        Returns:
            Tensor: Logits [B, num_classes]
        """
        A_feat_agg = self.aggregate_features(A_feat)
        V_feat_agg = self.aggregate_features(V_feat)
        if P_feat.ndim == 1:
            P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2:
            raise ValueError(f"P_feat expected 2D, got {P_feat.ndim}D")

        pred_audio = self.audio_branch(A_feat_agg)
        pred_video = self.video_branch(V_feat_agg)
        pred_pers = self.pers_branch(P_feat)

        logits = (pred_audio + pred_video + pred_pers) / 3  # Late fusion

        return logits
