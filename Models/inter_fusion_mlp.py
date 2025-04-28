import torch
import torch.nn as nn

# --- Intermediate Fusion MLP Model ---
class IntermediateFusionMLP(nn.Module):
    """
    Processes Audio/Video via separate encoders, then fuses with Personalized
    features before final classification.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, modality_hidden_dim, fusion_hidden_dim, num_classes, dropout_rate=0.5):
        """
        Args:
            audio_dim (int): Input audio feature dimension.
            video_dim (int): Input video feature dimension.
            pers_dim (int): Input personalized feature dimension.
            modality_hidden_dim (int): Output dimension of the individual modality encoders.
            fusion_hidden_dim (int): Dimension of the hidden layer after fusion.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        if not all(isinstance(d, int) and d > 0 for d in [audio_dim, video_dim, pers_dim, modality_hidden_dim, fusion_hidden_dim, num_classes]):
             raise ValueError("Invalid non-positive or non-integer dimension provided to IntermediateFusionMLP")

        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.pers_dim = pers_dim
        self.modality_hidden_dim = modality_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # --- Modality Encoders ---
        # Audio Encoder: Aggregated Audio -> Modality Hidden Dim
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, modality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Video Encoder: Aggregated Video -> Modality Hidden Dim
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, modality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Personalized Feature Encoder (optional, could just be identity or linear)
        # Using a simple linear layer for consistency and potential dimension matching
        self.pers_encoder = nn.Sequential(
            nn.Linear(pers_dim, modality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # --- Fusion Layer ---
        # Concatenated dim = 3 * modality_hidden_dim
        self.concatenated_dim = 3 * modality_hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.concatenated_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # --- Classification Head ---
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def aggregate_features(self, features):
        """Aggregates features over the sequence length dimension (dim=1)"""
        if features.ndim == 3: return torch.mean(features, dim=1)
        elif features.ndim == 2: return features # Already aggregated
        else: raise ValueError(f"Unsupported feature dimension: {features.ndim}")

    def forward(self, A_feat, V_feat, P_feat):
        """
        Args:
            A_feat (torch.Tensor): Audio features [Batch, SeqLen, AudioDim]
            V_feat (torch.Tensor): Video features [Batch, SeqLen, VideoDim]
            P_feat (torch.Tensor): Personalized features [Batch, PersDim]
        """
        # 1. Aggregate Audio/Video Features
        A_feat_agg = self.aggregate_features(A_feat) # -> [Batch, AudioDim]
        V_feat_agg = self.aggregate_features(V_feat) # -> [Batch, VideoDim]

        # Ensure P_feat is 2D [Batch, PersDim]
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        if P_feat.ndim != 2 or P_feat.shape[-1] != self.pers_dim:
             raise ValueError(f"P_feat shape {P_feat.shape} incorrect. Expected [B, {self.pers_dim}]")

        # 2. Pass through Modality Encoders
        audio_encoded = self.audio_encoder(A_feat_agg) # -> [Batch, ModalityHiddenDim]
        video_encoded = self.video_encoder(V_feat_agg) # -> [Batch, ModalityHiddenDim]
        pers_encoded = self.pers_encoder(P_feat)       # -> [Batch, ModalityHiddenDim]

        # 3. Concatenate Encoded Representations
        fused_features = torch.cat((audio_encoded, video_encoded, pers_encoded), dim=1) # -> [Batch, 3 * ModalityHiddenDim]

        # 4. Pass through Fusion Layer
        fused_output = self.fusion_layer(fused_features) # -> [Batch, FusionHiddenDim]

        # 5. Pass through Classifier
        logits = self.classifier(fused_output) # -> [Batch, NumClasses]

        return logits