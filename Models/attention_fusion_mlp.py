import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Attention Fusion MLP Model ---
class AttentionFusionMLP(nn.Module):
    """
    Processes Audio/Video/Personalized features, projects them, calculates
    attention weights, computes a weighted sum, and classifies the result.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, projection_dim, fusion_hidden_dim, num_classes, dropout_rate=0.5):
        super().__init__()
        if not all(isinstance(d, int) and d > 0 for d in [audio_dim, video_dim, pers_dim, projection_dim, fusion_hidden_dim, num_classes]):
             raise ValueError("Invalid non-positive or non-integer dimension provided to AttentionFusionMLP")
        self.audio_dim = audio_dim; self.video_dim = video_dim; self.pers_dim = pers_dim
        self.projection_dim = projection_dim; self.fusion_hidden_dim = fusion_hidden_dim
        self.num_classes = num_classes; self.dropout_rate = dropout_rate

        self.audio_proj = nn.Linear(audio_dim, projection_dim)
        self.video_proj = nn.Linear(video_dim, projection_dim)
        self.pers_proj = nn.Linear(pers_dim, projection_dim)

        att_scorer_hidden = max(projection_dim // 2, 1) # Ensure hidden dim is at least 1
        self.attention_scorer = nn.Sequential(
            nn.Linear(projection_dim, att_scorer_hidden),
            nn.Tanh(),
            nn.Linear(att_scorer_hidden, 1)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def aggregate_features(self, features):
        if features.ndim == 3: return torch.mean(features, dim=1)
        elif features.ndim == 2: return features
        else: raise ValueError(f"Unsupported feature dimension: {features.ndim}")

    def forward(self, A_feat, V_feat, P_feat):
        if not all(isinstance(t, torch.Tensor) for t in [A_feat, V_feat, P_feat]): raise TypeError("Inputs must be Tensors")
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        if P_feat.ndim != 2 or P_feat.shape[-1] != self.pers_dim: raise ValueError(f"P_feat shape {P_feat.shape} incorrect.")

        A_feat_agg = self.aggregate_features(A_feat); V_feat_agg = self.aggregate_features(V_feat)
        audio_proj = self.audio_proj(A_feat_agg); video_proj = self.video_proj(V_feat_agg); pers_proj = self.pers_proj(P_feat)
        score_a = self.attention_scorer(audio_proj); score_v = self.attention_scorer(video_proj); score_p = self.attention_scorer(pers_proj)
        all_scores = torch.stack([score_a, score_v, score_p], dim=1)
        attention_weights = F.softmax(all_scores, dim=1)
        all_proj = torch.stack([audio_proj, video_proj, pers_proj], dim=1)
        context_vector = torch.sum(attention_weights * all_proj, dim=1)
        fused_output = self.fusion_layer(context_vector)
        logits = self.classifier(fused_output)
        return logits