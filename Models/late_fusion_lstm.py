import torch
import torch.nn as nn

class LateFusionLSTM(nn.Module):
    """
    A Late Fusion model using independent LSTM encoders for audio/video,
    and an MLP for personalized features, with fusion at the prediction level.
    """

    def __init__(self, audio_dim, video_dim, pers_dim, hidden_dim_lstm, hidden_dim_mlp, num_classes, lstm_layers=1, dropout_rate=0.1):
        super().__init__()

        # LSTM for Audio
        self.audio_lstm = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc_audio = nn.Linear(hidden_dim_lstm, num_classes)

        # LSTM for Video
        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc_video = nn.Linear(hidden_dim_lstm, num_classes)

        # MLP for Personalized Features
        self.mlp_pers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_mlp, num_classes)
        )

    def forward(self, A_feat, V_feat, P_feat):
        """
        Args:
            A_feat: Audio features [batch, seq_len, audio_dim]
            V_feat: Video features [batch, seq_len, video_dim]
            P_feat: Personalized features [batch, pers_dim]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = A_feat.size(0)

        # Audio branch
        _, (A_hidden, _) = self.audio_lstm(A_feat)
        A_last = A_hidden[-1]  # [batch, hidden_dim]
        pred_audio = self.fc_audio(A_last)

        # Video branch
        _, (V_hidden, _) = self.video_lstm(V_feat)
        V_last = V_hidden[-1]
        pred_video = self.fc_video(V_last)

        # Personalized branch
        if P_feat.ndim == 1:
            P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2:
            raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")
        pred_pers = self.mlp_pers(P_feat)

        # Late fusion: simple average
        logits = (pred_audio + pred_video + pred_pers) / 3

        return logits
