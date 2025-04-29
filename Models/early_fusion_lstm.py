import torch
import torch.nn as nn

class EarlyFusionLSTM(nn.Module):
    """
    A sequential Early Fusion model using LSTM encoders for audio/video,
    and MLP for fusion and classification.
    """

    def __init__(self, audio_dim, video_dim, pers_dim, hidden_dim_lstm, hidden_dim_mlp, num_classes, lstm_layers=1, dropout_rate=0.5):
        """
        Args:
            audio_dim (int): Dimensionality of audio features.
            video_dim (int): Dimensionality of video features.
            pers_dim (int): Dimensionality of personalized features.
            hidden_dim_lstm (int): Hidden dimension of each LSTM.
            hidden_dim_mlp (int): Hidden dimension of MLP.
            num_classes (int): Number of output classes.
            lstm_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate for MLP.
        """
        super().__init__()

        self.audio_lstm = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False  # You can set True if you want
        )

        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_dim_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)

        # After LSTMs, we have:
        # [audio_hidden] + [video_hidden] + [personalized_feat]
        self.concatenated_dim = hidden_dim_lstm + hidden_dim_lstm + pers_dim

        # MLP for classification
        self.fc1 = nn.Linear(self.concatenated_dim, hidden_dim_mlp)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_mlp, num_classes)

    def forward(self, A_feat, V_feat, P_feat):
        """
        Args:
            A_feat (Tensor): Audio input, shape [batch, seq_len, audio_dim].
            V_feat (Tensor): Video input, shape [batch, seq_len, video_dim].
            P_feat (Tensor): Personalized features, shape [batch, pers_dim].

        Returns:
            logits (Tensor): [batch, num_classes]
        """
        batch_size = A_feat.size(0)

        # Pass through LSTMs
        _, (A_hidden, _) = self.audio_lstm(A_feat)  # A_hidden: [num_layers, batch, hidden_dim]
        _, (V_hidden, _) = self.video_lstm(V_feat)

        # Take the last layer's hidden state
        A_last = A_hidden[-1]  # [batch, hidden_dim]
        V_last = V_hidden[-1]

        # Ensure P_feat is 2D
        if P_feat.ndim == 1:
            P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2:
            raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")

        # Concatenate
        fusion = torch.cat((A_last, V_last, P_feat), dim=1)  # [batch, concatenated_dim]

        # Pass through MLP
        x = self.fc1(fusion)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits