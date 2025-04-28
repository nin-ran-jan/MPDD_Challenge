import torch
import torch.nn as nn

class EarlyFusionMLP(nn.Module):
    """
    An Early Fusion MLP model that aggregates sequential audio/video features,
    concatenates them with personalized features, and passes them through
    a shallow MLP for classification.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Initializes the EarlyFusionMLP model.

        Args:
            audio_dim (int): The dimensionality of the audio features (e.g., 512 for Wav2Vec).
            video_dim (int): The dimensionality of the video features (e.g., 709 for OpenFace).
            pers_dim (int): The dimensionality of the personalized features (e.g., 1024 for RoBERTa).
            hidden_dim (int): The number of neurons in the hidden layer.
            num_classes (int): The number of output classes (e.g., 2, 3, or 5).
            dropout_rate (float): The dropout probability.
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.pers_dim = pers_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Calculate the total dimension after aggregation and concatenation
        self.concatenated_dim = audio_dim + video_dim + pers_dim

        # Define the MLP layers
        # Layer 1: Input (concatenated) -> Hidden
        self.fc1 = nn.Linear(self.concatenated_dim, self.hidden_dim)

        # Activation Function
        self.relu = nn.ReLU()

        # Dropout Layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # Layer 2: Hidden -> Output (Logits)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        # Don't print during Optuna runs, it's too verbose
        # print(f"Initialized EarlyFusionMLP:")
        # print(f"  - Input Dims: Audio={audio_dim}, Video={video_dim}, Pers={pers_dim}")
        # print(f"  - Concatenated Dim: {self.concatenated_dim}")
        # print(f"  - Hidden Dim: {hidden_dim}")
        # print(f"  - Output Classes: {num_classes}")
        # print(f"  - Dropout Rate: {dropout_rate}")


    def aggregate_features(self, features):
        """
        Aggregates features over the sequence length dimension (dim=1) using mean pooling.
        Assumes input shape [batch_size, seq_len, feature_dim].
        Handles cases where input might already be aggregated ([batch_size, feature_dim]).
        """
        if features.ndim == 3:
            # Input is sequential [batch, seq_len, dim], aggregate over seq_len (dim=1)
            return torch.mean(features, dim=1)
        elif features.ndim == 2:
            # Input is already aggregated [batch, dim], return as is
            # print("Warning: Received already aggregated features.") # Optional warning
            return features
        else:
            raise ValueError(f"Unsupported feature dimension: {features.ndim}. Expected 2 or 3.")


    def forward(self, A_feat, V_feat, P_feat):
        """
        Performs the forward pass of the model.

        Args:
            A_feat (torch.Tensor): Audio features (shape: [batch, max_len, audio_dim] or [batch, audio_dim]).
            V_feat (torch.Tensor): Video features (shape: [batch, max_len, video_dim] or [batch, video_dim]).
            P_feat (torch.Tensor): Personalized features (shape: [batch, pers_dim]).

        Returns:
            torch.Tensor: Output logits (shape: [batch, num_classes]).
        """
        # 1. Aggregate Audio and Video features if they are sequential
        A_feat_agg = self.aggregate_features(A_feat) # Shape: [batch, audio_dim]
        V_feat_agg = self.aggregate_features(V_feat) # Shape: [batch, video_dim]

        # Ensure personalized features are 2D [batch, pers_dim]
        if P_feat.ndim == 1:
             # This might happen if batch size is 1 and loader doesn't keep dim
             P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2:
             raise ValueError(f"Unexpected personalized feature dimension: {P_feat.ndim}. Expected 2.")


        # 2. Concatenate aggregated features
        # Ensure dimensions match expected values before concatenating
        # Note: These checks might slow down training slightly. Remove if performance critical after debugging.
        if A_feat_agg.shape[1] != self.audio_dim:
             raise ValueError(f"Audio feature dim mismatch: expected {self.audio_dim}, got {A_feat_agg.shape[1]}")
        if V_feat_agg.shape[1] != self.video_dim:
             raise ValueError(f"Video feature dim mismatch: expected {self.video_dim}, got {V_feat_agg.shape[1]}")
        if P_feat.shape[1] != self.pers_dim:
             raise ValueError(f"Personalized feature dim mismatch: expected {self.pers_dim}, got {P_feat.shape[1]}")

        concatenated_features = torch.cat((A_feat_agg, V_feat_agg, P_feat), dim=1) # Shape: [batch, concatenated_dim]

        # 3. Pass through MLP
        x = self.fc1(concatenated_features)
        x = self.relu(x)
        x = self.dropout(x) # Apply dropout after activation, before the next layer
        logits = self.fc2(x) # Output logits

        return logits