from dataclasses import dataclass
import json
from typing import List, Optional


@dataclass
class Config:
    data_root_path: str
    window_split_time: int = 1
    audio_feature_method: str = "wav2vec"
    video_feature_method: str = "openface"
    labelcount: int = 2
    track_option: str = "Track1"
    feature_max_len: int = 26
    batch_size: int = 2
    lr: float = 0.00002
    num_epochs: int = 200
    device: str = "mps"
    cv_folds: int = 5
    seed: int = 32

    # For Early MLP
    num_epochs_tuning: int = 15
    num_epochs_final: int = 50
    optuna_trials: int = 50
    model_save_path: str = './best_early_fusion_mlp.pth'
    audio_dim: Optional[int] = None # Will be inferred
    video_dim: Optional[int] = None # Will be inferred
    pers_dim: Optional[int] = None  # Will be inferred
    num_classes: Optional[int] = None # Will be set from labelcount

    # For Early LSTM
    gamma: float = 2.0
    alpha: float = 0.25

    # for transformers
    optuna_timeout: Optional[int] = 3600  # Optional: Timeout for Optuna study in seconds (e.g., 1 hour). Use None for no timeout.
    class_weights: Optional[List[float]] = None 
    calculate_weights_dynamically: bool = True # Calculate weights from dataset if True

    @staticmethod
    def from_json(json_file_path: str) -> 'Config':
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return Config(**data)