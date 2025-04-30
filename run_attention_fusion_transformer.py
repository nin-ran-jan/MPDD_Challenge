# -*- coding: utf-8 -*-
"""
Training Script for Late Fusion Transformer Model
- Uses ATTENTION-BASED FUSION via a fusion transformer layer.
- Uses CLS Token Aggregation for Audio/Video streams.
- All parameters are defined within this script.
- Uses WeightedRandomSampler for handling class imbalance.
- Uses Warmup + Cosine LR Scheduling.
- Saves the best model based on Validation Macro F1-score.
- Implements Early Stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import numpy as np
import os
import json
import time
import traceback
import math
import torch.nn.functional as F # For GELU etc.

# External Imports (Ensure these exist in your project structure)
from sklearn.metrics import accuracy_score, classification_report, f1_score
try:
    # Assuming these paths are correct relative to your script's location
    from DataSets.audioVisualDataset import AudioVisualDataset
    # Model defined below
    from Utils.focal_loss import FocalLoss
    from Utils.test_val_split import train_val_split1, train_val_split2
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure DataSets, Utils directories are correctly structured and importable.")
    exit(1)

# Optional: torchinfo for model summary
try:
    import torchinfo
except ImportError:
    torchinfo = None
    print("torchinfo not found. Install it (`pip install torchinfo`) for model summaries.")


# --- Model Definition (Attention Fusion Version) ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 2 != 0: print("Warning: d_model in PositionalEncoding is odd.")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model dimension carefully when assigning cosine
        if d_model % 2 != 0: pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:pe[:, 0, 1::2].size(1)]
        else: pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LateFusionAttentionTransformer(nn.Module):
    """
    Late Fusion Transformer using attention-based fusion.
    Processes modalities, aggregates via CLS token, then fuses using another transformer layer.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, embed_dim, nhead, num_encoder_layers,
                 dim_feedforward, hidden_dim_mlp, num_classes, max_len,
                 fusion_n_layers, # Number of layers in fusion transformer
                 fusion_nhead,    # Number of heads in fusion transformer
                 dropout_rate=0.1):
        super().__init__()

        # --- Modality Specific CLS Tokens & Encoders ---
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.cls_token_audio, std=0.02)
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.cls_token_video, std=0.02)
        # Positional encoding for sequences including the CLS token
        self.pos_encoder = PositionalEncoding(embed_dim, dropout_rate, max_len + 1)

        self.audio_projection = nn.Linear(audio_dim, embed_dim)
        self.video_projection = nn.Linear(video_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, activation=F.gelu, batch_first=True, norm_first=False
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.video_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Personalized Feature Branch ---
        self.mlp_pers_layers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Project personalized features to embed_dim for fusion
        self.pers_projection = nn.Linear(hidden_dim_mlp, embed_dim)

        # --- Attention Fusion Components ---
        self.fusion_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.fusion_cls_token, std=0.02)
        # Fusion transformer layer(s)
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=fusion_nhead, dim_feedforward=dim_feedforward, # Reusing dim_feedforward
            dropout=dropout_rate, activation=F.gelu, batch_first=True, norm_first=False
        )
        self.fusion_transformer_encoder = nn.TransformerEncoder(fusion_encoder_layer, num_layers=fusion_n_layers)

        # --- Final Classification Head ---
        # Takes the output of the fusion_cls_token
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )


    def forward(self, A_feat, V_feat, P_feat):
        batch_size = A_feat.size(0)

        # --- Modality Encoding ---
        # Audio
        A_proj = self.audio_projection(A_feat)
        cls_audio_expanded = self.cls_token_audio.expand(batch_size, -1, -1)
        A_with_cls = torch.cat((cls_audio_expanded, A_proj), dim=1)
        A_enc = self.pos_encoder(A_with_cls)
        A_out = self.audio_transformer_encoder(A_enc)
        A_agg = A_out[:, 0] # CLS token output [batch, embed_dim]

        # Video
        V_proj = self.video_projection(V_feat)
        cls_video_expanded = self.cls_token_video.expand(batch_size, -1, -1)
        V_with_cls = torch.cat((cls_video_expanded, V_proj), dim=1)
        V_enc = self.pos_encoder(V_with_cls)
        V_out = self.video_transformer_encoder(V_enc)
        V_agg = V_out[:, 0] # CLS token output [batch, embed_dim]

        # Personalized
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")
        P_proc = self.mlp_pers_layers(P_feat)
        P_agg = self.pers_projection(P_proc) # Project to embed_dim [batch, embed_dim]

        # --- Attention Fusion ---
        # Prepare sequence: [FUSION_CLS, A_agg, V_agg, P_agg]
        fusion_cls_expanded = self.fusion_cls_token.expand(batch_size, -1, -1) # [B, 1, D]
        # Reshape modality features to be sequence elements: [B, 1, D]
        A_agg_seq = A_agg.unsqueeze(1)
        V_agg_seq = V_agg.unsqueeze(1)
        P_agg_seq = P_agg.unsqueeze(1)

        # Concatenate along sequence dimension (dim=1)
        fusion_input_seq = torch.cat([fusion_cls_expanded, A_agg_seq, V_agg_seq, P_agg_seq], dim=1) # Shape: [B, 4, D]

        # Pass through fusion transformer
        # No mask needed here as sequence length is fixed (4)
        fusion_out = self.fusion_transformer_encoder(fusion_input_seq) # Shape: [B, 4, D]

        # Get the output corresponding to the FUSION_CLS token (first position)
        fused_representation = fusion_out[:, 0] # Shape: [B, D]

        # --- Final Classification ---
        logits = self.output_mlp(fused_representation) # Shape: [B, num_classes]

        return logits


# --- Training and Evaluation Functions ---
# (Keep the train_epoch and evaluate functions from the previous iteration,
# including gradient clipping and robust report generation in evaluate)
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm=1.0, scheduler=None):
    """Trains the model for one epoch with gradient clipping and optional LR step."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0

    processed_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
            continue
        try:
            audio_feat = batch['A_feat'].to(device, non_blocking=True)
            video_feat = batch['V_feat'].to(device, non_blocking=True)
            pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
            labels = batch['emo_label'].to(device, non_blocking=True)

            batch_size = labels.size(0)
            if batch_size == 0: continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(audio_feat, video_feat, pers_feat)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN or Inf loss encountered in train batch {batch_idx}. Skipping backward/step.")
                continue

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()
            if scheduler is not None:
                 # Check if it's a step-based scheduler (LambdaLR is step-based)
                 if not hasattr(scheduler, 'is_epoch_scheduler') or not scheduler.is_epoch_scheduler:
                      scheduler.step()

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
            processed_batches += 1

        except Exception as e:
            print(f"    ERROR during train batch {batch_idx}: {e}")
            # traceback.print_exc()
            continue

    if num_samples == 0: return 0.0, 0.0
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0, {}

    processed_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
                continue
            try:
                audio_feat = batch['A_feat'].to(device, non_blocking=True)
                video_feat = batch['V_feat'].to(device, non_blocking=True)
                pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
                labels = batch['emo_label'].to(device, non_blocking=True)

                batch_size = labels.size(0)
                if batch_size == 0: continue

                outputs = model(audio_feat, video_feat, pers_feat)
                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    WARNING: NaN or Inf loss encountered in eval batch {batch_idx}. Contribution ignored.")
                    continue

                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
                processed_batches += 1
            except Exception as e:
                print(f"    ERROR during eval batch {batch_idx}: {e}")
                # traceback.print_exc()
                continue

    if num_samples == 0: return 0.0, 0.0, {}
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    try:
        report = classification_report(
            all_labels,
            all_preds,
            zero_division=0,
            output_dict=True,
            labels=list(range(num_classes)),
            target_names=[f'Class_{i}' for i in range(num_classes)]
        )
    except Exception as e:
         print(f"    Error generating classification report: {e}")
         report = {}

    return avg_loss, accuracy, report


# --- Learning Rate Scheduler Function ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Creates cosine LR schedule with linear warmup. """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
    scheduler.is_epoch_scheduler = False # Mark as step-based
    return scheduler


# --- Main Execution Block ---
if __name__ == '__main__':

    # ==========================================================================
    # ==                         PARAMETER DEFINITIONS                        ==
    # ==========================================================================
    print(f"--- Script Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("--- Defining Experiment Parameters (Attention Fusion) ---")

    # --- Paths ---
    DATA_ROOT_PATH = "/Users/kaushaldamania/deepl/MPDD-Elderly"
    MODEL_SAVE_PATH = './best_attn_fusion_transformer.pth' # New name

    # --- Data & Feature Settings ---
    WINDOW_SPLIT_TIME = 5
    AUDIO_FEATURE_METHOD = "wav2vec"
    VIDEO_FEATURE_METHOD = "openface"
    FEATURE_MAX_LEN = 26 # Max length of INPUT sequence (before modality CLS token)
    PERS_DIM = 1024

    # --- Classification & Splitting ---
    LABEL_COUNT = 2
    TRACK_OPTION = "Track1"
    VAL_RATIO = 0.1
    VAL_PERCENTAGE = 0.1
    SEED = 32

    # --- Model Architecture (Attention Fusion) ---
    EMBED_DIM = 128
    N_HEAD = 4                 # Heads for modality encoders
    NUM_ENCODER_LAYERS = 2     # Layers for modality encoders
    DIM_FEEDFORWARD = 512
    HIDDEN_DIM_MLP = 256       # MLP for personalized features
    # ** NEW: Fusion Transformer Params **
    FUSION_N_LAYERS = 1        # Layers for the fusion transformer (Start with 1 or 2)
    FUSION_NHEAD = 4           # Heads for the fusion transformer (Can match N_HEAD or differ)
    # FUSION_HIDDEN_DIM removed (handled by transformer layer)
    DROPOUT_RATE = 0.4         # ** Adjusted Dropout ** (Try 0.2, 0.3, 0.4)

    # --- Training & Optimization ---
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-4        # Keep previously successful value or tune (e.g., 5e-5, 2e-4)
    NUM_EPOCHS_FINAL = 75
    DEVICE_NAME = "mps"
    CLIP_GRAD_NORM = 1.0
    WARMUP_PROPORTION = 0.1

    # --- Loss Function & Balancing ---
    FOCAL_LOSS_GAMMA = 3.0
    MANUAL_CLASS_WEIGHTS = [1.0, 2.5] # Keep weights that worked previously
    USE_WEIGHTED_SAMPLER = True

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 25 # Increase patience slightly? Maybe 20?

    # ==========================================================================
    # ==                       END PARAMETER DEFINITIONS                      ==
    # ==========================================================================

    # --- Derived Paths ---
    DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
    PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
    AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Audio', f"{AUDIO_FEATURE_METHOD}")
    VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Visual', f"{VIDEO_FEATURE_METHOD}")

    # --- Verify Parameters ---
    print("\n--- Verifying Parameters ---")
    if EMBED_DIM % N_HEAD != 0: raise ValueError(f"EMBED_DIM ({EMBED_DIM}) must be divisible by N_HEAD ({N_HEAD})")
    if EMBED_DIM % FUSION_NHEAD != 0: raise ValueError(f"EMBED_DIM ({EMBED_DIM}) must be divisible by FUSION_NHEAD ({FUSION_NHEAD})")
    if MANUAL_CLASS_WEIGHTS is not None and len(MANUAL_CLASS_WEIGHTS) != LABEL_COUNT: raise ValueError(f"MANUAL_CLASS_WEIGHTS length must match LABEL_COUNT")
    print("Parameters verified.")

    # --- Setup Device and Seed ---
    print("\n--- Setting Up Device and Seed ---")
    # ... (Keep device setup logic) ...
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = DEVICE_NAME
    # ... (Rest of device checking logic) ...
    if DEVICE == 'cuda': torch.cuda.manual_seed_all(SEED)
    print(f"Selected device: {DEVICE}")


    # --- Basic Path Checks ---
    print("\n--- Checking Paths ---")
    # ... (Keep path checks) ...
    paths_ok = True
    for p, name in [(DEV_JSON_PATH, "Dev JSON"), (PERSONALIZED_FEATURE_PATH, "Personalized Features"), (AUDIO_FEATURE_DIR, "Audio Features Dir"), (VIDEO_FEATURE_DIR, "Video Features Dir")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found at: {p}")
            paths_ok = False
        else: print(f"Found {name}: {p}")
    if not paths_ok: exit(1)


    # --- Split Data ---
    print("\n--- Splitting Data ---")
    # ... (Keep data splitting logic) ...
    try:
        if TRACK_OPTION=='Track1':
            train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=VAL_RATIO, random_seed=SEED)
        elif TRACK_OPTION=='Track2':
            train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=VAL_PERCENTAGE, seed=SEED)
        else:
            print(f"Error: Invalid TRACK_OPTION '{TRACK_OPTION}'."); exit(1)
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")
    except Exception as e: print(f"Error splitting data: {e}"); exit(1)
    if not train_data or not val_data: print("Error: Data splitting failed."); exit(1)

    # --- Determine Feature Dimensions ---
    print("\n--- Determining Feature Dimensions ---")
    # ... (Keep dynamic dimension determination) ...
    audio_dim, video_dim = None, None
    try:
        for fname in sorted(os.listdir(AUDIO_FEATURE_DIR)):
            if fname.lower().endswith('.npy'):
                try:
                    audio_dim = np.load(os.path.join(AUDIO_FEATURE_DIR, fname), allow_pickle=False).shape[-1]
                    print(f"  Determined Audio Dim: {audio_dim} (from {fname})")
                    break
                except Exception as load_e: print(f"    Could not load {fname}: {load_e}")
        if audio_dim is None: print(f"ERR: No loadable .npy files found in {AUDIO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining audio dim: {e}"); exit(1)
    try:
        for fname in sorted(os.listdir(VIDEO_FEATURE_DIR)):
             if fname.lower().endswith('.npy'):
                try:
                    video_dim = np.load(os.path.join(VIDEO_FEATURE_DIR, fname), allow_pickle=False).shape[-1]
                    print(f"  Determined Video Dim: {video_dim} (from {fname})")
                    break
                except Exception as load_e: print(f"    Could not load {fname}: {load_e}")
        if video_dim is None: print(f"ERR: No loadable .npy files found in {VIDEO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining video dim: {e}"); exit(1)

    NUM_CLASSES = LABEL_COUNT
    print(f"Final Dims Used: Audio={audio_dim}, Video={video_dim}, Pers={PERS_DIM}, Classes={NUM_CLASSES}")

    # --- Create Datasets ---
    print("\n--- Creating Datasets ---");
    # ... (Keep dataset creation) ...
    try:
        full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        val_dataset = AudioVisualDataset(json_data=val_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        print(f"Dataset sizes: Train={len(full_train_dataset)}, Val={len(val_dataset)}")
    except Exception as e: print(f"Error creating datasets: {e}"); traceback.print_exc(); exit(1)
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets are empty."); exit(1)


    # --- Calculate Class Weights and Sampler ---
    # ... (Keep weight/sampler calculation logic using MANUAL_CLASS_WEIGHTS and USE_WEIGHTED_SAMPLER) ...
    focal_loss_weights_tensor = None
    sampler = None
    # ... (Rest of the calculation logic from previous response) ...
    if NUM_CLASSES > 1:
        print("\n--- Processing Class Distribution for Loss Weights and Sampler ---")
        # ... (Label extraction logic) ...
        train_labels = []
        label_extraction_failed = False
        try:
            print("  Extracting labels from training dataset...")
            for i in range(len(full_train_dataset)):
                try:
                    sample = full_train_dataset[i]
                    if isinstance(sample, dict) and 'emo_label' in sample:
                        train_labels.append(int(sample['emo_label']))
                    else: print(f"Warning: Could not extract label from training sample index {i}. Format: {type(sample)}")
                except Exception as sample_e: print(f"Warning: Error accessing sample index {i}: {sample_e}")
            print(f"  Attempted extraction for {len(full_train_dataset)} samples, got {len(train_labels)} labels.")
            if len(train_labels) != len(full_train_dataset): print(f"  Warning: Label count mismatch.")
            if not train_labels: label_extraction_failed = True
        except Exception as e: print(f"  Could not extract training labels: {e}."); label_extraction_failed = True

        if not label_extraction_failed and train_labels:
            class_counts = Counter(train_labels)
            print(f"  Training class counts: {dict(sorted(class_counts.items()))}")
            num_train_samples = len(train_labels)
            # --- Determine Focal Loss Weights ---
            if MANUAL_CLASS_WEIGHTS is not None:
                focal_loss_weights_tensor = torch.tensor(MANUAL_CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
                print(f"  Using MANUALLY DEFINED class weights for FocalLoss: {focal_loss_weights_tensor.cpu().numpy()}")
            # --- Create Sampler ---
            if USE_WEIGHTED_SAMPLER:
                print("  Attempting to create WeightedRandomSampler...")
                if class_counts and num_train_samples > 0:
                    class_sample_weights = {cls: num_train_samples / count if count > 0 else 0 for cls, count in class_counts.items()}
                    sample_weights_list = [class_sample_weights.get(label, 0) for label in train_labels]
                    if sample_weights_list and sum(sample_weights_list) > 0:
                         min_pos_weight = min((w for w in sample_weights_list if w > 0), default=1e-9)
                         safe_sample_weights = [max(w, min_pos_weight * 1e-6) for w in sample_weights_list]
                         sampler = WeightedRandomSampler(torch.DoubleTensor(safe_sample_weights), len(safe_sample_weights), replacement=True)
                         print(f"  Created WeightedRandomSampler for training.")
                    else: print("  Warning: Could not create valid sample weights. Sampler disabled."); sampler = None
                else: print("  Warning: No class counts found, cannot create sampler."); sampler = None
            else: print("  WeightedRandomSampler is disabled."); sampler = None
        else: print("  Label extraction failed or no labels found. Cannot calculate weights or create sampler."); sampler = None


    # --- Model Instantiation ---
    print("\n--- Instantiating Model (Attention Fusion Version) ---")
    try:
        # *** CHANGE: Use the new Attention Fusion model class ***
        final_model = LateFusionAttentionTransformer(
            audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
            embed_dim=EMBED_DIM, nhead=N_HEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
            num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
            fusion_n_layers=FUSION_N_LAYERS, # New param
            fusion_nhead=FUSION_NHEAD,       # New param
            dropout_rate=DROPOUT_RATE,
        ).to(DEVICE)
        print("Model instantiated successfully.")
    except Exception as e: print(f"Final model instantiation error: {e}"); traceback.print_exc(); exit(1)

    # --- Model Summary ---
    if torchinfo:
        # ... (Keep model summary logic, input shapes remain the same) ...
         print("\n--- Model Architecture & Parameters ---")
         example_audio_shape = (BATCH_SIZE, FEATURE_MAX_LEN, audio_dim)
         example_video_shape = (BATCH_SIZE, FEATURE_MAX_LEN, video_dim)
         example_pers_shape = (BATCH_SIZE, PERS_DIM)
         try:
             torchinfo.summary(final_model,
                               input_size=[example_audio_shape, example_video_shape, example_pers_shape],
                               col_names=["input_size", "output_size", "num_params", "mult_adds"],
                               depth=5, device=DEVICE, verbose=0)
         except Exception as e_summary:
             print(f"Could not generate torchinfo summary: {e_summary}")
             total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
             print(f"Total Trainable Parameters (Manual Count): {total_params:,}")


    # --- Create DataLoaders ---
    print(f"\n--- Creating DataLoaders (Batch Size: {BATCH_SIZE}, Sampler: {sampler is not None}) ---")
    # ... (Keep DataLoader creation, using sampler if available) ...
    num_workers = 0
    final_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, shuffle=False if sampler else True, pin_memory=True if DEVICE == 'cuda' else False, drop_last=False)
    final_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE == 'cuda' else False)
    if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: DataLoaders are empty."); exit(1)
    print(f"Train DataLoader length: {len(final_train_loader)}, Val DataLoader length: {len(final_val_loader)}")


    # --- Optimizer ---
    print(f"\n--- Setting up Optimizer (AdamW) ---")
    # ... (Keep AdamW setup using LEARNING_RATE, WEIGHT_DECAY) ...
    print(f"Using AdamW optimizer with Base LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # --- Criterion (Loss Function) ---
    print(f"\n--- Setting up Loss Function (FocalLoss) ---")
    # ... (Keep FocalLoss setup using FOCAL_LOSS_GAMMA, focal_loss_weights_tensor) ...
    print(f"Using FocalLoss with Gamma={FOCAL_LOSS_GAMMA}, Weights={focal_loss_weights_tensor}")
    final_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)


    # --- Learning Rate Scheduler ---
    scheduler = None
    lr_scheduler_active = False
    if len(final_train_loader) > 0:
        # ... (Keep Warmup+Cosine scheduler setup) ...
        try:
            steps_per_epoch = len(final_train_loader)
            TOTAL_TRAINING_STEPS = steps_per_epoch * NUM_EPOCHS_FINAL
            NUM_WARMUP_STEPS = int(TOTAL_TRAINING_STEPS * WARMUP_PROPORTION)
            print(f"\n--- Setting up LR Scheduler ---")
            print(f"Using Cosine schedule with Warmup.")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total steps: {TOTAL_TRAINING_STEPS}, Warmup steps: {NUM_WARMUP_STEPS}")

            scheduler = get_cosine_schedule_with_warmup(
                final_optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=TOTAL_TRAINING_STEPS
            )
            print("LR Scheduler created.")
            lr_scheduler_active = True
        except Exception as e: print(f"Error setting up LR scheduler: {e}. Proceeding without scheduler.")
    else: print("Training loader has length 0. Cannot setup LR scheduler.")

    # --- Training Loop Setup ---
    best_final_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    training_log = []

    # --- Final Training Loop ---
    print(f"\n--- Starting Final Training Loop ({NUM_EPOCHS_FINAL} epochs) ---")
    start_train_time = time.time()

    for epoch in range(NUM_EPOCHS_FINAL):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_FINAL}")
        current_lr = final_optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.3e}")

        # --- Train ---
        print("  Starting training...")
        train_loss, train_acc = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, DEVICE, clip_grad_norm=CLIP_GRAD_NORM, scheduler=scheduler if lr_scheduler_active else None)

        # --- Validate ---
        print("  Starting validation...")
        val_loss, val_acc, val_report = evaluate(final_model, final_val_loader, final_criterion, DEVICE, NUM_CLASSES)

        current_macro_f1 = 0.0
        f1_extracted = False
        # ... (Keep F1 extraction logic) ...
        if val_report and 'macro avg' in val_report and isinstance(val_report['macro avg'], dict) and 'f1-score' in val_report['macro avg']:
             f1_val = val_report['macro avg']['f1-score']
             if f1_val is not None and not np.isnan(f1_val):
                 current_macro_f1 = float(f1_val)
                 f1_extracted = True
             else: print("  Warning: Validation Macro F1 score is NaN or None.")
        else: print("  Warning: Macro F1 score not found in validation report structure.")


        # --- Print epoch results ---
        print(f"  Results: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, Macro F1={current_macro_f1:.4f}")

        # Log results
        # ... (Keep logging logic) ...
        epoch_log = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': current_macro_f1, 'lr': current_lr }
        training_log.append(epoch_log)


        # --- Check for Best Model & Saving ---
        # ... (Keep F1-based saving logic) ...
        if f1_extracted and current_macro_f1 > best_final_macro_f1:
            delta = current_macro_f1 - best_final_macro_f1 if best_final_macro_f1 > -1.0 else current_macro_f1
            print(f"  -> Macro F1 improved by {delta:.4f} (from {max(0, best_final_macro_f1):.4f} to {current_macro_f1:.4f}). Saving model...")
            best_final_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            try:
                torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
                print(f"  -> Saved best model to {MODEL_SAVE_PATH}")
                best_val_report_path = MODEL_SAVE_PATH.replace('.pth', '_best_report.json')
                with open(best_val_report_path, 'w') as f: json.dump(val_report, f, indent=2)
                print(f"  -> Saved best validation report to {best_val_report_path}")
            except Exception as e_save: print(f"  -> Error saving model/report: {e_save}")
        else:
            epochs_without_improvement += 1
            print(f"  -> Macro F1 did not improve for {epochs_without_improvement} epoch(s). Best F1: {max(0, best_final_macro_f1):.4f} at epoch {best_epoch if best_epoch != -1 else 'N/A'}.")


        # --- Early Stopping Check ---
        # ... (Keep early stopping logic) ...
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without validation Macro F1 improvement.")
            break


        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch completed in {epoch_duration:.2f}s")


    # --- After the loop ---
    # ... (Keep training completion summary) ...
    total_training_time = time.time() - start_train_time
    print("\n--- Final Training Complete ---")
    print(f"Total training time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    if best_epoch != -1: print(f"Best Validation Macro F1 score ({best_final_macro_f1:.4f}) achieved at epoch {best_epoch}")
    else: print("No model saved. Macro F1 may not have improved.")

    # Save training log
    # ... (Keep log saving) ...
    log_path = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    try:
        with open(log_path, 'w') as f: json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
    except Exception as e: print(f"Error saving training log: {e}")


    # --- Evaluation ---
    print("\n--- Evaluating Best Saved Model on Validation Set ---")
    model_eval_path = MODEL_SAVE_PATH
    if best_epoch != -1 and os.path.exists(model_eval_path):
        try:
            print(f"Loading best model state from epoch {best_epoch} saved at {model_eval_path}")
            # *** CHANGE: Instantiate the Attention Fusion model ***
            eval_model = LateFusionAttentionTransformer(
                audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
                embed_dim=EMBED_DIM, nhead=N_HEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
                num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
                fusion_n_layers=FUSION_N_LAYERS, # Add fusion params
                fusion_nhead=FUSION_NHEAD,       # Add fusion params
                dropout_rate=DROPOUT_RATE,
            ).to(DEVICE)
            eval_model.load_state_dict(torch.load(model_eval_path, map_location=DEVICE))
            eval_model.eval()
            print("Model loaded successfully.")

            # Re-create criterion for evaluation loss calculation
            eval_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

            # Perform evaluation
            print("Running final evaluation...")
            final_loss, final_acc, final_report = evaluate(eval_model, final_val_loader, eval_criterion, DEVICE, NUM_CLASSES)

            print(f"\nFinal Evaluation Results (using model from epoch {best_epoch}):")
            print(f"  Loss: {final_loss:.4f}")
            print(f"  Accuracy: {final_acc:.4f}")
            if final_report and 'macro avg' in final_report:
                print(f"  Macro F1: {final_report['macro avg'].get('f1-score', 0.0):.4f}")
                print("  Classification Report:")
                print(json.dumps(final_report, indent=2))
            else:
                print("  Classification report could not be generated or was empty.")

        except Exception as e:
            print(f"Error evaluating best model from {model_eval_path}: {e}")
            traceback.print_exc()
    else:
        if best_epoch == -1: print("No best epoch recorded. Skipping final evaluation.")
        else: print(f"Best model path ({model_eval_path}) not found.")

    print(f"\n--- Script End Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")