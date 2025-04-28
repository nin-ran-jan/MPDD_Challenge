import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
import optuna # For hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataClasses.config import Config
from DataLoaders.audioVisualLoader import create_audio_visual_loader

from DataSets.audioVisualDataset import AudioVisualDataset
from Models.early_fusion_mlp import EarlyFusionMLP
from Models.inter_fusion_mlp import IntermediateFusionMLP
from Utils.test_val_split import train_val_split1, train_val_split2

import torchinfo

# --- Training and Evaluation Functions (with device placement fix) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader) == 0: return 0.0, 0.0

    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, dict) or 'A_feat' not in batch: continue
        try:
            # --- Move data to device ---
            audio_feat = batch['A_feat'].to(device)
            video_feat = batch['V_feat'].to(device)
            pers_feat = batch['personalized_feat'].to(device)
            labels = batch['emo_label'].to(device)
            # --- End Move data ---

            batch_size = labels.size(0)
            optimizer.zero_grad()
            outputs = model(audio_feat, video_feat, pers_feat)
            loss = criterion(outputs, labels)
            if torch.isnan(loss): print(f"NaN loss train batch {batch_idx}"); continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
        except Exception as e:
            print(f"Error train batch {batch_idx}: {e}")
            # Print device info on error for debugging
            # print(f"  Model Device={next(model.parameters()).device}")
            # if 'audio_feat' in locals(): print(f"  Audio Device={audio_feat.device}")
            # if 'video_feat' in locals(): print(f"  Video Device={video_feat.device}")
            # if 'pers_feat' in locals(): print(f"  Pers Device={pers_feat.device}")
            # if 'labels' in locals(): print(f"  Labels Device={labels.device}")
            continue # Consider raising e for debugging

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if num_samples > 0 else 0
    return avg_loss, accuracy

# --- Updated evaluate function to return labels and predictions ---
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader) == 0: return 0.0, 0.0, {}, [], [] # Return empty lists

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not isinstance(batch, dict) or 'A_feat' not in batch: continue
            try:
                # --- Move data to device ---
                audio_feat = batch['A_feat'].to(device)
                video_feat = batch['V_feat'].to(device)
                pers_feat = batch['personalized_feat'].to(device)
                labels = batch['emo_label'].to(device)
                # --- End Move data ---

                batch_size = labels.size(0)
                outputs = model(audio_feat, video_feat, pers_feat)
                loss = criterion(outputs, labels)
                if torch.isnan(loss): print(f"NaN loss eval batch {batch_idx}"); continue
                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
            except Exception as e:
                print(f"Error eval batch {batch_idx}: {e}")
                # Print device info on error for debugging
                # print(f"  Model Device={next(model.parameters()).device}")
                # if 'audio_feat' in locals(): print(f"  Audio Device={audio_feat.device}")
                # if 'video_feat' in locals(): print(f"  Video Device={video_feat.device}")
                # if 'pers_feat' in locals(): print(f"  Pers Device={pers_feat.device}")
                # if 'labels' in locals(): print(f"  Labels Device={labels.device}")
                continue # Consider raising e for debugging

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if num_samples > 0 else 0
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True) if num_samples > 0 else {}
    # Return labels and preds along with other metrics
    return avg_loss, accuracy, report_dict, all_labels, all_preds


# --- Optuna Objective Function with Cross-Validation ---
def objective(trial, full_train_dataset, config):
    # --- Suggest Hyperparameters ---
    modality_hidden_dim = trial.suggest_categorical("modality_hidden_dim", [32, 64, 128])
    fusion_hidden_dim = trial.suggest_categorical("fusion_hidden_dim", [64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = config.batch_size

    n_splits = config.cv_folds
    labels_np = []
    try: # Extract labels directly
        label_key = {2: 'bin_category', 3: 'tri_category', 5: 'pen_category'}.get(config.labelcount)
        if label_key and hasattr(full_train_dataset, 'data') and isinstance(full_train_dataset.data, list):
             valid_labels = [int(item[label_key]) for item in full_train_dataset.data if isinstance(item, dict) and label_key in item and isinstance(item[label_key], (int, float, str)) and str(item[label_key]).isdigit()]
             labels_np = np.array(valid_labels)
             if len(labels_np) != len(full_train_dataset.data): print(f"Warn: Extracted {len(labels_np)}/{len(full_train_dataset.data)} labels.")
        else: raise ValueError("Cannot extract labels directly.")
    except Exception as e: # Fallback iteration
         print(f"Direct label extract failed ({e}). Falling back.")
         try:
             temp_loader = DataLoader(full_train_dataset, batch_size=config.batch_size); labels_list = [b['emo_label'].numpy() for b in temp_loader if 'emo_label' in b];
             if labels_list: labels_np = np.concatenate(labels_list)
         except Exception as e_iter: print(f"Label iter failed: {e_iter}"); return 0.0
    if len(labels_np) == 0: print("Error: No labels extracted."); return 0.0

    unique_labels, counts = np.unique(labels_np, return_counts=True)
    if len(counts) == 0: print("Error: No unique labels."); return 0.0
    min_samples = np.min(counts) if len(counts) > 0 else 0
    actual_n_splits = min(n_splits, min_samples)
    if actual_n_splits < 2: print(f"Warn: Smallest class ({min_samples}) too small for {n_splits}-CV."); return 0.0
    if actual_n_splits < n_splits: print(f"Warn: Reducing CV folds to {actual_n_splits}.")

    skf = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=config.seed)
    fold_accuracies = []
    print(f"\nTrial {trial.number}: mod_hidden={modality_hidden_dim}, fusion_hidden={fusion_hidden_dim}, dr={dropout_rate:.2f}, lr={lr:.6f}, wd={weight_decay:.6f}")

    audio_dim, video_dim, pers_dim, num_classes = config.audio_dim, config.video_dim, config.pers_dim, config.num_classes
    if not all([audio_dim, video_dim, pers_dim, num_classes]): print("Error: Invalid dims."); return 0.0

    split_indices = np.arange(len(labels_np))
    global_step_counter = 0

    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(split_indices, labels_np)):
        print(f"  Fold {fold+1}/{actual_n_splits}...")
        train_orig_idx = split_indices[train_split_idx]
        val_orig_idx = split_indices[val_split_idx]
        if np.max(train_orig_idx) >= len(full_train_dataset) or np.max(val_orig_idx) >= len(full_train_dataset):
            print(f"Error: Fold {fold+1} indices out of bounds. Skipping fold.")
            continue

        cv_train_dataset = Subset(full_train_dataset, train_orig_idx)
        cv_val_dataset = Subset(full_train_dataset, val_orig_idx)
        if len(cv_train_dataset) == 0 or len(cv_val_dataset) == 0: print(f"Warn: Fold {fold+1} empty subset."); continue

        cv_train_loader = DataLoader(cv_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        cv_val_loader = DataLoader(cv_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        try:
            # --- Use IntermediateFusionMLP ---
            model = IntermediateFusionMLP(
                audio_dim=audio_dim, video_dim=video_dim, pers_dim=pers_dim,
                modality_hidden_dim=modality_hidden_dim, # New hyperparam
                fusion_hidden_dim=fusion_hidden_dim,     # Renamed hyperparam
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
            # --- Ensure model is on device before training fold ---
            model.to(config.device)
        except ValueError as e: print(f"Model init error: {e}"); continue

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # --- Move criterion to device ---
        criterion = nn.CrossEntropyLoss().to(config.device)
        best_fold_acc = 0.0

        for epoch in range(config.num_epochs_tuning):
            try:
                # Pass the correct device
                train_loss, train_acc = train_epoch(model, cv_train_loader, optimizer, criterion, config.device)
                if train_acc is not None:
                     # Pass the correct device, capture labels/preds if needed, ignore here
                     val_loss, val_acc, _, _, _ = evaluate(model, cv_val_loader, criterion, config.device)
                     if val_acc is not None:
                          best_fold_acc = max(best_fold_acc, val_acc)
                          trial.report(val_acc, global_step_counter)
                          global_step_counter += 1
                          if trial.should_prune(): raise optuna.TrialPruned()
                     else: print(f"    Eval failed."); break
                else: print(f"    Train failed."); break
            except optuna.TrialPruned: raise
            except Exception as e_epoch: print(f"Error Fold {fold+1} Epoch {epoch+1}: {e_epoch}"); break

        fold_accuracies.append(best_fold_acc)
        print(f"  Fold {fold+1} Best Val Acc: {best_fold_acc:.4f}")

    average_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
    print(f"Trial {trial.number} Avg CV Acc: {average_accuracy:.4f}")
    if not fold_accuracies: print(f"Warn: Trial {trial.number} no folds complete."); return 0.0
    return average_accuracy


# --- Main Execution ---
if __name__ == '__main__':

    config = Config.from_json('config.json')
    DATA_ROOT_PATH = config.data_root_path
    DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
    PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
    AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Audio', f"{config.audio_feature_method}")
    VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Visual', f"{config.video_feature_method}")

    # Basic Path Checks
    if not os.path.exists(DEV_JSON_PATH): print(f"ERROR: DEV_JSON_PATH not found: {DEV_JSON_PATH}"); exit()
    if not os.path.exists(PERSONALIZED_FEATURE_PATH): print(f"ERROR: PERSONALIZED_FEATURE_PATH not found: {PERSONALIZED_FEATURE_PATH}"); exit()
    if not os.path.isdir(AUDIO_FEATURE_DIR): print(f"ERROR: AUDIO_FEATURE_DIR not found: {AUDIO_FEATURE_DIR}"); exit()
    if not os.path.isdir(VIDEO_FEATURE_DIR): print(f"ERROR: VIDEO_FEATURE_DIR not found: {VIDEO_FEATURE_DIR}"); exit()

    # Setup Device and Seed
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    # --- Check for MPS device specifically ---
    if config.device == 'mps' and torch.backends.mps.is_available():
         if not torch.backends.mps.is_built():
             print("MPS not available because the current PyTorch install was not built with MPS enabled.")
             config.device = 'cpu'; print("Using CPU.")
         else:
             print("Using MPS device.")
             config.device = 'mps'
    elif config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed); print(f"Using CUDA: {torch.cuda.get_device_name(0)}"); config.device = 'cuda'
    else:
        if config.device != 'cpu': print(f"Warn: Device '{config.device}' specified but unavailable/unsupported. Using CPU.")
        config.device = 'cpu'; print("Using CPU.")

    # Split Data
    print("Splitting data..."); train_data, val_data = [], []
    try:
        if config.track_option=='Track1': train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=0.1, random_seed=config.seed)
        elif config.track_option=='Track2': train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=0.1, seed=config.seed)
        else: print(f"Error: Invalid track_option '{config.track_option}'."); exit()
    except Exception as e: print(f"Error splitting data: {e}"); exit()
    if not train_data or not val_data: print("Error: Data splitting failed."); exit()
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")

    # Determine Feature Dimensions
    print("Determining feature dimensions..."); audio_dim_found, video_dim_found = None, None
    try:
        for fname in os.listdir(AUDIO_FEATURE_DIR):
            if fname.endswith('.npy'): audio_dim_found = np.load(os.path.join(AUDIO_FEATURE_DIR, fname)).shape[-1]; break
        if audio_dim_found is None: print(f"ERR: No .npy in {AUDIO_FEATURE_DIR}"); exit()
        print(f"  Audio Dim: {audio_dim_found}")
    except Exception as e: print(f"ERR determining audio dim: {e}"); exit()
    try:
        for fname in os.listdir(VIDEO_FEATURE_DIR):
            if fname.endswith('.npy'): video_dim_found = np.load(os.path.join(VIDEO_FEATURE_DIR, fname)).shape[-1]; break
        if video_dim_found is None: print(f"ERR: No .npy in {VIDEO_FEATURE_DIR}"); exit()
        print(f"  Video Dim: {video_dim_found}")
    except Exception as e: print(f"ERR determining video dim: {e}"); exit()

    # Update Config with determined dimensions
    config.audio_dim, config.video_dim, config.pers_dim = audio_dim_found, video_dim_found, 1024
    config.num_classes = config.labelcount
    print(f"Final Dims: A={config.audio_dim}, V={config.video_dim}, P={config.pers_dim}, Cls={config.num_classes}")

    # Create Datasets
    print("Creating Datasets...");
    try:
        full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        val_dataset = AudioVisualDataset(json_data=val_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
    except Exception as e: print(f"Error creating datasets: {e}"); exit()
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets empty."); exit()

    # Optuna Tuning
    print("\n--- Starting Hyperparameter Tuning ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    try: study.optimize(lambda trial: objective(trial, full_train_dataset, config), n_trials=config.optuna_trials)
    except Exception as e: print(f"Optuna error: {e}");
    print("\n--- Optuna Study Complete ---")
    if not study.trials: print("No Optuna trials ran."); exit()
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials: print("No Optuna trials completed."); exit()
    best_trial = study.best_trial; best_params = best_trial.params
    print(f"Best trial #{best_trial.number}: Acc={best_trial.value:.4f}"); [print(f"  {k}: {v}") for k, v in best_params.items()]

    # Final Training
    print("\n--- Training Final Model ---")
    try:
        # --- Use IntermediateFusionMLP ---
        final_model = IntermediateFusionMLP(
            audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
            modality_hidden_dim=best_params['modality_hidden_dim'], # Use best param
            fusion_hidden_dim=best_params['fusion_hidden_dim'],     # Use best param
            num_classes=config.num_classes,
            dropout_rate=best_params['dropout_rate']
        ).to(config.device) # Move model to device
    except ValueError as e: print(f"Final model init error: {e}"); exit()

    # --- Verify Model Device ---
    model_device = next(final_model.parameters()).device
    print(f"Final model is on device: {model_device}")
    if model_device.type != config.device:
        print(f"ERROR: Model is on {model_device.type} but config specifies {config.device}!")
        final_model.to(config.device) # Try moving again
        model_device = next(final_model.parameters()).device
        print(f"Re-attempted move. Model is now on device: {model_device}")
        if model_device.type != config.device:
             exit(f"Failed to move model to correct device type ({config.device}).")

    # --- Model Summary ---
    if torchinfo:
        print("\n--- Final Model Architecture & Parameters ---")
        example_audio_shape = (config.batch_size, config.feature_max_len, config.audio_dim)
        example_video_shape = (config.batch_size, config.feature_max_len, config.video_dim)
        example_pers_shape = (config.batch_size, config.pers_dim)
        try:
            # --- Use IntermediateFusionMLP for summary ---
            summary_model = IntermediateFusionMLP(
                 audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
                 modality_hidden_dim=best_params['modality_hidden_dim'],
                 fusion_hidden_dim=best_params['fusion_hidden_dim'],
                 num_classes=config.num_classes, dropout_rate=best_params['dropout_rate']
            ).to('cpu') # Use CPU for summary
            torchinfo.summary(summary_model, input_size=[example_audio_shape, example_video_shape, example_pers_shape],
                              col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3, verbose=0)
            print(summary_model)
            del summary_model
        except Exception as e_summary:
            print(f"Could not generate torchinfo summary: {e_summary}")
            total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
            print(f"Total Trainable Parameters (Manual Count): {total_params:,}")
    else:
        total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        print(f"\nTotal Trainable Parameters: {total_params:,}")


    final_train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    final_val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: Final loaders empty."); exit()

    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    # --- Move criterion to device ---
    final_criterion = nn.CrossEntropyLoss().to(config.device)
    best_final_val_acc, best_epoch = 0.0, -1

    # --- Add explicit model.to(device) before final training loop ---
    print(f"Ensuring final model is on device {config.device} before training...")
    final_model.to(config.device)
    # --- End add explicit model.to(device) ---

    for epoch in range(config.num_epochs_final):
        try:
            # Pass the correct device from config
            train_loss, train_acc = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, config.device)
            # --- Capture all return values from evaluate ---
            val_loss, val_acc, val_report, _, _ = evaluate(final_model, final_val_loader, final_criterion, config.device)
            print(f"Epoch {epoch+1}/{config.num_epochs_final}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            if val_acc is not None and val_acc > best_final_val_acc:
                best_final_val_acc, best_epoch = val_acc, epoch + 1
                try: torch.save(final_model.state_dict(), config.model_save_path); print(f"  -> Saved best model (Epoch {best_epoch}, Val Acc: {best_final_val_acc:.4f})")
                except Exception as e_save: print(f"Save error: {e_save}")
        except Exception as e_final_epoch: print(f"Error final epoch {epoch+1}: {e_final_epoch}"); break

    print("\n--- Final Training Complete ---")
    print(f"Best Val Acc ({best_final_val_acc:.4f}) at epoch {best_epoch}")

    # Evaluation
    print("\n--- Evaluating Best Saved Model ---")
    if best_epoch != -1 and os.path.exists(config.model_save_path):
        try:
            # --- Use IntermediateFusionMLP ---
            eval_model = IntermediateFusionMLP(
                 audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
                 modality_hidden_dim=best_params['modality_hidden_dim'],
                 fusion_hidden_dim=best_params['fusion_hidden_dim'],
                 num_classes=config.num_classes,
                 dropout_rate=best_params['dropout_rate']
            ).to(config.device) # Move model to device
            eval_model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
            eval_model.eval()
            # --- Capture all return values from evaluate ---
            _, final_acc, final_report_dict, final_true_labels, final_preds = evaluate(eval_model, final_val_loader, final_criterion, config.device)

            print(f"Final Best Model Validation Accuracy: {final_acc:.4f}")
            print("Final Best Model Classification Report:")
            # Use the report dict for printing
            print(json.dumps(final_report_dict, indent=2))

            # --- Calculate and Print Confusion Matrix ---
            if final_true_labels and final_preds: # Check if lists are not empty
                 cm = confusion_matrix(final_true_labels, final_preds)
                 print("\nFinal Best Model Confusion Matrix:")
                 print(cm)

            else:
                 print("\nCould not generate confusion matrix (no predictions or labels found).")
            # --- End Confusion Matrix ---

        except Exception as e: print(f"Error evaluating best model: {e}")
    else: print("Best model not saved/found. Skipping final evaluation.")