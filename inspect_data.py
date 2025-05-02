import torch
import numpy as np
import os
import json
import traceback
from torch.utils.data import DataLoader

try:
    from DataClasses.config import Config
    from DataSets.audioVisualDataset import AudioVisualDataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("ERROR: Ensure this script is run from the MPDD-challege-main directory")
    print("       or that the DataClasses and DataSets modules are accessible.")
    exit()

def inspect_data(config_path='config.json', num_batches_to_check=5):
    """
    Loads data using the project's config and DataLoader, then inspects
    a few batches for NaNs, Infs, and basic statistics.
    """
    print(f"--- Starting Data Inspection using '{config_path}' ---")
    try:
        config = Config.from_json(config_path)
        print("\nLoaded Configuration:")
        print(f"  Data Root: {config.data_root_path}")
        print(f"  Window Time: {config.window_split_time}s")
        print(f"  Audio Features: {config.audio_feature_method}")
        print(f"  Video Features: {config.video_feature_method}")
        print(f"  Label Count: {config.labelcount}")
        print(f"  Max Seq Len: {config.feature_max_len}")
        print(f"  Batch Size: {config.batch_size}")

        DATA_ROOT_PATH = config.data_root_path
        DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
        PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
        AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Audio', f"{config.audio_feature_method}")
        VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Visual', f"{config.video_feature_method}")

        print("\nValidating paths from config...")
        required_paths = {"DEV_JSON": DEV_JSON_PATH, "PERS_FEAT": PERSONALIZED_FEATURE_PATH, "AUDIO_DIR": AUDIO_FEATURE_DIR, "VIDEO_DIR": VIDEO_FEATURE_DIR}
        paths_ok = True
        for name, path in required_paths.items():
            is_dir = name.endswith("_DIR")
            exists = os.path.isdir(path) if is_dir else os.path.exists(path)
            print(f"  Checking {name}: {path} ... {'Found' if exists else 'NOT FOUND!'}")
            if not exists:
                paths_ok = False
        if not paths_ok:
            print("\nERROR: One or more required data paths not found. Aborting inspection.")
            return

        print(f"\nLoading training data definitions from: {DEV_JSON_PATH}")
        try:
            with open(DEV_JSON_PATH, 'r') as f:
                full_training_json_data = json.load(f)
            if not full_training_json_data or not isinstance(full_training_json_data, list):
                raise ValueError(f"Invalid format or empty file: {DEV_JSON_PATH}")
            print(f"Found {len(full_training_json_data)} entries in the JSON.")
        except Exception as e:
            print(f"ERROR: Failed to load or parse {DEV_JSON_PATH}: {e}")
            return

        print("\nCreating Dataset instance...")
        try:
            dataset = AudioVisualDataset(
                json_data=full_training_json_data,
                label_count=config.labelcount,
                personalized_feature_file=PERSONALIZED_FEATURE_PATH,
                max_len=config.feature_max_len,
                audio_path=AUDIO_FEATURE_DIR,
                video_path=VIDEO_FEATURE_DIR,
                isTest=False
            )
            if len(dataset) == 0:
                print("ERROR: Dataset created but is empty. Check JSON content and feature paths.")
                return
            print(f"Dataset created successfully with {len(dataset)} items.")
        except Exception as e:
             print(f"ERROR: Failed to create AudioVisualDataset: {e}")
             traceback.print_exc()
             return

        print("Creating DataLoader instance...")
        inspect_batch_size = config.batch_size
        dataloader = DataLoader(dataset, batch_size=inspect_batch_size, shuffle=False, num_workers=0)

        print(f"\n--- Checking first {num_batches_to_check} batches (Batch Size: {inspect_batch_size}) ---")
        all_stats = {'A_feat': [], 'V_feat': [], 'personalized_feat': []}
        nan_inf_found = {'A_feat': False, 'V_feat': False, 'personalized_feat': False}
        batch_count = 0

        for batch in dataloader:
            if batch_count >= num_batches_to_check:
                break

            if not isinstance(batch, dict):
                print(f"Batch {batch_count}: Item yielded is not a dictionary, skipping. Type: {type(batch)}")
                batch_count += 1
                continue

            print(f"\n--- Analyzing Batch {batch_count} ---")
            for key in ['A_feat', 'V_feat', 'personalized_feat']:
                if key not in batch:
                    print(f"  Key '{key}' not found in batch.")
                    continue

                tensor = batch[key]
                if not isinstance(tensor, torch.Tensor):
                     print(f"  Data for key '{key}' is not a tensor (Type: {type(tensor)}). Skipping.")
                     continue

                print(f"  {key} - Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")

                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                nan_inf_issue = False

                if has_nan:
                    print(f"  🚨 ALERT: NaN found in {key}!")
                    nan_inf_found[key] = True
                    nan_inf_issue = True
                if has_inf:
                    print(f"  🚨 ALERT: Inf found in {key}!")
                    nan_inf_found[key] = True
                    nan_inf_issue = True

                if not nan_inf_issue and tensor.numel() > 0:
                    try:
                        tensor_float = tensor.float()
                        stats = {
                            'min': torch.min(tensor_float).item(),
                            'max': torch.max(tensor_float).item(),
                            'mean': torch.mean(tensor_float).item(),
                            'std': torch.std(tensor_float).item()
                        }
                        print(f"  {key} - Stats: Min={stats['min']:.4f}, Max={stats['max']:.4f}, Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")
                        all_stats[key].append(stats)
                    except Exception as e_stats:
                         print(f"  Error calculating stats for {key}: {e_stats}")
                elif tensor.numel() == 0:
                     print(f"  {key} - Tensor is empty.")
                else:
                     print(f"  {key} - Skipping stats calculation due to NaN/Inf.")

            if 'emo_label' in batch:
                labels = batch['emo_label']
                print(f"  emo_label - Shape: {labels.shape}, Dtype: {labels.dtype}, Values: {labels.tolist()}")
            else:
                print("  Key 'emo_label' not found in batch.")

            batch_count += 1

    except Exception as e:
        print(f"\n--- ERROR during data inspection setup or loop ---")
        print(e)
        traceback.print_exc()

    finally:
        print("\n--- Input Data Inspection Summary ---")
        overall_issue = False
        for key in ['A_feat', 'V_feat', 'personalized_feat']:
            if nan_inf_found[key]:
                print(f"🚨 CRITICAL: NaN or Inf values WERE DETECTED in '{key}' features within the first {num_batches_to_check} batches.")
                overall_issue = True
            else:
                print(f"✅ OK: No NaN or Inf values found in '{key}' features within the first {num_batches_to_check} batches checked.")
                if all_stats[key]:
                     try:
                         all_means = [s['mean'] for s in all_stats[key]]
                         all_stds = [s['std'] for s in all_stats[key]]
                         all_maxs = [s['max'] for s in all_stats[key]]
                         all_mins = [s['min'] for s in all_stats[key]]
                         if all_means:
                             avg_mean = np.mean(all_means)
                             avg_std = np.mean(all_stds)
                             max_val = np.max(all_maxs)
                             min_val = np.min(all_mins)
                             print(f"   -> Stats Summary ({key}): AvgMean={avg_mean:.4f}, AvgStd={avg_std:.4f}, OverallMax={max_val:.4f}, OverallMin={min_val:.4f}")
                         else:
                              print(f"   -> No valid stats collected for {key}.")
                     except Exception as e_summary_stats:
                          print(f"   -> Error calculating stats summary for {key}: {e_summary_stats}")
                else:
                     print(f"   -> No stats calculated for {key} (possibly due to NaN/Inf or empty tensors).")
        if overall_issue:
            print("\nRECOMMENDATION: The NaN/Inf values MUST be addressed.")
        else:
            print("\nRECOMMENDATION: Based on checks of initial batches, input data appears numerically valid (no NaNs/Infs).")
        print("--- Inspection Complete ---")


if __name__ == '__main__':
    inspect_data(config_path='config.json', num_batches_to_check=10)