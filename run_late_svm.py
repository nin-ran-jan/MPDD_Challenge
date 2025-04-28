import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
from DataLoaders.audioVisualLoader import create_audio_visual_loader
from DataClasses.config import Config
import os
import json

from Models.late_fusion_svm import LateFusionSVM, extract_features_for_sklearn, report_svm_parameters, train_svm_modality
from Utils.test_val_split import train_val_split1, train_val_split2

# --- Example Usage ---
if __name__ == '__main__':

    # --- Configuration ---
    config:Config = Config.from_json('config.json')

    DATA_ROOT_PATH=config.data_root_path
    DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
    PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')


    # Choose ONE audio and ONE video feature type by pointing to their directories
    AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Audio', f"{config.audio_feature_method}") + '/' # e.g., '/data/features/wav2vec_npy/'
    VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Visual', f"{config.video_feature_method}") + '/' # e.g., '/data/features/resnet50_npy/'

    LABEL_COUNT = config.labelcount # Or 2 or 5, depending on your task ('tri_category')
    MAX_LEN = config.feature_max_len # As used in your Dataset class for padding/truncation
    BATCH_SIZE = config.batch_size # For data loading (doesn't affect SVM training directly much)
    CV_FOLDS = config.cv_folds # For GridSearchCV (adjust based on training set size)

    train_data = []
    val_data = []

    if config.track_option=='Track1':
        train_data, val_data, train_category_count, val_category_count = train_val_split1(DEV_JSON_PATH, val_ratio=0.1, random_seed=32)
    elif config.track_option=='Track2':
        train_data, val_data, train_category_count, val_category_count = train_val_split2(DEV_JSON_PATH, val_percentage=0.1,seed=32)


    # --- Create Datasets and DataLoaders ---
    print("Creating Training Dataset...")
    train_dataloader = create_audio_visual_loader(
        json_data=train_data,
        label_count=LABEL_COUNT,
        personalized_feature_file=PERSONALIZED_FEATURE_PATH,
        max_len=MAX_LEN,
        audio_path=AUDIO_FEATURE_DIR,
        video_path=VIDEO_FEATURE_DIR,
        isTest=False,
        batch_size=BATCH_SIZE
    )
    # Check if dataset loading resulted in data
    if len(train_dataloader) == 0:
         print("Error: Training dataset is empty. Check JSON data and feature paths.")
         exit()

    print("Creating Validation Dataset...")
    val_dataloader = create_audio_visual_loader(
        json_data=val_data,
        label_count=LABEL_COUNT,
        personalized_feature_file=PERSONALIZED_FEATURE_PATH,
        max_len=MAX_LEN,
        audio_path=AUDIO_FEATURE_DIR,
        video_path=VIDEO_FEATURE_DIR,
        isTest=False, # Evaluate on validation set with labels
        batch_size=BATCH_SIZE
    )
    if len(val_dataloader) == 0:
         print("Error: Validation dataset is empty. Check JSON data and feature paths.")
         # Decide if you want to exit or continue without validation
         # exit()


    # --- Extract Features for Scikit-learn ---
    print("\nExtracting features for training...")
    X_audio_train, X_video_train, X_pers_train, y_train = extract_features_for_sklearn(train_dataloader)

    print("\nExtracting features for validation...")
    X_audio_val, X_video_val, X_pers_val, y_val = extract_features_for_sklearn(val_dataloader)

    # Check if data extraction was successful
    if X_audio_train.shape[0] == 0 or X_audio_val.shape[0] == 0:
         print("Error: Feature extraction resulted in empty arrays. Cannot proceed.")
         exit()

    # --- Train Individual SVMs ---
    svm_audio, scaler_audio = train_svm_modality(X_audio_train, y_train, "Audio", cv_folds=CV_FOLDS)
    svm_video, scaler_video = train_svm_modality(X_video_train, y_train, "Video", cv_folds=CV_FOLDS)
    svm_pers, scaler_pers = train_svm_modality(X_pers_train, y_train, "Personalized", cv_folds=CV_FOLDS)

    # --- Optional: Save Trained Models ---
    # model_save_dir = './trained_svms'
    # os.makedirs(model_save_dir, exist_ok=True)
    # if svm_audio: joblib.dump(svm_audio, os.path.join(model_save_dir,'svm_audio.joblib'))
    # if scaler_audio: joblib.dump(scaler_audio, os.path.join(model_save_dir,'scaler_audio.joblib'))
    # if svm_video: joblib.dump(svm_video, os.path.join(model_save_dir,'svm_video.joblib'))
    # if scaler_video: joblib.dump(scaler_video, os.path.join(model_save_dir,'scaler_video.joblib'))
    # if svm_pers: joblib.dump(svm_pers, os.path.join(model_save_dir,'svm_pers.joblib'))
    # if scaler_pers: joblib.dump(scaler_pers, os.path.join(model_save_dir,'scaler_pers.joblib'))
    # print(f"Trained models and scalers saved to {model_save_dir}")

    # --- Create Late Fusion Model ---
    # Choose fusion strategy: 'average_proba' or 'majority_vote'
    fusion_model = LateFusionSVM(
        svm_audio, scaler_audio,
        svm_video, scaler_video,
        svm_pers, scaler_pers,
        fusion_strategy='average_proba' # Recommended start
    )
    fusion_model.eval() # Set to evaluation mode (though not strictly necessary for SVMs)

    # --- Evaluate on Validation Set ---
    print("\n--- Evaluating Fusion Model on Validation Set ---")

    # Pass the already extracted validation features
    # Convert numpy arrays back to tensors for the model's forward pass
    A_feat_val_tensor = torch.from_numpy(X_audio_val).float()
    V_feat_val_tensor = torch.from_numpy(X_video_val).float()
    P_feat_val_tensor = torch.from_numpy(X_pers_val).float()


    with torch.no_grad():
        y_pred_fused = fusion_model(A_feat_val_tensor, V_feat_val_tensor, P_feat_val_tensor)
        y_pred_fused_np = y_pred_fused.cpu().numpy()


    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred_fused_np)
    report = classification_report(y_val, y_pred_fused_np, zero_division=0)

    print(f"Validation Accuracy (Fused): {accuracy:.4f}")
    print("Validation Classification Report (Fused):")
    print(report)

    # --- Evaluate Individual Modalities (Optional Baseline) ---
    print("\n--- Evaluating Individual Modalities on Validation Set ---")
    if svm_audio and scaler_audio:
        X_audio_val_scaled = scaler_audio.transform(np.nan_to_num(X_audio_val))
        y_pred_audio = svm_audio.predict(X_audio_val_scaled)
        acc_audio = accuracy_score(y_val, y_pred_audio)
        print(f"Validation Accuracy (Audio Only): {acc_audio:.4f}")
        # print(classification_report(y_val, y_pred_audio, zero_division=0))


    if svm_video and scaler_video:
        X_video_val_scaled = scaler_video.transform(np.nan_to_num(X_video_val))
        y_pred_video = svm_video.predict(X_video_val_scaled)
        acc_video = accuracy_score(y_val, y_pred_video)
        print(f"Validation Accuracy (Video Only): {acc_video:.4f}")
        # print(classification_report(y_val, y_pred_video, zero_division=0))

    if svm_pers and scaler_pers:
        X_pers_val_scaled = scaler_pers.transform(np.nan_to_num(X_pers_val))
        y_pred_pers = svm_pers.predict(X_pers_val_scaled)
        acc_pers = accuracy_score(y_val, y_pred_pers)
        print(f"Validation Accuracy (Personalized Only): {acc_pers:.4f}")
        # print(classification_report(y_val, y_pred_pers, zero_division=0))
    

    report_svm_parameters(svm_audio, scaler_audio, "Audio")
    report_svm_parameters(svm_video, scaler_video, "Video")
    report_svm_parameters(svm_pers, scaler_pers, "Personalized")
    
    total_elements = 0
    if svm_audio and scaler_audio:
        total_elements += (2 * scaler_audio.n_features_in_) + svm_audio.support_vectors_.size + svm_audio.dual_coef_.size + svm_audio.intercept_.size
    if svm_video and scaler_video:
         total_elements += (2 * scaler_video.n_features_in_) + svm_video.support_vectors_.size + svm_video.dual_coef_.size + svm_video.intercept_.size
    if svm_pers and scaler_pers:
         total_elements += (2 * scaler_pers.n_features_in_) + svm_pers.support_vectors_.size + svm_pers.dual_coef_.size + svm_pers.intercept_.size
    
    print(f"\n--- Overall Model Complexity ---")
    print(f"Total stored elements across all scalers and SVMs: {int(total_elements)}") # Use int() for cleaner output