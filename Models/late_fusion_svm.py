import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Helper Function to Extract Features and Labels for Scikit-learn
def extract_features_for_sklearn(dataloader):
    all_a_feats_agg = [] # Store aggregated features
    all_v_feats_agg = [] # Store aggregated features
    all_p_feats = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # --- Get features from batch ---
            # A_feat and V_feat are expected to be [batch_size, max_len, feature_dim]
            # based on the unchanged Dataset returning padded/truncated sequences.
            audio_batch = batch['A_feat']
            video_batch = batch['V_feat']
            pers_batch = batch['personalized_feat'] # Key from the provided Dataset
            label_batch = batch['emo_label']       # Key from the provided Dataset

            # --- Aggregate Audio and Video Features ---
            # Apply mean pooling over the sequence length dimension (dim=1)
            # Resulting shape: [batch_size, feature_dim]
            if audio_batch.ndim == 3: # Check if aggregation is needed
                 audio_batch_agg = torch.mean(audio_batch, dim=1)
            elif audio_batch.ndim == 2: # Already aggregated somehow? Use as is.
                 print("Warning: Audio batch received in extract_features_for_sklearn was already 2D.")
                 audio_batch_agg = audio_batch
            else: # Should not happen
                 raise ValueError(f"Unexpected audio batch dimension: {audio_batch.ndim}")

            if video_batch.ndim == 3: # Check if aggregation is needed
                 video_batch_agg = torch.mean(video_batch, dim=1)
            elif video_batch.ndim == 2: # Already aggregated somehow? Use as is.
                 print("Warning: Video batch received in extract_features_for_sklearn was already 2D.")
                 video_batch_agg = video_batch
            else: # Should not happen
                 raise ValueError(f"Unexpected video batch dimension: {video_batch.ndim}")


            # --- Append aggregated features and others to lists ---
            all_a_feats_agg.append(audio_batch_agg.cpu().numpy())
            all_v_feats_agg.append(video_batch_agg.cpu().numpy())
            all_p_feats.append(pers_batch.cpu().numpy()) # Personalized features are already [batch_size, dim]
            all_labels.append(label_batch.cpu().numpy())

    # Check if any features were loaded
    if not all_a_feats_agg:
        print("Error: No data loaded.")
        return np.empty((0,1)), np.empty((0,1)), np.empty((0,1)), np.empty((0))

    # Concatenate batches -> Result should be 2D: [total_samples, feature_dim]
    try:
        # Use the lists containing aggregated features
        X_audio = np.concatenate(all_a_feats_agg, axis=0)
        X_video = np.concatenate(all_v_feats_agg, axis=0)
        X_pers = np.concatenate(all_p_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)
    except ValueError as e:
        print(f"Error during concatenation: {e}")
        print("This might happen if features within a batch have inconsistent shapes AFTER aggregation.")
        if all_a_feats_agg: print("First aggregated audio batch item shape:", all_a_feats_agg[0].shape)
        if all_v_feats_agg: print("First aggregated video batch item shape:", all_v_feats_agg[0].shape)
        if all_p_feats: print("First pers batch item shape:", all_p_feats[0].shape)
        raise e

    # Handle potential NaNs or Infs (important for SVM)
    X_audio = np.nan_to_num(X_audio)
    X_video = np.nan_to_num(X_video)
    X_pers = np.nan_to_num(X_pers)

    # --- Verification Step (Should Pass Now) ---
    if X_audio.ndim != 2 or X_video.ndim != 2 or X_pers.ndim != 2:
        print(f"ERROR: Features are not 2D after concatenation and aggregation!")
        print(f"Shapes: Audio={X_audio.shape}, Video={X_video.shape}, Pers={X_pers.shape}")
        raise RuntimeError("Aggregation failed to produce 2D features.")
    else:
         print(f"Extracted features (aggregated): Audio shape={X_audio.shape}, Video shape={X_video.shape}, Pers shape={X_pers.shape}, Labels shape={y.shape}")

    return X_audio, X_video, X_pers, y

# --- SVM Training and Tuning ---
def train_svm_modality(X_train, y_train, modality_name, cv_folds=5, param_grid=None):
    """Trains and tunes an SVM for a single modality."""
    print(f"\n--- Training SVM for {modality_name} ---")

    # Feature Scaling (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define parameter grid for GridSearchCV
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1], # 'scale' is 1 / (n_features * X.var())
            'probability': [True] # Enable probability estimates for fusion
        }
        # Remove gamma for linear kernel as it's not used
        # GridSearchCV handles this, but being explicit can avoid warnings/errors sometimes
        # Alternatively, provide separate grids per kernel in GridSearchCV

    # Cross-validation strategy (e.g., Stratified K-Fold for classification)
    # Using n_splits=min(cv_folds, np.min(np.bincount(y_train))) for small datasets
    n_splits = min(cv_folds, np.min(np.bincount(y_train)))
    if n_splits < 2:
        print(f"Warning: Not enough samples in the smallest class for {n_splits}-fold CV. Using 2 folds.")
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    # Grid Search with Cross-Validation
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

    print(f"Running GridSearchCV for {modality_name}...")
    try:
         grid_search.fit(X_train_scaled, y_train)
    except ValueError as e:
        print(f"ERROR during GridSearchCV for {modality_name}: {e}")
        print("This might happen if a class has fewer samples than CV folds.")
        print("Returning None for this modality.")
        return None, None # Return None for model and scaler


    print(f"Best parameters for {modality_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {modality_name}: {grid_search.best_score_:.4f}")

    best_svm = grid_search.best_estimator_

    return best_svm, scaler # Return the best model and the scaler used

# --- Late Fusion Model Wrapper ---
class LateFusionSVM(nn.Module):
    def __init__(self, svm_audio, scaler_audio, svm_video, scaler_video, svm_pers, scaler_pers, fusion_strategy='average_proba'):
        super().__init__()
        # Store the PRE-TRAINED scikit-learn models and scalers
        self.svm_audio = svm_audio
        self.scaler_audio = scaler_audio
        self.svm_video = svm_video
        self.scaler_video = scaler_video
        self.svm_pers = svm_pers
        self.scaler_pers = scaler_pers

        # Check if models exist
        if self.svm_audio is None or self.svm_video is None or self.svm_pers is None:
            print("Warning: One or more SVM models are None. Predictions might fail.")

        self.fusion_strategy = fusion_strategy # e.g., 'average_proba', 'majority_vote'

        # Determine number of classes from one of the fitted SVMs
        self.n_classes = 0 # Default
        if self.svm_audio and hasattr(self.svm_audio, 'classes_'):
             self.n_classes = len(self.svm_audio.classes_)
        elif self.svm_video and hasattr(self.svm_video, 'classes_'):
             self.n_classes = len(self.svm_video.classes_)
        elif self.svm_pers and hasattr(self.svm_pers, 'classes_'):
             self.n_classes = len(self.svm_pers.classes_)
        else:
             print("Warning: Cannot determine number of classes from fitted SVMs. Check if training failed.")
             # You might need to pass n_classes as an argument if training can fail reliably
             # self.n_classes = default_n_classes # e.g. 3 passed to init

    # REMOVE or COMMENT OUT the aggregate_features method from THIS class definition.
    # It should not be called within the forward pass here.
    # def aggregate_features(self, features):
    #     ...

    def forward(self, A_feat, V_feat, P_feat):
        """
        Performs late fusion prediction.
        Assumes input features are BATCHES of ALREADY AGGREGATED tensors [batch_size, feature_dim].
        """
        # --- Input Handling ---
        if not isinstance(A_feat, torch.Tensor): A_feat = torch.tensor(A_feat, dtype=torch.float32)
        if not isinstance(V_feat, torch.Tensor): V_feat = torch.tensor(V_feat, dtype=torch.float32)
        if not isinstance(P_feat, torch.Tensor): P_feat = torch.tensor(P_feat, dtype=torch.float32)

        # --- Convert to NumPy (Inputs should be 2D: [batch_size, feature_dim]) ---
        # DO NOT aggregate features here again.
        A_feat_np = A_feat.cpu().numpy()
        V_feat_np = V_feat.cpu().numpy()
        P_feat_np = P_feat.cpu().numpy()

        # Add checks to ensure inputs are 2D before scaling
        if A_feat_np.ndim != 2: raise ValueError(f"Expected 2D audio features in forward, got {A_feat_np.ndim}D")
        if V_feat_np.ndim != 2: raise ValueError(f"Expected 2D video features in forward, got {V_feat_np.ndim}D")
        if P_feat_np.ndim != 2: raise ValueError(f"Expected 2D personalized features in forward, got {P_feat_np.ndim}D")

        # --- Scaling ---
        # Apply the SAME scaler used during training
        if self.scaler_audio:
             A_feat_scaled = self.scaler_audio.transform(np.nan_to_num(A_feat_np))
        else:
             A_feat_scaled = np.nan_to_num(A_feat_np) # Use unscaled if scaler missing

        if self.scaler_video:
             V_feat_scaled = self.scaler_video.transform(np.nan_to_num(V_feat_np))
        else:
             V_feat_scaled = np.nan_to_num(V_feat_np) # Use unscaled

        if self.scaler_pers:
             P_feat_scaled = self.scaler_pers.transform(np.nan_to_num(P_feat_np))
        else:
             P_feat_scaled = np.nan_to_num(P_feat_np) # Use unscaled


        # --- Individual Predictions ---
        batch_size = A_feat_scaled.shape[0]
        # Ensure default proba has correct shape [batch_size, n_classes]
        # Handle case where self.n_classes might be 0 if init failed
        n_classes_to_use = self.n_classes if self.n_classes > 0 else 2 # Default guess if needed
        default_proba = np.ones((batch_size, n_classes_to_use)) / n_classes_to_use

        # Use .copy() for default_proba to avoid modifying it if used multiple times
        proba_a = self.svm_audio.predict_proba(A_feat_scaled) if self.svm_audio else default_proba.copy()
        proba_v = self.svm_video.predict_proba(V_feat_scaled) if self.svm_video else default_proba.copy()
        proba_p = self.svm_pers.predict_proba(P_feat_scaled) if self.svm_pers else default_proba.copy()

        # --- Fusion ---
        if self.fusion_strategy == 'average_proba':
            # Check shapes before averaging
            if not (proba_a.shape[0] == proba_v.shape[0] == proba_p.shape[0] == batch_size):
                 raise RuntimeError(f"Batch size mismatch in probabilities: a:{proba_a.shape}, v:{proba_v.shape}, p:{proba_p.shape}")
            if not (proba_a.shape[1] == proba_v.shape[1] == proba_p.shape[1]):
                # If n_classes differ (e.g., one model failed training), this will break.
                # A robust way might be needed, e.g., skip the failing model in the average.
                 print(f"Warning: Number of classes mismatch in probabilities: a:{proba_a.shape}, v:{proba_v.shape}, p:{proba_p.shape}")
                 # Simple fix: Use only models that succeeded (assuming at least one did)
                 valid_probas = [p for p in [proba_a, proba_v, proba_p] if p.shape[1] == n_classes_to_use]
                 if not valid_probas:
                      final_proba = default_proba # Fallback if all failed
                 else:
                     final_proba = np.sum(valid_probas, axis=0) / len(valid_probas)
            else:
                 # Normal case: all shapes match
                 final_proba = (proba_a + proba_v + proba_p) / 3.0

            final_pred = np.argmax(final_proba, axis=1)

        elif self.fusion_strategy == 'majority_vote':
             # Get class predictions
             pred_a = self.svm_audio.predict(A_feat_scaled) if self.svm_audio else np.full(batch_size, -1)
             pred_v = self.svm_video.predict(V_feat_scaled) if self.svm_video else np.full(batch_size, -1)
             pred_p = self.svm_pers.predict(P_feat_scaled) if self.svm_pers else np.full(batch_size, -1)

             preds = np.stack([pred_a, pred_v, pred_p], axis=1)
             final_pred = []
             for row_preds in preds:
                 valid_preds = row_preds[row_preds != -1]
                 if len(valid_preds) == 0:
                     final_pred.append(0) # Default class prediction
                 else:
                     counts = np.bincount(valid_preds.astype(int)) # Ensure int for bincount
                     final_pred.append(np.argmax(counts))
             final_pred = np.array(final_pred)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Return predictions as a torch tensor
        return torch.tensor(final_pred, dtype=torch.long)
    
def report_svm_parameters(svm_model, scaler, modality_name):
    """Prints relevant information about a trained SVM and its scaler."""
    if svm_model is None or scaler is None:
        print(f"\n--- {modality_name} Model (Not Trained) ---")
        return

    print(f"\n--- {modality_name} Model ---")

    # Scaler Info
    n_features = scaler.n_features_in_
    total_scaler_params = 2 * n_features
    print(f"Scaler:")
    print(f"  - Input Features: {n_features}")
    print(f"  - Total parameters (means + scales): {total_scaler_params}")
    # print(f"  - Means shape: {scaler.mean_.shape}") # Uncomment if needed
    # print(f"  - Scales shape: {scaler.scale_.shape}") # Uncomment if needed

    # SVM Info
    print(f"SVM (SVC):")
    print(f"  - Kernel: {svm_model.kernel}")
    print(f"  - C: {svm_model.C}")
    if svm_model.kernel == 'rbf':
        print(f"  - Gamma: {svm_model.gamma}")
    print(f"  - Number of classes: {len(svm_model.classes_)}")
    print(f"  - Support vectors per class: {svm_model.n_support_}")
    total_sv = sum(svm_model.n_support_)
    print(f"  - Total support vectors: {total_sv}")

    # Calculate memory footprint elements
    sv_elements = svm_model.support_vectors_.size if hasattr(svm_model, 'support_vectors_') else 0
    dc_elements = svm_model.dual_coef_.size if hasattr(svm_model, 'dual_coef_') else 0
    ic_elements = svm_model.intercept_.size if hasattr(svm_model, 'intercept_') else 0
    total_svm_elements = sv_elements + dc_elements + ic_elements

    print(f"  - Support Vectors array shape: {svm_model.support_vectors_.shape}")
    print(f"  - Dual Coefficients array shape: {svm_model.dual_coef_.shape}")
    print(f"  - Intercepts array shape: {svm_model.intercept_.shape}")
    print(f"  - Total stored elements (SVs + Dual Coefs + Intercepts): {total_svm_elements}")

# --- Example Usage (after training) ---
# Assuming svm_audio, scaler_audio, etc., are your trained objects

# report_svm_parameters(svm_audio, scaler_audio, "Audio")
# report_svm_parameters(svm_video, scaler_video, "Video")
# report_svm_parameters(svm_pers, scaler_pers, "Personalized")

# total_elements = 0
# if svm_audio and scaler_audio:
#     total_elements += (2 * scaler_audio.n_features_in_) + svm_audio.support_vectors_.size + svm_audio.dual_coef_.size + svm_audio.intercept_.size
# if svm_video and scaler_video:
#      total_elements += (2 * scaler_video.n_features_in_) + svm_video.support_vectors_.size + svm_video.dual_coef_.size + svm_video.intercept_.size
# if svm_pers and scaler_pers:
#      total_elements += (2 * scaler_pers.n_features_in_) + svm_pers.support_vectors_.size + svm_pers.dual_coef_.size + svm_pers.intercept_.size

# print(f"\n--- Overall Model Complexity ---")
# print(f"Total stored elements across all scalers and SVMs: {int(total_elements)}") # Use int() for cleaner output