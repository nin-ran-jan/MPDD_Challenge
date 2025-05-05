# How to run it
## Files with everything in a everythin in a script (no imports)
- run_transformer_inter_fusion (For inter fusion transformer with Regularization)
```
python run_transformer_inter_fusion.py
```
- 

# Config explanation (not necesarry to understand)

-  data_root_path: path to your data directory (Elderly or Young)
-  window_split_time: 1 
-  audio_feature_method : Audio feature type, options {wav2vec, opensmile, mfccs}
-  video_feature_method : Video feature type, options {openface, resnet, densenet}
-  labelcount: Number of labels, e.g., 2 for binary classification {1,2,5}
-  track_option: Track option, e.g., "Track1"
-  feature_max_len: Maximum length of features, e.g., 26
-  batch_size: Batch size for training, e.g., 16 (Optional based on your model)
-  lr: Learning rate, e.g., 0.00002 (Optional based on your model)
-  num_epochs: Number of epochs for training, e.g., 200 (Optional Based on your model)
-  device: Device to use for computation, e.g., "mps"
-  seed: random seed