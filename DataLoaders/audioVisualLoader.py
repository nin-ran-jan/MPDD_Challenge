from DataSets.audioVisualDataset import AudioVisualDataset
from torch.utils.data import DataLoader

def create_audio_visual_loader(json_data, label_count, personalized_feature_file, max_len=10, batch_size=32, audio_path='', video_path='', isTest=False, shuffle=False, num_workers=0, pin_memory=False) -> DataLoader:
    """
    Create an audio-visual data loader.
    
    Parameters:
    - json_data: JSON data containing the dataset information.
    - label_count: Number of labels in the dataset.
    - personalized_feature_file: Path to the personalized feature file.
    - max_len: Maximum length of the sequences.
    - batch_size: Size of each batch.
    - audio_path: Path to the audio files.
    - video_path: Path to the video files.
    - isTest: Boolean indicating if it's a test set.
    - shuffle: Boolean indicating whether to shuffle the data.
    - num_workers: Number of subprocesses to use for data loading.
    - pin_memory: Boolean indicating whether to use pinned memory.
    
    Returns:
    - DataLoader object for the audio-visual dataset.
    """
    
    # Create an instance of AudioVisualDataset
    dataloader = DataLoader(AudioVisualDataset(
        json_data=json_data,
        label_count=label_count,
        personalized_feature_file=personalized_feature_file,
        max_len=max_len,
        batch_size=batch_size,
        audio_path=audio_path,
        video_path=video_path,
        isTest=isTest), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    
    return dataloader