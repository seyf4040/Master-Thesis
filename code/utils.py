import os
import zipfile
import pandas as pd
from datasets import load_dataset 


def download_huggingface_dataset(dataset_name, split='train', save_path='data', cache_dir='hf_cache'):
    """
    Downloads a dataset from Hugging Face's datasets library and saves it to a specified path.
    
    Parameters:
    - dataset_name: str, the name of the dataset (e.g., 'KoalaAI/Text-Moderation-v2-small')
    - split: str, the dataset split to load (e.g., 'train', 'test', etc.)
    - save_path: str, path to save the dataset as a CSV file
    - cache_dir: str, path to cache the Hugging Face dataset (default: 'hf_cache')
    
    Returns:
    - dataset: a Hugging Face Dataset object
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    # Convert to Pandas DataFrame for easier manipulation if needed
    df = dataset.to_pandas()
    
    # Save the dataset as a CSV file
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'{dataset_name.replace("/", "_")}_{split}.csv')
    df.to_csv(file_path, index=False)
    
    print(f'Dataset "{dataset_name}" ({split} split) downloaded and saved to {file_path}')
    
    return df

def load_data(train_file, test_file, labels_file=None):
    """
    Load train, test, and optional test labels CSV files using pandas.
    
    Parameters:
    - train_file: str, path to the training CSV file
    - test_file: str, path to the test CSV file
    - labels_file: str, path to the test labels CSV file (optional)
    
    Returns:
    - train_data: pandas DataFrame, the training data
    - test_data: pandas DataFrame, the test data
    - test_labels: pandas DataFrame or None, the test labels data if provided
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    test_labels = None
    if labels_file:
        test_labels = pd.read_csv(labels_file)
    
    return train_data, test_data, test_labels

def filter_invalid_labels(test, test_labels, label_column='toxic'):
    """
    Filters out rows with invalid labels from `test_labels` and synchronizes `test` DataFrame accordingly.
    
    Parameters:
    - test: pandas DataFrame, the test dataset
    - test_labels: pandas DataFrame, the labels dataset
    - label_column: str, the column in `test_labels` to filter on (default: 'toxic')
    
    Returns:
    - filtered_test: pandas DataFrame, the filtered test dataset
    - filtered_test_labels: pandas DataFrame, the filtered labels dataset
    """
    # Step 1: Filter out rows where `label_column` is -1
    filtered_test_labels = test_labels[test_labels[label_column] != -1]

    # Step 2: Get the corresponding IDs that remain after filtering
    valid_ids = filtered_test_labels['id']

    # Step 3: Filter `test` DataFrame to keep only rows with IDs in `valid_ids`
    filtered_test = test[test['id'].isin(valid_ids)]

    # Now, `filtered_test` and `filtered_test_labels` are synchronized
    return filtered_test, filtered_test_labels

def preprocess_data(train_file, test_file, labels_file=None):
    """
    Loads and preprocesses data, including filtering out rows with invalid labels.
    
    Parameters:
    - train_file: str, path to the training CSV file
    - test_file: str, path to the test CSV file
    - labels_file: str, path to the test labels CSV file (optional)
    
    Returns:
    - train_data: pandas DataFrame, the training data
    - filtered_test: pandas DataFrame, the filtered test dataset
    - filtered_test_labels: pandas DataFrame or None, the filtered labels dataset if provided
    """
    # Load data
    train_data, test_data, test_labels = load_data(train_file, test_file, labels_file)
    
    # If test labels are provided, filter invalid labels
    if test_labels is not None:
        test_data, test_labels = filter_invalid_labels(test_data, test_labels)
    
    return train_data, test_data, test_labels


def filter_labels_with_threshold(test_labels, threshold=0.5):
    """
    Filters the `test_labels` DataFrame by keeping the majority annotation for each `rev_id` 
    based on a custom threshold.

    Parameters:
    - test_labels: pandas DataFrame containing the labels with `rev_id` and `attack` columns.
    - threshold: float between 0 and 1. The fraction of annotations that need to be of the same type 
                 (either 0.0 or 1.0) to be considered the valid.

    Returns:
    - A DataFrame with filtered labels, considering the threshold for valid voting.
    """

    # Step 1: Create a function to apply the threshold logic to each rev_id group
    def apply_threshold(group):
        # Count occurrences of each label (0.0 and 1.0)
        count_1 = (group['attack'] == 1.0).sum()

        total = len(group)
        
        # Determine majority based on threshold
        if count_1 / total >= threshold:
            return 1.0
        else:
            return 0.0  

    # Step 2: Apply the threshold logic to each group of rev_id
    filtered_labels = test_labels.groupby('rev_id').apply(apply_threshold).reset_index(name='attack')

    # Step 3: Merge with the original `test_labels` to keep other columns (optional)
    # filtered_labels = pd.merge(filtered_labels, test_labels[['rev_id', 'attack']], on=['rev_id', 'attack'], how='left')

    return filtered_labels



