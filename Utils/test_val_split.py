import json
import random

from sklearn.base import defaultdict


def train_val_split1(file_path, val_ratio=0.1, random_seed=3407):
    """
    Track1 dataset split

    Read the JSON file and split it into training and validation sets according to the specified rules:
      - Data with label=4 are split in a 2:1 ratio;
      - Data with label=3 and id=69 are placed directly into the validation set;
      - The remaining data are split according to val_ratio.
    
    Ensure that:
      - Samples with the same ID do not appear in both the training and validation sets.
      - The return format is consistent with train_val_split.

    Parameters:
        file_path (str): path to the JSON data file
        val_ratio (float): proportion of the validation set, default is 0.1
        random_seed (int): random seed, default is 3407

    Returns:
        tuple: (training data list, validation data list, training set category counts, validation set category counts)
    """
    random.seed(random_seed)

    with open(file_path, 'r') as file:
        data = json.load(file)

    train_data, val_data = [], []
    label_to_ids = defaultdict(set)
    id_to_samples = defaultdict(list)

    for item in data:
        pen_category = item["pen_category"]
        id_ = item["id"]
        label_to_ids[pen_category].add(id_)
        id_to_samples[id_].append(item)

    train_ids, val_ids = set(), set()

    for pen_category, ids in label_to_ids.items():
        ids = list(ids)

        # Process label=4 (split in a 2:1 ratio)
        if pen_category == 4:
            for id_ in ids:
                samples = id_to_samples[id_]
                if len(samples) >= 3:
                    random.shuffle(samples)
                    train_data.extend(samples[:2])
                    val_data.extend(samples[2:3])
                else:
                    train_data.extend(samples)
            continue

        # Process the case of label=3 and id=69
        if pen_category == 3:
            for id_ in ids:
                if id_ == "69":  # ID 87 directly placed into the validation set
                    val_data.extend(id_to_samples[id_])
                else:
                    train_data.extend(id_to_samples[id_])
            continue

        # Other categories are randomly split according to the proportion
        random.shuffle(ids)
        split_index = int(len(ids) * (1 - val_ratio))
        train_ids.update(ids[:split_index])
        val_ids.update(ids[split_index:])

    # Split data based on ID
    for id_ in train_ids:
        train_data.extend(id_to_samples[id_])
    for id_ in val_ids:
        val_data.extend(id_to_samples[id_])

    # Calculate category statistics
    train_category_count = defaultdict(int)
    val_category_count = defaultdict(int)

    for entry in train_data:
        train_category_count[entry['pen_category']] += 1
    for entry in val_data:
        val_category_count[entry['pen_category']] += 1

    return train_data, val_data, train_category_count, val_category_count

def train_val_split2(file_path, val_percentage=0.10, seed=None):
    """
    Track2 dataset split

    Group data by person ID from the given JSON file and select 10% of the person IDs, ensuring that the proportion of each label remains consistent with the original data. For each person, either all their data is selected or none is selected.

    Parameters:
    - file_path: path to the JSON file
    - val_percentage: proportion of data to select (based on the number of person IDs, default is 10%)
    - seed: random seed

    Returns:
    - train_data: training set
    - val_data: validation set
    - train_category_count: total count of tri_category labels in the training set
    - val_category_count: total count of tri_category labels in the validation set
    """

    if seed is not None:
        random.seed(seed)

    with open(file_path, 'r') as file:
        data = json.load(file)

    grouped_by_person = defaultdict(list)
    for entry in data:
        # Extract person ID (assumes format "personID_topicID.npy")
        person_id = entry['audio_feature_path'].split('_')[0]
        grouped_by_person[person_id].append(entry)

    # Evenly distribute persons based on label category (young dataset is split according to tri_category)
    tri_category_person = defaultdict(list)
    for person_id, entries in grouped_by_person.items():
        tri_category = entries[0]['tri_category']
        tri_category_person[tri_category].append(person_id)

    total_person_count = len(grouped_by_person)
    num_persons_to_select = round(total_person_count * val_percentage)

    selected_person_ids = set()

    # Calculate the number of persons per category and the number to be selected
    selected_per_category = defaultdict(int)
    for category, person_ids in tri_category_person.items():
        num_category_persons = len(person_ids)
        num_category_to_select = round(num_category_persons * val_percentage + 0.001)
        selected_per_category[category] = num_category_to_select

    for category, person_ids in tri_category_person.items():
        num_category_to_select = selected_per_category[category]
        selected_person_ids.update(random.sample(person_ids, num_category_to_select))

    # Build the validation set data
    val_data = []
    for entry in data:
        person_id = entry['audio_feature_path'].split('_')[0]
        if person_id in selected_person_ids:
            val_data.append(entry)

    # Training set
    train_data = [entry for entry in data if entry not in val_data]

    # Count the total number of tri_category labels in train_data and val_data
    train_category_count = defaultdict(int)
    val_category_count = defaultdict(int)

    for entry in train_data:
        train_category_count[entry['tri_category']] += 1

    for entry in val_data:
        val_category_count[entry['tri_category']] += 1
    # Save train_data and val_data to JSON file (if needed)


    return train_data, val_data, train_category_count, val_category_count
