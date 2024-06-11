import pandas as pd
import torch
import numpy as np
import glob
import re
from tqdm import tqdm
from itertools import combinations
import random

tqdm.pandas()

# Load author references data
author_references_test = pd.read_hdf(
    "/mnt/home/amadovic/neural_author_disambiguator/author_references_nov22nd_v2.h5"
).reset_index()


# Function to extract chunk number from file path
def extract_chunk_number(file_path):
    match = re.search(r"_(\d+)\.pt$", file_path)
    return int(match.group(1)) if match else None


# Function to process embeddings from a directory
def process_embeddings(directory_path, file_pattern):
    files = glob.glob(directory_path + file_pattern)
    sorted_file_paths = sorted(files, key=extract_chunk_number)
    embeddings_list = []

    for file in sorted_file_paths:
        print(file)
        embeddings = torch.tensor(torch.load(file, map_location="cpu"))
        embeddings_list.append(embeddings)

    flattened_embeddings_list = [
        item
        for sublist in embeddings_list
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    concatenated_embeddings = torch.cat(flattened_embeddings_list, dim=0)
    concatenated_np_embeddings = concatenated_embeddings.detach().cpu().numpy()

    return concatenated_np_embeddings


# Process specter embeddings
specter_embeddings = process_embeddings(
    directory_path="/mnt/home/amadovic/neural_author_disambiguator/specter_embeddings/",
    file_pattern="author_embeddings_batch_*",
)

# Process affiliation embeddings
aff_chars2vec = np.load(
    "/mnt/home/amadovic/neural_author_disambiguator/aff_embeddings.npy"
)

# Process author embeddings
author_chars2vec = np.load(
    "/mnt/home/amadovic/neural_author_disambiguator/author_embeddings.npy"
)

# Check shapes of embeddings
specter_embeddings.shape, author_chars2vec.shape, aff_chars2vec.shape


# Function to flatten lists
def flatten_lists(column):
    return column.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)


# Process blocks for hard negatives
def process_blocks(group):
    data_list = []
    for (i_index, i), (j_index, j) in combinations(group.iterrows(), 2):
        data = {
            "index1": i_index,
            "@path": i["@path"],
            "abstract": i["abstract"],
            "author": i["author"],
            "aff": i["aff"],
            "index2": j_index,
            "@path2": j["@path"],
            "abstract2": j["abstract"],
            "author2": j["author"],
            "aff2": j["aff"],
            "label": i["@path"] == j["@path"],
        }
        data_list.append(data)
    if data_list:
        result_df = pd.DataFrame(data_list).apply(flatten_lists)
        return result_df
    else:
        return pd.DataFrame()


group = author_references_test.groupby(["block"])
hard_negatives = group.progress_apply(process_blocks).reset_index(drop=True)


# Process random pairs for easy negatives
def process_random_pairs(author_references, hard_negative_rate, hard_negative_size):
    data_list = []
    for _ in range(int(len(hard_negative_size) * hard_negative_rate)):
        random_index_1 = random.randint(0, len(author_references_test) - 1)
        random_index_2 = random.randint(0, len(author_references_test) - 1)
        while random_index_1 == random_index_2:
            random_index_1 = random.randint(0, len(author_references_test))
        random_row1 = author_references_test.iloc[random_index_1]
        random_row2 = author_references_test.iloc[random_index_2]

        data = {
            "index1": random_index_1,
            "@path": random_row2["@path"],
            "abstract": random_row2["abstract"],
            "author": random_row2["author"],
            "aff": random_row2["aff"],
            "index2": random_index_2,
            "@path2": random_row1["@path"],
            "abstract2": random_row1["abstract"],
            "author2": random_row1["author"],
            "aff2": random_row1["aff"],
            "label": random_row1["@path"] == random_row2["@path"],
        }
        data_list.append(data)
    if data_list:
        result_df = pd.DataFrame(data_list).query("label == False")
        return result_df
    else:
        return pd.DataFrame()


easy_negatives = process_random_pairs(author_references_test, 0.0, hard_negatives)

# Combine hard and easy negatives
author_index_pairs = pd.concat([hard_negatives, easy_negatives])
author_index_pairs.drop_duplicates(inplace=True)

# Balance the dataset
random_state = 11
sample_size_per_class = (
    author_index_pairs.groupby("label")
    .count()
    .query("label == False")["index1"]
    .values[0]
)
class_0_data = author_index_pairs[author_index_pairs["label"] == True]
class_1_data = author_index_pairs[author_index_pairs["label"] == False]

sample_class_0 = class_0_data.sample(n=sample_size_per_class, random_state=random_state)
sample_class_1 = class_1_data.sample(n=sample_size_per_class, random_state=random_state)

balanced_sample = pd.concat([sample_class_0, sample_class_1])
balanced_sample = balanced_sample.sample(frac=1).reset_index(drop=True)

# Prepare embeddings for pairs
auth1_chars2vec = []
auth2_chars2vec = []
aff1_chars2vec = []
aff2_chars2vec = []
specter1_embed = []
specter2_embed = []

for _, row in tqdm(balanced_sample.iterrows()):
    auth1_chars2vec.append(author_chars2vec[int(row["index1"])])
    auth2_chars2vec.append(author_chars2vec[int(row["index2"])])
    aff1_chars2vec.append(aff_chars2vec[int(row["index1"])])
    aff2_chars2vec.append(aff_chars2vec[int(row["index2"])])
    specter1_embed.append(specter_embeddings[int(row["index1"])])
    specter2_embed.append(specter_embeddings[int(row["index2"])])

# Convert lists to numpy arrays
author1_embed = np.concatenate(
    [auth1_chars2vec, aff1_chars2vec, specter1_embed], axis=1
)
author2_embed = np.concatenate(
    [auth2_chars2vec, aff2_chars2vec, specter2_embed], axis=1
)
