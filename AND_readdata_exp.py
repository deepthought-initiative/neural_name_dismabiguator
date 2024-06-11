# Import necessary modules
import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import random

# Set a seed for reproducibility
seed = 11

# Also set the seed for numpy and random
np.random.seed(seed)
random.seed(seed)

# Load the test set references
author_references = pd.read_json(
    "/mnt/scratch/amadovic/comparison_datasets/specter_chars2vec_labels_test.json"
)
author2_embed = np.load(
    "/mnt/scratch/amadovic/comparison_datasets/specter_chars2vec_author1_embed.npy"
)
author1_embed = np.load(
    "/mnt/scratch/amadovic/comparison_datasets/specter_chars2vec_author2_embed.npy"
)

# Ensure indices are shuffled to maintain randomness
n_samples = len(author_references)
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Split the shuffled indices into train, validation, and test sets
train_split = int(0.7 * n_samples)
val_split = int(0.2 * n_samples)
test_split = n_samples - train_split - val_split

train_indices = indices[:train_split]
val_indices = indices[train_split : train_split + val_split]

# Shuffle the indices again before splitting for the test set
np.random.shuffle(indices)
test_indices = indices[train_split + val_split :]


# Define a custom IterableDataset for handling large datasets in chunks
class ChunkedDataset(IterableDataset):
    def __init__(self, indices, chunk_size, shuffle_train, seed=0):
        self.indices = indices
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.total_samples = len(indices)
        self.chunk_idx = 0
        self.start_idx = 0
        self.seed = seed if seed is not None else 0
        self.shuffle_train = shuffle_train
        self.epoch = 0  # Initialize epoch counter
        if self.shuffle_train:
            self.shuffle_indices()  # Initial shuffle if needed
        print(self.total_samples)


def __iter__(self):
    if self.shuffle_train:
        np.random.seed(self.seed + self.epoch)  # Update seed with epoch number
        self.shuffle_indices()  # Shuffle training indices at the start of each epoch
        self.epoch += 1  # Increment epoch counter
    self.chunk_idx = 0  # Reset chunk index for new epoch
    self.start_idx = 0  # Reset start index for new epoch
    return self


def __len__(self):
    return self.total_samples


def __next__(self):
    if self.current_chunk is None or self.chunk_idx >= len(self.current_chunk):
        self.load_next_chunk()

    if self.chunk_idx < len(self.current_chunk):
        sample = self.current_chunk[self.chunk_idx]
        self.chunk_idx += 1
        return sample
    else:
        raise StopIteration


def shuffle_indices(self):
    # Shuffle the indices
    np.random.shuffle(self.indices)


def load_next_chunk(self):
    # Load the next chunk of data
    end_idx = min(self.start_idx + self.chunk_size, len(self.indices))
    chunk_indices = self.indices[self.start_idx : end_idx]

    # Ensure that chunk_indices are correctly mapped across the datasets
    chunk_author1_embed = author1_embed[chunk_indices]
    chunk_author2_embed = author2_embed[chunk_indices]
    chunk_labels = author_references.label.iloc[chunk_indices].values

    # Shuffle chunk data internally
    indices = np.arange(len(chunk_indices))
    np.random.shuffle(indices)
    chunk_author1_embed = chunk_author1_embed[indices]
    chunk_author2_embed = chunk_author2_embed[indices]
    chunk_labels = chunk_labels[indices]

    self.current_chunk = TensorDataset(
        torch.tensor(
            np.stack([chunk_author1_embed, chunk_author2_embed], axis=1),
            dtype=torch.float32,
        ),
        torch.tensor(chunk_labels, dtype=torch.bool),
    )

    # Free up memory
    del chunk_author1_embed
    del chunk_author2_embed
    del chunk_labels

    self.chunk_idx = 0
    self.start_idx = end_idx


# Define a PyTorch Lightning DataModule for managing data loaders
class PairsANDDataModule2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        chunk_size: int = 100000,
        shuffle_train: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle_train = shuffle_train

    def setup(self, stage: str = None):
        # Setup datasets for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_dataset = ChunkedDataset(
                train_indices, self.chunk_size, shuffle_train=True, seed=0
            )
            self.val_dataset = ChunkedDataset(
                val_indices, self.chunk_size, shuffle_train=False, seed=None
            )
        if stage == "test" or stage is None:
            self.test_dataset = ChunkedDataset(
                test_indices, self.chunk_size, shuffle_train=False, seed=None
            )

    def train_dataloader(self):
        # Return a DataLoader for the training set
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Return a DataLoader for the validation set
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True
        )

    def test_dataloader(self):
        # Return a DataLoader for the test set
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )
