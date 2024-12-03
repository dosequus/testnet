from torch.utils.data import Dataset
import torch
import tables

class PuzzleDataset(Dataset):
    def __init__(self, hdf5_path, split="train", train_ratio=0.8):
        """
        A PyTorch Dataset for chess puzzles stored in HDF5 format.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            split (str): "train" or "val".
            train_ratio (float): Ratio of data used for training.
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.train_ratio = train_ratio
        
        # Load the dataset and determine split indices
        with tables.open_file(hdf5_path, mode="r") as h5file:
            self.dataset_size = h5file.root.states.shape[0]
        split_index = int(self.dataset_size * train_ratio)
        self.indices = range(0, split_index) if split == "train" else range(split_index, self.dataset_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index in the HDF5 file
        actual_idx = self.indices[idx]
        with tables.open_file(self.hdf5_path, mode="r") as h5file:
            state = torch.tensor(h5file.root.states[actual_idx])
            mask = torch.tensor(h5file.root.masks[actual_idx], dtype=torch.float32)
            policy = torch.tensor(h5file.root.policies[actual_idx], dtype=torch.float32)
            value = torch.tensor(h5file.root.values[actual_idx], dtype=torch.float32)
            rating = torch.tensor(h5file.root.ratings[actual_idx])
        return state, mask, policy, value, rating