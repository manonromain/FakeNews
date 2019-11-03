from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sequential_data):
        """
        Args: list of tensors, each tensor is a sequence of tweet_node embeddings
        """
        self.data = sequential_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
