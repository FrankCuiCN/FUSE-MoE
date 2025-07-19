import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class FineWebEdu3B(Dataset):
    def __init__(self, data_dir, mode, context_window):
        super().__init__()
        # Initialize attributes
        self.data_dir = data_dir  # Location of the dataset
        self.mode = mode  # "train" or "val"
        self.context_window = context_window  # Sequence length of each sample
        self._context_window = context_window + 1  # Add one for teacher forcing
        # ----- #
        # Select file paths based on mode
        # ----- #
        self.file_paths = []  # Path to each .bin file
        if self.mode == "train":
            # 100 million tokens per file; 1 file reserved for validation
            for idx in range(29):
                file_name = f"finewebedu_train_{(idx + 1):06d}.bin"
                self.file_paths.append(os.path.join(self.data_dir, file_name))
        elif self.mode == "val":
            file_name = "finewebedu_val_000000.bin"
            self.file_paths.append(os.path.join(self.data_dir, file_name))
        else:
            raise ValueError("mode must be 'train' or 'val'")
        # ----- #
        # Load everything into RAM
        # ----- #
        self.tokens = []
        for file_path in tqdm(self.file_paths):
            # Open file in read-binary mode
            with open(file_path, "rb") as f:
                # Skip the first 256*4 bytes (metadata or header)
                f.seek(256 * 4)
                # Read data as 16-bit unsigned integers
                self.tokens.append(np.fromfile(f, dtype=np.uint16))
        self.tokens = np.concatenate(self.tokens)
        self.tokens = torch.from_numpy(self.tokens)  # Expect uint16
        # Calculate how many complete samples can be formed
        self.num_sample = len(self.tokens) // self._context_window
        # Drop last
        self.num_token = self.num_sample * self._context_window
        self.tokens = self.tokens[:self.num_token]
        self.tokens = self.tokens.reshape(self.num_sample, self._context_window)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        # Workaround: uint16 is not compatible with NCCL (for DDP)
        #     so casting to int64 at this stage
        sample = self.tokens[idx]             # (context_window + 1,)
        inputs = sample[:-1].to(torch.int64)  # (context_window,)
        targets = sample[1:].to(torch.int64)  # (context_window,)
        return inputs, targets
