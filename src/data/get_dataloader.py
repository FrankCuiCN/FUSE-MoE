import torch
from torch.utils.data import Dataset, DataLoader
from config.config_template import ConfigTemplate
from data.fineweb_edu_1b import FineWebEdu1B
from data.fineweb_edu_3b import FineWebEdu3B
from data.fineweb_edu_10b import FineWebEdu10B

class DummyDataset(Dataset):
    """Workaround for Accelerate"""
    def __init__(self):
        pass
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return None


def get_dataloader_train(config: ConfigTemplate, dummy=False) -> DataLoader:
    if dummy:
        dataset = DummyDataset()  # Workaround for Accelerate
    else:
        if config.data_name == "FineWebEdu1B":
            dataset = FineWebEdu1B(
                data_dir=config.data_dir,
                mode="train",
                context_window=config.context_window,
            )
        elif config.data_name == "FineWebEdu3B":
            dataset = FineWebEdu3B(
                data_dir=config.data_dir,
                mode="train",
                context_window=config.context_window,
            )
        elif config.data_name == "FineWebEdu10B":
            dataset = FineWebEdu10B(
                data_dir=config.data_dir,
                mode="train",
                context_window=config.context_window,
            )
        else:
            raise ValueError()

    # Ask: what are some considerations for efficient data loading?
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Consider: add this to ConfigTemplate
        num_workers=config.dataloader_num_worker,
        pin_memory=config.dataloader_pin_memory,
        drop_last=True,  # Consider: add this to ConfigTemplate
    )
    return dataloader


def get_dataloader_val(config: ConfigTemplate, dummy=False) -> DataLoader:
    if dummy:
        dataset = DummyDataset()  # Workaround for Accelerate
    else:
        if config.data_name == "FineWebEdu1B":
            dataset = FineWebEdu1B(
                data_dir=config.data_dir,
                mode="val",
                context_window=config.context_window,
            )
        elif config.data_name == "FineWebEdu3B":
            dataset = FineWebEdu3B(
                data_dir=config.data_dir,
                mode="val",
                context_window=config.context_window,
            )
        elif config.data_name == "FineWebEdu10B":
            dataset = FineWebEdu10B(
                data_dir=config.data_dir,
                mode="val",
                context_window=config.context_window,
            )
        else:
            raise ValueError()

    # Note: We evaluate on a random subset from the validation set
    #     Therefore, we set shuffle and drop_last as True
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size_fwd * config.num_gpu,
        shuffle=True,
        num_workers=config.dataloader_num_worker,
        pin_memory=config.dataloader_pin_memory,
        drop_last=True,
    )
    return dataloader
