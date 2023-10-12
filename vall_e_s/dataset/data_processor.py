import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from dataset.files_dataset import FilesAudioDataset
from dataset.utils import calculate_bandwidth
from dataset.collate import collate
from dataset.sampler import DynamicBatchSampler

class OffsetDataset(Dataset):
    def __init__(self, dataset, start, end, test=False):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        self.test = test
        assert 0 <= self.start < self.end <= len(self.dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset.get_item(self.start + item, test=self.test)

class DataProcessor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = FilesAudioDataset(cfg)
        duration = 1 if cfg.prior else 600
        #cfg.bandwidth = calculate_bandwidth(self.dataset, cfg, duration=duration)
        self.create_datasets(cfg)
        self.create_samplers(cfg)
        self.create_data_loaders(cfg)
        self.print_stats(cfg)

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)

    def create_datasets(self, cfg):
        train_len = int(len(self.dataset) * cfg.train_test_split)
        self.train_dataset = OffsetDataset(self.dataset, 0, train_len, test=False)
        self.valid_dataset = OffsetDataset(self.dataset, train_len, len(self.dataset), test=True)

    def create_samplers(self, num_buckets=10, max_duration=120):
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        self.train_sampler = DynamicBatchSampler(train_sampler, self.train_dataset.get_dur,
                                                   num_bukets=num_buckets,
                                                   max_size=20, max_tokens=self.cfg.max_duration)
        valid_sampler = torch.utils.data.RandomSampler(self.test_dataset)
        self.valid_sampler = DynamicBatchSampler(valid_sampler, self.test_dataset.get_dur,
                                                num_buckets=num_buckets,
                                                max_size=20, max_tokens=self.cfg.max_duration)

    def create_data_loaders(self, cfg):
        # Loader to load mini-batches
        if cfg.labels:
            collate_fn = collate

        logging.info('Creating Data Loader')
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.bs, num_workers=cfg.nworkers,
                                       sampler=self.train_sampler, pin_memory=False,
                                       drop_last=True, collate_fn=collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=cfg.bs, num_workers=cfg.nworkers,
                                      sampler=self.test_sampler, pin_memory=False,
                                      drop_last=False, collate_fn=collate_fn)

    def print_stats(self, hps):
        logging.info(f"Train {len(self.train_dataset)} samples. Test {len(self.test_dataset)} samples")
        logging.info(f'Train sampler: {self.train_sampler}')
        logging.info(f'Train loader: {len(self.train_loader)}')