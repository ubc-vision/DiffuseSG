import os

import torch
from torch import distributed as dist
from torch.utils.data import DistributedSampler, DataLoader, random_split


def split_test_set(test_dl, total_samples, batch_size, dist_helper, seed=None):
    """
    Split the testing dataset to match the number of samples to be generated.
    """
    # dataset_select, dataset_discard = None, None
    if total_samples < len(test_dl.dataset):
        # to generate fewer samples than the test set, we can just randomly select a subset of the test set
        split_seed = 42 if seed is None else seed
        dataset_select, dataset_discard = random_split(test_dl.dataset, [total_samples, len(test_dl.dataset) - total_samples],
                                                       generator=torch.Generator().manual_seed(split_seed))
    elif total_samples == len(test_dl.dataset):
        # to generate the same number of samples as the test set, we can just use the test set
        dataset_select = test_dl.dataset
    else:
        # to generate more samples than the test set, we need to repeat the test set
        _num_residue = total_samples % len(test_dl.dataset)
        _num_repeat = total_samples // len(test_dl.dataset)
        if _num_residue == 0:
            dataset_select = torch.utils.data.ConcatDataset([test_dl.dataset] * _num_repeat)
        else:
            _num_repeat = total_samples // len(test_dl.dataset)
            dataset_residue, _ = random_split(test_dl.dataset, [_num_residue, len(test_dl.dataset) - _num_residue], generator=torch.Generator().manual_seed(42))
            dataset_select = torch.utils.data.ConcatDataset([test_dl.dataset] * _num_repeat + [dataset_residue])

    if dist_helper.is_ddp:
        sampler = DistributedSampler(dataset_select)
        batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
        sampler_dl = DataLoader(dataset_select, sampler=sampler, batch_size=batch_size_per_gpu,
                                pin_memory=False, num_workers=min(6, os.cpu_count()))
    else:
        sampler_dl = DataLoader(dataset_select, batch_size=batch_size, shuffle=False,
                                pin_memory=False, num_workers=min(6, os.cpu_count()))

    return sampler_dl
