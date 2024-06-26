import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_stats(log_dir, stat_keys, version=None, silent=True):
    # log_dir = f'../lightning_logs/{experiment}/lightning_logs'
    if version is None:
        # Gather latest stats
        versions = os.listdir(log_dir)
        versions = sorted([int(v.split('version_')[1]) for v in versions], reverse=True)
        if len(versions) > 1:
            print(f'Multiple versions in {log_dir}')
        version = versions[0]
        if not silent:
            print(f'Gathering stats for version {version}')

    log_dir = f'{log_dir}/version_{version}'
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    stats = {}
    for k in stat_keys:
        try:
            _, step_nums, vals = zip(*event_acc.Scalars(k))
            stats[k] = np.array(vals)
            stats[k + '_step'] = np.array(step_nums)
        except KeyError:
            print(f'Could not find key {k} in {log_dir}')
            stats[k] = np.array([np.nan])

    return stats
