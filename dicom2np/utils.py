from collections import defaultdict, Counter
from pathlib import Path
from typing import List

import pydicom
import numpy as np
from tqdm import tqdm


class ApplicationError(Exception):
    pass


def dicom2np(dcm: pydicom.Dataset) -> np.ndarray:
    r = dcm.pixel_array

    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        sc = dcm.WindowCenter
        sw = dcm.WindowWidth

        mx = sc + sw / 2
        mn = sc - sw / 2

        r = 255 * (r - mn) / (mx - mn)

        r[r > 255] = 255
        r[r < 0] = 0

    return r


def process_one(path: Path, study_instance_uid: str) -> pydicom.Dataset:
    dcm = pydicom.dcmread(path)

    # no study instance uid
    if not hasattr(dcm, 'StudyInstanceUID'):
        raise ApplicationError(f'no study instance uid ({path})')

    if study_instance_uid is not None and study_instance_uid != dcm.StudyInstanceUID:
        raise ApplicationError(f'study instance id inconsistent ({path}): {study_instance_uid} and {dcm.StudyInstanceUID}')

    # no pixel array
    if not hasattr(dcm, 'pixel_array'):
        raise ApplicationError(f'no pixel array ({path})')

    # no series number
    if not hasattr(dcm, 'SeriesNumber'):
        raise ApplicationError(f'no series number ({path})')

    # no slice index
    if not hasattr(dcm, 'InstanceNumber'):
        raise ApplicationError(f'no instance number ({path})')

    # no rescale intercept
    if not hasattr(dcm, 'RescaleIntercept'):
        raise ApplicationError(f'no rescale intercept ({path})')

    # no rescale slope
    if not hasattr(dcm, 'RescaleSlope'):
        raise ApplicationError(f'no rescale slope ({path})')
    
    return dcm



def process_study(paths: List[Path],
                  save_dir: Path,
                  lower: int,
                  upper: int,
                  ) -> None:
    """process one 4D-CT image"""
    dcms = defaultdict(dict)

    study_instance_uid = None
    for _, path in enumerate(paths):
        try:
            dcm = process_one(path)
        except Exception as e:
            tqdm.write(f'{e}')
            continue

        study_instance_uid = dcm.StudyInstanceUID

        dcms[int(dcm.SeriesNumber)][int(dcm.InstanceNumber) - 1] = dcm

    counts = Counter(len(instances) for _, instances in dcms.items())
    max_counts = counts.most_common(n=1)[0][1]
    if max_counts < 10:
        # num of instances == 10
        raise ApplicationError(f'num of instances = {max_counts} < 10 ({study_instance_uid})')

    dcms = {series: instances
            for series, instances in dcms.items()
            if counts[len(instances)] == max_counts}
    if len(dcms) < 10:
        raise ApplicationError('unknown error')

    xs = []
    size = None
    for _, instances in sorted(dcms.items(), key=lambda x: x[0])[-10:]:
        if size is not None and size != len(instances):
            raise ApplicationError(f'size mismatch ({study_instance_uid})')
        size = len(instances)

        data = []
        for _, instance in sorted(instances.items(), key=lambda x: x[0]):
            rs = instance.RescaleSlope
            ri = instance.RescaleIntercept
            x = instance.pixel_array
            x = x * rs + ri
            data.append(x)

        x = np.stack(data)

        min_val = x.min()
        x[(x < lower) | (x > upper)] = min_val

        x = 255 * (x - min_val) / (upper - min_val)
        xs.append(x)

    x = np.stack(xs)
    np.savez_compressed(save_dir / f'{study_instance_uid}.npz', x)

    return


def process(paths: List[Path],
            save_dir: Path,
            lower: int,
            upper: int,
            ) -> None:
    studies = defaultdict(list)

    print('preparse files...')
    for path in tqdm(paths):
        dcm = pydicom.dcmread(path)
        studies[dcm.StudyInstanceUID].append(path)

    print('process studies...')
    for study, paths in tqdm(studies.items()):
        try:
            process_study(paths, save_dir, lower, upper)
        except Exception as e:
            tqdm.write(f'[{study}] {e}')

    return
