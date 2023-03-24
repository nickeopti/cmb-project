import math
import os.path
import pickle
import torch
from typing import Any, Literal

import cc3d
import nibabel
import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223]


def get_file_data(base_path: str, file_name: str):
    image_path = os.path.join(base_path, 'imagesTs', f'{file_name}_0000.nii.gz')
    label_path = os.path.join(base_path, 'labelsTs', f'{file_name}.nii.gz')
    meta_data_path = os.path.join(base_path, 'pred_and_softmax', f'{file_name}.pkl')
    prediction_path = os.path.join(base_path, 'pred_and_softmax', f'{file_name}.npz')

    image = nibabel.load(image_path)
    label = nibabel.load(label_path)
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    prediction = np.load(prediction_path)['softmax'][1]

    return image, label, meta_data, prediction


def get_cropped_data(
        image: nibabel.nifti1.Nifti1Image,
        label: nibabel.nifti1.Nifti1Image,
        meta_data: list[dict[str, Any]],
):
    image_data = image.get_fdata().T
    label_data = label.get_fdata().T

    bbox = meta_data[0]['crop_bbox']

    cropped_image_data = image_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    cropped_label_data = label_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

    return cropped_image_data, cropped_label_data


def get_components(data: np.ndarray, connectivity: Literal[6, 18, 26] = 18, return_N: bool = False):
    components = cc3d.connected_components(data, connectivity=connectivity, return_N=return_N)
    return components


def get_boxes(components: np.ndarray, meta_data: nibabel.nifti1.Nifti1Header, size: int = 20):
    stats = cc3d.statistics(components)
    centroids = stats['centroids'][1:].round().astype(int)

    zooms = meta_data.get_zooms()[::-1]
    extents = tuple([math.ceil(size / zoom / 2) for zoom in zooms])

    boxes = [
        (
            slice(a - extents[0], a + extents[0]),
            slice(b - extents[1], b + extents[1]),
            slice(c - extents[2], c + extents[2])
        )
        for a, b, c in centroids
    ]

    return boxes


def is_box_valid(image, box):
    if isinstance(image, np.ndarray):
        shape = image.shape
    elif isinstance(image, nibabel.nifti1.Nifti1Image):
        shape = image.header.get_data_shape()
    elif isinstance(image, nibabel.nifti1.Nifti1Header):
        shape = image.get_data_shape()
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            shape = image.shape
        elif image.dim() == 4:
            shape = image.shape[1:]
        else:
            raise ValueError
    else:
        return ValueError

    return not any(
        s.start < 0 or s.stop > shape[i]
        for i, s in enumerate(box)
    )


def extract_image_box(image, box):
    if not is_box_valid(image, box):
        return None

    return image[(..., *box)]


def compute_object_metrics(labels_components, predictions_components):
    # labels_components, n_labels = get_components(labels, return_N=True)
    # predictions_components, n_predictions = get_components(predictions, return_N=True)
    n_labels = len(np.unique(labels_components)) - 1
    n_predictions = len(np.unique(predictions_components)) - 1
    labels_components_ones = labels_components.copy()
    labels_components_ones[labels_components_ones != 0] = 1

    for p in range(n_labels, 0, -1):
        labels_components[labels_components == p] = PRIMES[p]
    for p in range(n_predictions, 0, -1):
        predictions_components[predictions_components == p] = PRIMES[n_labels + p]
    
    unique = np.unique(labels_components * predictions_components)
    unique = unique[unique != 0]
    
    n_tp = sum(np.any(unique % PRIMES[p] == 0) for p in range(n_labels))
    n_fn = n_labels - n_tp
    # n_fp = n_predictions - len(unique)
    n_fp = n_predictions - len(np.unique(predictions_components * labels_components_ones)) + 1

    return n_tp, n_fp, n_fn, n_labels
