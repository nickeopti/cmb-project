import enum
import glob
import os.path
import random

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

import util


class InputType(enum.Enum):
    IMAGE = enum.auto()
    PREDICTION = enum.auto()
    COMBINED = enum.auto()

    @classmethod
    def channels(input_type: "InputType"):
        if input is InputType.COMBINED:
            return 2
        else:
            return 1


class CMBDataset(Dataset):
    def __init__(
        self,
        base_path,
        glob_pattern: str = None,
        split_file: str = None,
        split_partition: str = None,
        threshold: float = 0.2,
        size: int = 20,
        input_type: InputType = InputType.IMAGE,
    ) -> None:
        if glob_pattern is None:
            assert split_file is not None
            assert split_partition is not None

            with open(split_file, "r") as f:
                splits = yaml.load(f)
                base_file_names = splits[split_partition]

        if split_file is None or split_partition is None:
            assert glob_pattern is not None

            file_names = glob.glob(os.path.join(base_path, "labelsTs", glob_pattern))
            base_file_names = [
                os.path.basename(file_name)[:-7] for file_name in file_names
            ]

        self.base_path = base_path
        self.base_file_names = base_file_names

        self.threshold = threshold
        self.size = size
        self.input_type = input_type


class LazyBox(CMBDataset):
    def __len__(self) -> int:
        return len(self.base_file_names)

    def __getitem__(self, index):
        base_file_name = self.base_file_names[index]
        image, label, meta_data, prediction = util.get_file_data(
            self.base_path, base_file_name
        )

        cropped_image, cropped_label = util.get_cropped_data(image, label, meta_data)

        components = util.get_components(prediction >= self.threshold)
        boxes = util.get_boxes(components, image.header, self.size)
        boxes = [box for box in boxes if util.is_box_valid(image, box)]

        if not boxes:
            return self[(index + 1) % len(self)]

        box = boxes[random.randrange(0, len(boxes))]
        label = np.any(cropped_label[box] > 0)
        image = cropped_image[box]
        prediction = cropped_image[box]

        if self.input_type is InputType.IMAGE:
            x = image[None]
        elif self.input_type is InputType.PREDICTION:
            x = prediction[None]
        elif self.input_type is InputType.COMBINED:
            x = np.stack((image, prediction))
        else:
            raise RuntimeError

        return torch.from_numpy(x).float(), float(label)


class LazyBoxVal(CMBDataset):
    def __len__(self) -> int:
        return len(self.base_file_names)

    def __getitem__(self, index):
        base_file_name = self.base_file_names[index]
        image, label, meta_data, prediction = util.get_file_data(
            self.base_path, base_file_name
        )

        cropped_image, cropped_label = util.get_cropped_data(image, label, meta_data)

        components = util.get_components(prediction >= self.threshold)
        components_comp = util.get_components(prediction >= 0.5)
        labels_components = util.get_components(cropped_label)

        boxes = util.get_boxes(components, image.header, size=20)
        boxes = [[(s.start, s.stop) for s in box] for box in boxes]

        if self.input_type is InputType.IMAGE:
            x = cropped_image[None]
        elif self.input_type is InputType.PREDICTION:
            x = prediction[None]
        elif self.input_type is InputType.COMBINED:
            x = np.stack((cropped_image, prediction))
        else:
            raise RuntimeError

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(components.astype(np.int32)),
            torch.from_numpy(components_comp.astype(np.int32)),
            torch.from_numpy(labels_components.astype(np.int32)),
            [list(enumerate(boxes, start=1))],
        )
