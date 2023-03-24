import enum
import glob
import os.path
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import util


class InputType(enum.Enum):
    IMAGE = enum.auto()
    PREDICTION = enum.auto()
    COMBINED = enum.auto()

    @classmethod
    def channels(input_type: 'InputType'):
        if input is InputType.COMBINED:
            return 2
        else:
            return 1


class LazyBox(Dataset):
    def __init__(
        self, base_path: str, glob_pattern: str, threshold: float = 0.2, size: int = 20, input_type: InputType = InputType.IMAGE,
    ) -> None:
        file_names = glob.glob(os.path.join(base_path, "labelsTs", glob_pattern))
        base_file_names = [os.path.basename(file_name)[:-7] for file_name in file_names]

        self.base_path = base_path
        self.base_file_names = base_file_names

        self.threshold = threshold
        self.size = size
        self.input_type = input_type

    def __len__(self) -> int:
        return len(self.base_file_names)

    def __getitem__(self, index):
        base_file_name = self.base_file_names[index]
        image, label, meta_data, prediction = util.get_file_data(self.base_path, base_file_name)

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


class Box(Dataset):
    def __init__(
        self, base_path: str, glob_pattern: str, threshold: float = 0.2
    ) -> None:
        file_names = glob.glob(os.path.join(base_path, "labelsTs", glob_pattern))
        base_file_names = [os.path.basename(file_name)[:-7] for file_name in file_names]

        images, labels, meta_data, predictions = zip(
            *(util.get_file_data(base_path, file_name) for file_name in base_file_names)
        )
        cropped_images, cropped_labels = zip(
            *(
                util.get_cropped_data(image, label, meta)
                for image, label, meta in zip(images, labels, meta_data)
            )
        )
        components = [
            util.get_components(prediction >= threshold) for prediction in predictions
        ]

        self.images = images
        self.labels = labels
        self.cropped_images = cropped_images
        self.cropped_labels = cropped_labels
        self.predictions = predictions
        self.components = components

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        header = self.images[index].header
        cropped_image = self.cropped_images[index]
        cropped_label = self.cropped_labels[index]
        predictions = self.predictions[index]
        components = self.components[index]

        boxes = util.get_boxes(components, header, size=20)
        boxes = [
            box
            for box in boxes
            if not any(
                s.start < 0 or s.stop > cropped_image.shape[i]
                for i, s in enumerate(box)
            )
        ]
        if not boxes:
            return self[(index + 1) % len(self)]
        box = boxes[random.randrange(0, len(boxes))]

        label = np.any(cropped_label[box] > 0)
        image = cropped_image[box]
        prediction = predictions[box]

        return (
            torch.tensor(image).float().unsqueeze(0),
            torch.tensor(prediction).float().unsqueeze(0),
            torch.tensor(float(label)).unsqueeze(0),
        )
        # return torch.tensor(prediction).float().unsqueeze(0), torch.tensor(prediction).float().unsqueeze(0), torch.tensor(float(label)).unsqueeze(0)
        # return torch.stack((torch.tensor(image).float(), torch.tensor(prediction).float())), torch.tensor(prediction).float().unsqueeze(0), torch.tensor(float(label)).unsqueeze(0)



class LazyBoxVal(Dataset):
    def __init__(
        self, base_path: str, glob_pattern: str, threshold: float = 0.2, size: int = 20, input_type: InputType = InputType.IMAGE,
    ) -> None:
        file_names = glob.glob(os.path.join(base_path, "labelsTs", glob_pattern))
        base_file_names = [os.path.basename(file_name)[:-7] for file_name in file_names]

        self.base_path = base_path
        self.base_file_names = base_file_names

        self.threshold = threshold
        self.size = size
        self.input_type = input_type

    def __len__(self) -> int:
        return len(self.base_file_names)

    def __getitem__(self, index):
        base_file_name = self.base_file_names[index]
        image, label, meta_data, prediction = util.get_file_data(self.base_path, base_file_name)

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


class BoxVal(Dataset):
    def __init__(
        self, base_path: str, glob_pattern: str, threshold: float = 0.2
    ) -> None:
        file_names = glob.glob(os.path.join(base_path, "labelsTs", glob_pattern))
        base_file_names = [os.path.basename(file_name)[:-7] for file_name in file_names]

        images, labels, meta_data, predictions = zip(
            *(util.get_file_data(base_path, file_name) for file_name in base_file_names)
        )
        cropped_images, cropped_labels = zip(
            *(
                util.get_cropped_data(image, label, meta)
                for image, label, meta in zip(images, labels, meta_data)
            )
        )
        components = [
            util.get_components(prediction >= threshold) for prediction in predictions
        ]
        components_comp = [
            util.get_components(prediction >= 0.5) for prediction in predictions
        ]
        labels_components = [
            util.get_components(cropped_label) for cropped_label in cropped_labels
        ]

        self.images = images
        self.labels = labels
        self.cropped_images = cropped_images
        self.cropped_labels = cropped_labels
        self.labels_components = labels_components
        self.predictions = predictions
        self.components = components
        self.components_comp = components_comp
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        header = self.images[index].header
        cropped_image = self.cropped_images[index]
        cropped_label = self.cropped_labels[index]
        labels_components = self.labels_components[index]
        components = self.components[index]
        components_comp = self.components_comp[index]

        boxes = util.get_boxes(components, header, size=20)
        boxes = [[(s.start, s.stop) for s in box] for box in boxes]

        return (
            torch.tensor(cropped_image).float().unsqueeze(0),
            torch.tensor(components.astype(np.int32)).unsqueeze(0),
            torch.tensor(components_comp.astype(np.int32)).unsqueeze(0),
            torch.tensor(labels_components.astype(np.int32)).unsqueeze(0),
            [list(enumerate(boxes, start=1))],
        )
        # return torch.tensor(cropped_image).float().unsqueeze(0), torch.tensor(predictions).float().unsqueeze(0), torch.tensor(cropped_label).float().unsqueeze(0), [list(enumerate(boxes, start=1))], self.threshold
        # return torch.tensor(predictions).float().unsqueeze(0), torch.tensor(predictions).float().unsqueeze(0), torch.tensor(cropped_label).float().unsqueeze(0), [list(enumerate(boxes, start=1))], self.threshold
        # return torch.stack((torch.tensor(cropped_image).float(), torch.tensor(predictions).float())), torch.tensor(predictions).float().unsqueeze(0), torch.tensor(cropped_label).float().unsqueeze(0), [list(enumerate(boxes, start=1))], self.threshold


class ClassificationDataset(Dataset):
    def __init__(self, image_dir: str, prediction_dir: str, label_dir: str) -> None:
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        self.prediction_files = sorted(
            glob.glob(os.path.join(prediction_dir, "*.nii.gz"))
        )
        self.label_dir = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))

        # TODO: Use the intersection of all sets, only

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index):
        image = nib.load(self.image_files[index]).get_fdata()
        prediction = nib.load(self.prediction_files[index]).get_fdata()
        label = nib.load(self.label_dir[index]).get_fdata()

        return np.stack((image, prediction)), label


# image, predictions, labels

# 29_T3_MRI_SWI_BFC_50mm_HM_0000.nii.gz
# 29_T3_MRI_SWI_BFC_50mm_HM.nii.gz
# 29_T3_MRI_SWI_BFC_50mm_HM.nii.gz

# 308_T1_MRI_SWI_BFC_50mm_HM_0000.nii.gz
# 308_T1_MRI_SWI_BFC_50mm_HM.nii.gz
# 308_T1_MRI_SWI_BFC_50mm_HM.nii.gz
