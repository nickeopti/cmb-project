import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import numpy as np
import torch

import arguments
import data
import util
from model import CMBClassifier


class CSVBuffer:
    def __init__(self, metrics_path: str, header: str = None) -> None:
        self.metrics_path = metrics_path
        self.buffer: List[Tuple[Any, ...]] = []
        self.lock = threading.Lock()

        if header is not None:
            with open(metrics_path, 'w') as f:
                f.write(f'{header}\n')
    
    def append(self, item):
        self.lock.acquire()
        self.buffer.append(item)
        self.lock.release()
    
    def flush(self):
        self.lock.acquire()
        with open(self.metrics_path, 'a') as f:
            f.writelines(','.join(map(str, item)) + '\n' for item in self.buffer)
        
        # reset buffer
        self.buffer = []
        
        self.lock.release()


def compute_image_metrics(buffer, image, candidate_components, predictions, baseline_components, labels, boxes):
    for threshold in np.linspace(0, 1, 25):
        candidates = candidate_components.numpy().copy()

        for i, box in boxes[0]:
            box = [slice(a, b) for a, b in box]
            image_box = util.extract_image_box(image, box)

            if image_box is None:
                candidates[candidates == i] = 0
            else:
                with torch.no_grad():
                    prediction = model(image_box.unsqueeze(0))

                    if torch.sigmoid(prediction).item() < threshold:
                        candidates[candidates == i] = 0
        
        tp, fp, fn, n = util.compute_object_metrics(
            labels.numpy().copy(),
            candidates.copy(),
        )
        buffer.append(('proposed', threshold, tp, fp, fn, n))

    for threshold in np.linspace(dataset.threshold, 1, 25):
        baseline = candidate_components.numpy().copy()

        for i, box in boxes[0]:
            box = [slice(a, b) for a, b in box]
            pred_box = util.extract_image_box(predictions, box)                

            if pred_box is None or torch.all(pred_box < threshold):
                baseline[baseline == i] = 0
        
        tp, fp, fn, n = util.compute_object_metrics(
            labels.numpy().copy(),
            baseline.copy(),
        )
        buffer.append(('baseline', threshold, tp, fp, fn, n))

        # print(threshold, tp, fp, fn, n)
        # print(f'{np.unique(baseline.copy())=}')
    
    buffer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')

    dataset: data.LazyBoxVal = arguments.add_arguments(parser, "dataset", data.LazyBoxVal)()
    buffer: CSVBuffer = arguments.add_arguments(parser, 'buffer', CSVBuffer)(header='method,threshold,tp,fp,fn,n')

    args = parser.parse_args()

    model = CMBClassifier.load_from_checkpoint(args.checkpoint, activation_function=torch.nn.CELU, n_input_channels=1, learning_rate=0.0005)

    def process_image(i):
        print(i)
        blah = dataset[i]
        compute_image_metrics(buffer, *blah)
    
    print(len(dataset))
    with ThreadPoolExecutor(os.cpu_count() * 2) as pool:
        pool.map(process_image, range(10))
