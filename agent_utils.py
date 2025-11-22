import cv2
import numpy as np


def preprocess_batch(obs_batch: np.ndarray, height: int = 84, width: int = 84) -> np.ndarray:
    """
    Resize a batch of RGB observations to (height, width) while preserving 3 channels.
    Returns uint8 array shaped (batch, 3, height, width).
    """
    num_envs = obs_batch.shape[0]
    processed = np.zeros((num_envs, 3, height, width), dtype=np.uint8)
    for i in range(num_envs):
        frame = cv2.resize(obs_batch[i], (width, height), interpolation=cv2.INTER_AREA)
        # cv2 outputs HWC; transpose to CHW
        processed[i] = frame.transpose(2, 0, 1)
    return processed
