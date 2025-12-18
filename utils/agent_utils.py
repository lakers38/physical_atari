import cv2
import numpy as np


def preprocess_batch(obs_batch: np.ndarray, height: int = 84, width: int = 84) -> np.ndarray:
    num_envs = obs_batch.shape[0]
    processed = np.zeros((num_envs, height, width), dtype=np.uint8)
    for i in range(num_envs):
        frame = cv2.cvtColor(obs_batch[i], cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        processed[i] = frame
    return processed
