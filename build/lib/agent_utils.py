import cv2
import numpy as np


def preprocess_batch(obs_batch: np.ndarray, height: int = 84, width: int = 84) -> np.ndarray:
    num_envs = obs_batch.shape[0]
    # Fast path: already grayscale and correctly sized
    if obs_batch.ndim == 3 and obs_batch.shape[1] == height and obs_batch.shape[2] == width:
        # Ensure contiguous uint8
        return obs_batch.astype(np.uint8, copy=False)

    processed = np.zeros((num_envs, height, width), dtype=np.uint8)
    for i in range(num_envs):
        frame = obs_batch[i]
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 2:
            # Already grayscale
            pass
        else:
            raise ValueError(f"Unexpected observation shape: {frame.shape}")

        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        processed[i] = frame
    return processed
