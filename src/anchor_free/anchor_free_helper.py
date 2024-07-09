import numpy as np

from ..helpers import bbox_helper

def get_loc_label(target: np.ndarray) -> np.ndarray:
    
    seq_len, = target.shape

    bboxes = bbox_helper.seq2bbox(target)
    offsets = bbox2offset(bboxes, seq_len)

    return offsets


def get_ctr_label(target: np.ndarray,
                  offset: np.ndarray,
                  eps: float = 1e-8
                  ) -> np.ndarray:
    
    target = np.asarray(target, dtype=bool)
    ctr_label = np.zeros(target.shape, dtype=np.float32)

    offset_left, offset_right = offset[target, 0], offset[target, 1]
    ctr_label[target] = np.minimum(offset_left, offset_right) / (
        np.maximum(offset_left, offset_right) + eps)

    return ctr_label


def bbox2offset(bboxes: np.ndarray, seq_len: int) -> np.ndarray:
    
    pos_idx = np.arange(seq_len, dtype=np.float32)
    offsets = np.zeros((seq_len, 2), dtype=np.float32)

    for lo, hi in bboxes:
        bbox_pos = pos_idx[lo:hi]
        offsets[lo:hi] = np.vstack((bbox_pos - lo, hi - 1 - bbox_pos)).T

    return offsets


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    
    offset_left, offset_right = offsets[:, 0], offsets[:, 1]
    seq_len, _ = offsets.shape
    indices = np.arange(seq_len)
    bbox_left = indices - offset_left
    bbox_right = indices + offset_right + 1
    bboxes = np.vstack((bbox_left, bbox_right)).T
    return bboxes
