from typing import List, Tuple
import numpy as np
from ..helpers import bbox_helper


def get_anchors(seq_len: int, scales: List[int]) -> np.ndarray:
    
    anchors = np.zeros((seq_len, len(scales), 2), dtype=np.int32)
    for pos in range(seq_len):
        for scale_idx, scale in enumerate(scales):
            anchors[pos][scale_idx] = [pos, scale]
    return anchors


def get_pos_label(anchors: np.ndarray,
                  targets: np.ndarray,
                  iou_thresh: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    
    seq_len, num_scales, _ = anchors.shape
    anchors = np.reshape(anchors, (seq_len * num_scales, 2))

    loc_label = np.zeros((seq_len * num_scales, 2))
    cls_label = np.zeros(seq_len * num_scales, dtype=np.int32)

    for target in targets:
        target = np.tile(target, (seq_len * num_scales, 1))
        iou = bbox_helper.iou_cw(anchors, target)
        pos_idx = np.where(iou > iou_thresh)
        cls_label[pos_idx] = 1
        loc_label[pos_idx] = bbox2offset(target[pos_idx], anchors[pos_idx])

    loc_label = loc_label.reshape((seq_len, num_scales, 2))
    cls_label = cls_label.reshape((seq_len, num_scales))

    return cls_label, loc_label


def get_neg_label(cls_label: np.ndarray, num_neg: int) -> np.ndarray:
    
    seq_len, num_scales = cls_label.shape
    cls_label = cls_label.copy().reshape(-1)
    cls_label[cls_label < 0] = 0  # reset negative samples

    neg_idx, = np.where(cls_label == 0)
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:num_neg]

    cls_label[neg_idx] = -1
    cls_label = np.reshape(cls_label, (seq_len, num_scales))
    return cls_label


def offset2bbox(offsets: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    
    offsets = offsets.reshape(-1, 2)
    anchors = anchors.reshape(-1, 2)

    offset_center, offset_width = offsets[:, 0], offsets[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    # Tc = Oc * Aw + Ac
    bbox_center = offset_center * anchor_width + anchor_center
    # Tw = exp(Ow) * Aw
    bbox_width = np.exp(offset_width) * anchor_width

    bbox = np.vstack((bbox_center, bbox_width)).T
    return bbox


def bbox2offset(bboxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:

    bbox_center, bbox_width = bboxes[:, 0], bboxes[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    # Oc = (Tc - Ac) / Aw
    offset_center = (bbox_center - anchor_center) / anchor_width
    # Ow = ln(Tw / Aw)
    offset_width = np.log(bbox_width / anchor_width)

    offset = np.vstack((offset_center, offset_width)).T
    return offset
