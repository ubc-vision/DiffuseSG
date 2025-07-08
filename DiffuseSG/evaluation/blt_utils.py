"""
Based on BLT code (modified).
https://github.com/google-research/google-research/tree/master/layout-blt
"""
"""Metric utilities for layouts."""


import numpy as np


def get_perceptual_iou(layout, canvas_size=32):
    """
    Computes the perceptual IOU on the layout level.
    Args:
        layout: [B, 4] normalized bounding boxes within one layout.
                Format: (min_x, min_y, max_x, max_y).

        canvas_size: The size of the canvas to draw the layout on. Default is 32.

    Returns:
        The value for the overlap index. If no overlaps are found, 0 is returned.
    """
    layout = np.array(layout, dtype=np.float32).reshape(-1, 4)
    # pdb.set_trace()
    assert layout.min() >= 0. and layout.max() <= 1.
    layout *= canvas_size
    layout_channels = []
    for bbox in layout:
        canvas = np.zeros((canvas_size, canvas_size, 1), dtype=np.float32)

        # cxcywh format: avoid round behavior at 0.5.
        # center_x, center_y = bbox[0], bbox[1]
        # width, height = bbox[2], bbox[3]
        # min_x = round(center_x - width / 2. + 1e-4)
        # max_x = round(center_x + width / 2. + 1e-4)
        # min_y = round(center_y - height / 2. + 1e-4)
        # max_y = round(center_y + height / 2. + 1e-4)
        # min_x = max(0, min_x)
        # min_y = max(0, min_y)
        # max_x = min(canvas_size, max_x)
        # max_y = min(canvas_size, max_y)
        # canvas[min_x:max_x, min_y:max_y] = 1.

        # xyxy format: round behavior at 0.5.
        min_x, min_y, max_x, max_y = np.round(bbox).astype(int)
        canvas[min_x:max_x, min_y:max_y] = 1.

        layout_channels.append(canvas)
    if not layout_channels or len(layout) == 1:
        return None
    sum_layout_channel = np.sum(np.concatenate(layout_channels, axis=-1), axis=-1)
    overlap_area = np.sum(np.greater(sum_layout_channel, 1.))
    bbox_area = np.sum(np.greater(sum_layout_channel, 0.))

    if bbox_area == 0.:
        return None

    return overlap_area / bbox_area


def get_average_iou(layout):
    """
    Computes the average amount of overlap between any two bounding boxes in a layout as IoU.

    Args:
        layout: 1-d integer array in which every 4 elements form a group
      of box in the format (min_x, min_y, max_x, max_y).

    Returns:
        The value for the overlap index. If no overlaps are found, 0 is returned.
    """

    iou_values = []
    # layout = normalize_bbox(layout)
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            bbox1 = layout[i]
            bbox2 = layout[j]
            iou_for_pair = _get_iou(bbox1, bbox2)
            # note: this is different from the commonly used iou metric
            if iou_for_pair > 0.:
                iou_values.append(iou_for_pair)
            # iou_values.append(iou_for_pair)

    return np.mean(iou_values) if len(iou_values) else None


def get_overlap_index(layout):
    """
    Computes the average area of overlap between any two bounding boxes in a layout.
    This metric comes from LayoutGAN (https://openreview.net/pdf?id=HJxB5sRcFQ).

    Args:
        layout: 1-d integer array in which every 4 elements form a group of box
          in the format (min_x, min_y, max_x, max_y).

    Returns:
        The value for the overlap index. If no overlaps are found, 0 is returned.
    """

    intersection_areas = []
    # layout = normalize_bbox(layout)
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            bbox1 = layout[i]
            bbox2 = layout[j]

            intersection_area = _get_intersection_area(bbox1, bbox2)
            if intersection_area > 0.:
                intersection_areas.append(intersection_area)
    return np.sum(intersection_areas) if intersection_areas else None


def get_alignment_loss(layout):
    """
    Calculates alignment loss of bounding boxes.

    Rewrites the function in the layoutvae: alignment_loss_lib.py by numpy.

    Args:
        layout: [asset_num, asset_dim] float array. An iterable of normalized
        bounding box coordinates in the format (x_min, y_min, x_max, y_max), with
        (0, 0) at the top-left coordinate.

    Returns:
        Alignment loss between bounding boxes.
    """
    if len(layout) <= 1:
        return None
    else:
        a = layout
        b = layout

        a, b = a[None, :, None], b[:, None, None]
        cartesian_product = np.concatenate([a + np.zeros_like(b), b + np.zeros_like(a)], axis=2)

        left_correlation = left_similarity(cartesian_product)
        center_correlation = center_similarity(cartesian_product)
        right_correlation = right_similarity(cartesian_product)
        correlations = np.stack([left_correlation, center_correlation, right_correlation], axis=2)
        min_correlation = np.sum(np.min(correlations, axis=(1, 2)))
        return min_correlation


def _get_iou(bb0, bb1):
    """
    Computes the intersection over union between two bounding boxes.
    Bounding box format: (min_x, min_y, max_x, max_y).
    """
    intersection_area = _get_intersection_area(bb0, bb1)

    bb0_area = _get_area(bb0)
    bb1_area = _get_area(bb1)

    if np.isclose(bb0_area + bb1_area - intersection_area, 0.):
        return 0
    return intersection_area / (bb0_area + bb1_area - intersection_area)


def _get_intersection_area(bb0, bb1):
    """
    Computes the intersection area between two bounding boxes.
    Bounding box format: (min_x, min_y, max_x, max_y).
    """
    x_0, y_0, x_1, y_1 = bb0
    u_0, v_0, u_1, v_1 = bb1

    intersection_x_0 = max(x_0, u_0)
    intersection_y_0 = max(y_0, v_0)
    intersection_x_1 = min(x_1, u_1)
    intersection_y_1 = min(y_1, v_1)
    intersection_area = _get_area([intersection_x_0, intersection_y_0, intersection_x_1, intersection_y_1])
    return intersection_area


def _get_area(bounding_box):
    """
    Computes the area of a bounding box.
    Bounding box format: (min_x, min_y, max_x, max_y).
    """
    x_0, y_0, x_1, y_1 = bounding_box
    return max(0., x_1 - x_0) * max(0., y_1 - y_0)


def left_similarity(correlated):
    """
    Calculates left alignment loss of bounding boxes.
    Args:
        correlated: [assets_num, assets_num, 2, 4]. Combinations of all pairs of
        assets so we can calculate the similarity between these bounding boxes
        in parallel.
    Returns:
        Left alignment similarities between all pairs of assets in the layout.
    """

    remove_diagonal_entries_mask = np.zeros((correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
    correlations = np.mean(np.abs(correlated[:, :, 0, :2] - correlated[:, :, 1, :2]), axis=-1)
    return correlations + remove_diagonal_entries_mask


def right_similarity(correlated):
    """Calculates right alignment loss of bounding boxes."""

    correlations = np.mean(np.abs(correlated[:, :, 0, 2:] - correlated[:, :, 1, 2:]), axis=-1)
    remove_diagonal_entries_mask = np.zeros((correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
    return correlations + remove_diagonal_entries_mask


def center_similarity(correlated):
    """Calculates center alignment loss of bounding boxes."""

    x0 = (correlated[:, :, 0, 0] + correlated[:, :, 0, 2]) / 2
    y0 = (correlated[:, :, 0, 1] + correlated[:, :, 0, 3]) / 2

    centroids0 = np.stack([x0, y0], axis=2)

    x1 = (correlated[:, :, 1, 0] + correlated[:, :, 1, 2]) / 2
    y1 = (correlated[:, :, 1, 1] + correlated[:, :, 1, 3]) / 2
    centroids1 = np.stack([x1, y1], axis=2)

    correlations = np.mean(np.abs(centroids0 - centroids1), axis=-1)
    remove_diagonal_entries_mask = np.zeros((correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)

    return correlations + remove_diagonal_entries_mask
