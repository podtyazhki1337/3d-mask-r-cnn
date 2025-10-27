import platform
import logging
import random
import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
from collections import defaultdict as _dd


from .data_generators import RPNGenerator, HeadGenerator


if platform.processor() == 'ppc64le':
    import core.custom_op.ppc64le_custom_op as custom_op
else:
    import core.custom_op.custom_op as custom_op

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, depth, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, z1, y2, x2, z2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        # Bounding box.
        horizontal_indicies = np.where(np.any(np.any(mask, axis=0), axis=1))[0]
        vertical_indicies = np.where(np.any(np.any(mask, axis=1), axis=1))[0]
        profound_indicies = np.where(np.any(np.any(mask, axis=0), axis=0))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = profound_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            z2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, z1, y2, x2, z2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, z1, y2, x2, z2]
    boxes: [boxes_count, (y1, x1, z1, y2, x2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[4], boxes[:, 4])
    z1 = np.maximum(box[2], boxes[:, 2])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps_3d(boxes1, boxes2, image_shape=None):
    """3D IoU с учетом анизотропии."""
    if boxes1 is None or boxes2 is None:
        return np.zeros((0, 0), dtype=np.float32)

    b1 = np.asarray(boxes1, dtype=np.float32)
    b2 = np.asarray(boxes2, dtype=np.float32)

    if b1.size == 0 or b2.size == 0:
        N1 = b1.shape[0] if b1.ndim == 2 else 0
        N2 = b2.shape[0] if b2.ndim == 2 else 0
        return np.zeros((N1, N2), dtype=np.float32)

    # Нормализация углов
    def _norm(b):
        y1 = np.minimum(b[:, 0], b[:, 3]);
        y2 = np.maximum(b[:, 0], b[:, 3])
        x1 = np.minimum(b[:, 1], b[:, 4]);
        x2 = np.maximum(b[:, 1], b[:, 4])
        z1 = np.minimum(b[:, 2], b[:, 5]);
        z2 = np.maximum(b[:, 2], b[:, 5])
        out = b.copy()
        out[:, 0], out[:, 1], out[:, 2] = y1, x1, z1
        out[:, 3], out[:, 4], out[:, 5] = y2, x2, z2
        return out

    b1, b2 = _norm(b1), _norm(b2)

    # Коэффициент анизотропии для IMAGE_DEPTH=12
    # aniso_z = 1.0
    # if image_shape is not None:
    #     H, W, D = image_shape
    #     aniso_z = max(1.0, np.sqrt(max(H, W) / max(D, 1.0)))

    # Векторизованное вычисление
    b1_exp = b1[:, np.newaxis, :]
    b2_exp = b2[np.newaxis, :, :]

    # Intersection
    y1_i = np.maximum(b1_exp[:, :, 0], b2_exp[:, :, 0])
    x1_i = np.maximum(b1_exp[:, :, 1], b2_exp[:, :, 1])
    z1_i = np.maximum(b1_exp[:, :, 2], b2_exp[:, :, 2])
    y2_i = np.minimum(b1_exp[:, :, 3], b2_exp[:, :, 3])
    x2_i = np.minimum(b1_exp[:, :, 4], b2_exp[:, :, 4])
    z2_i = np.minimum(b1_exp[:, :, 5], b2_exp[:, :, 5])

    h_i = np.maximum(y2_i - y1_i, 0.0)
    w_i = np.maximum(x2_i - x1_i, 0.0)
    #d_i = np.maximum(z2_i - z1_i, 0.0) * aniso_z
    d_i = np.maximum(z2_i - z1_i, 0.0)
    intersection = h_i * w_i * d_i

    # Volumes
    h1 = b1[:, 3] - b1[:, 0]
    w1 = b1[:, 4] - b1[:, 1]
    #d1 = (b1[:, 5] - b1[:, 2]) * aniso_z
    d1 = (b1[:, 5] - b1[:, 2])
    vol1 = (h1 * w1 * d1)[:, np.newaxis]

    h2 = b2[:, 3] - b2[:, 0]
    w2 = b2[:, 4] - b2[:, 1]
    #d2 = (b2[:, 5] - b2[:, 2]) * aniso_z
    d2 = (b2[:, 5] - b2[:, 2])
    vol2 = (h2 * w2 * d2)[np.newaxis, :]

    union = np.maximum(vol1 + vol2 - intersection, 1e-10)
    return np.clip(intersection / union, 0.0, 1.0).astype(np.float32)

# совместимость импортов (ничего больше в проекте править не нужно)
def compute_overlaps(boxes1, boxes2):
    return compute_overlaps_3d(boxes1, boxes2)


def apply_box_deltas_3d_graph(boxes, deltas, bbox_std_dev):
    """
    boxes : [N,6] НОРМАЛИЗОВАННЫЕ [0,1]
    deltas: [N,6] (dy,dx,dz, log(dh),log(dw),log(dd))
    """
    boxes = tf.cast(boxes, tf.float32)
    deltas = tf.cast(deltas, tf.float32)
    bbox_std_dev = tf.cast(bbox_std_dev, tf.float32)

    boxes.set_shape([None, 6])
    deltas.set_shape([None, 6])

    deltas = deltas * bbox_std_dev

    y1, x1, z1, y2, x2, z2 = [tf.squeeze(t, axis=1) for t in tf.split(boxes, 6, axis=1)]
    dy, dx, dz, dh, dw, dd = [tf.squeeze(t, axis=1) for t in tf.split(deltas, 6, axis=1)]

    h = y2 - y1
    w = x2 - x1
    d = z2 - z1

    cy = y1 + 0.5 * h
    cx = x1 + 0.5 * w
    cz = z1 + 0.5 * d

    LOG_SCALE_LIMIT = tf.math.log(1000.0 / 16.0)
    dh = tf.clip_by_value(dh, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
    dw = tf.clip_by_value(dw, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
    dd = tf.clip_by_value(dd, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)

    cy2 = cy + dy * h
    cx2 = cx + dx * w
    cz2 = cz + dz * d

    h2 = h * tf.exp(dh)
    w2 = w * tf.exp(dw)
    d2 = d * tf.exp(dd)

    y1n = cy2 - 0.5 * h2
    x1n = cx2 - 0.5 * w2
    z1n = cz2 - 0.5 * d2
    y2n = y1n + h2
    x2n = x1n + w2
    z2n = z1n + d2

    out = tf.stack([y1n, x1n, z1n, y2n, x2n, z2n], axis=1)
    out.set_shape([None, 6])
    return out


def norm_boxes_3d_graph(boxes, shape):
    """
    Нормализует 3D боксы из pixel в [0,1].
    ИСПРАВЛЕНО: БЕЗ shift для согласованности.
    """
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    d = tf.cast(shape[2], tf.float32)

    scale = tf.stack([h, w, d, h, w, d])  # БЕЗ -1!
    boxes = tf.cast(boxes, tf.float32)
    return tf.clip_by_value(boxes / scale, 0.0, 1.0)


def denorm_boxes_3d_graph(boxes, shape):
    """
    Денормализует 3D боксы из [0,1] в pixels.
    ИСПРАВЛЕНО: БЕЗ shift для согласованности.
    """
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    d = tf.cast(shape[2], tf.float32)

    scale = tf.stack([h, w, d, h, w, d])  # БЕЗ -1!
    boxes = tf.cast(boxes, tf.float32)
    return boxes * scale

def trim_zeros_graph(boxes, name=None):
    """
    Удаляет нулевой padding из боксов.
    boxes: [N, 6] в любых координатах
    Возвращает: (trimmed_boxes, non_zeros_mask)
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def box_refinement_graph(box, gt_box):
    """
    Вычисляет дельты для перехода от box к gt_box (3D).
    box:    [N,6] (y1,x1,z1,y2,x2,z2)
    gt_box: [N,6] (y1,x1,z1,y2,x2,z2)
    Возвращает: [N,6] дельты
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    
    h = box[:, 3] - box[:, 0]
    w = box[:, 4] - box[:, 1]
    d = box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * h
    center_x = box[:, 1] + 0.5 * w
    center_z = box[:, 2] + 0.5 * d
    
    gt_h = gt_box[:, 3] - gt_box[:, 0]
    gt_w = gt_box[:, 4] - gt_box[:, 1]
    gt_d = gt_box[:, 5] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_h
    gt_center_x = gt_box[:, 1] + 0.5 * gt_w
    gt_center_z = gt_box[:, 2] + 0.5 * gt_d
    
    dy = (gt_center_y - center_y) / tf.maximum(h, 1e-3)
    dx = (gt_center_x - center_x) / tf.maximum(w, 1e-3)
    dz = (gt_center_z - center_z) / tf.maximum(d, 1e-3)
    dh = tf.math.log(tf.maximum(gt_h / tf.maximum(h, 1e-3), 1e-6))
    dw = tf.math.log(tf.maximum(gt_w / tf.maximum(w, 1e-3), 1e-6))
    dd = tf.math.log(tf.maximum(gt_d / tf.maximum(d, 1e-3), 1e-6))
    
    return tf.stack([dy, dx, dz, dh, dw, dd], axis=1)


def batch_pack_graph(x, counts, num_rows):
    """
    Упаковывает тензор x в батч по counts.
    x: [sum(counts), ...]
    counts: [num_rows] - кол-во элементов на ряд
    num_rows: количество рядов (батч)
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[int(tf.reduce_sum(counts[:i])):int(tf.reduce_sum(counts[:i + 1]))])
    return tf.concat(outputs, axis=0)


def parse_image_meta_graph(image_meta):
    """
    Парсит image_meta тензор и возвращает словарь с полями.
    image_meta: [batch, meta_length] или [meta_length]
    Возвращает: dict с image_shape, window и т.д.
    """
    # Если 1D - добавляем batch dimension
    if len(image_meta.shape) == 1:
        image_meta = tf.expand_dims(image_meta, 0)
    
    # Формат meta: [image_id, H, W, D, ...window(6), ...]
    # Стандартная распаковка:
    image_id = image_meta[:, 0]
    image_shape = image_meta[:, 1:4]  # [B, 3] (H,W,D)
    window = image_meta[:, 4:10]      # [B, 6] (y1,x1,z1,y2,x2,z2)
    
    return {
        'image_id': tf.cast(image_id, tf.int32),
        'image_shape': tf.cast(image_shape, tf.int32),
        'window': window
    }





def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, Depth, instances]
    """
    if masks1 is None:
        masks1 = np.zeros((0, 0, 0), dtype=bool)
    if masks2 is None:
        masks2 = np.zeros((0, 0, 0), dtype=bool)
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def apply_box_deltas_3d(boxes, deltas, bbox_std_dev=None):
    """
    boxes : [N,6] (y1,x1,z1,y2,x2,z2) - НОРМАЛИЗОВАННЫЕ [0,1]
    deltas: [N,6] (dy,dx,dz, log(dh),log(dw),log(dd))
    """
    try:
        if tf.is_tensor(boxes) or tf.is_tensor(deltas):
            if bbox_std_dev is None:
                bbox_std_dev = tf.ones([6], dtype=tf.float32)
            return apply_box_deltas_3d_graph(boxes, deltas, bbox_std_dev)
    except (NameError, AttributeError):
        pass

    boxes = np.asarray(boxes, dtype=np.float32)
    deltas = np.asarray(deltas, dtype=np.float32)

    if bbox_std_dev is not None:
        deltas = deltas * np.asarray(bbox_std_dev, dtype=np.float32)

    # в центры/размеры
    h = boxes[:, 3] - boxes[:, 0]
    w = boxes[:, 4] - boxes[:, 1]
    d = boxes[:, 5] - boxes[:, 2]
    cy = boxes[:, 0] + 0.5 * h
    cx = boxes[:, 1] + 0.5 * w
    cz = boxes[:, 2] + 0.5 * d

    # ✅ ИСПРАВЛЕНИЕ: минимум для НОРМАЛИЗОВАННЫХ координат
    eps = 1e-6  # просто защита от деления на 0
    h = np.maximum(h, eps)
    w = np.maximum(w, eps)
    d = np.maximum(d, eps)

    dy, dx, dz, dh, dw, dd = deltas.T

    # защита от экспоненциальных выбросов
    dh = np.clip(dh, -4.0, 4.0)
    dw = np.clip(dw, -4.0, 4.0)
    dd = np.clip(dd, -4.0, 4.0)

    cy = cy + dy * h
    cx = cx + dx * w
    cz = cz + dz * d

    h = h * np.exp(dh)
    w = w * np.exp(dw)
    d = d * np.exp(dd)

    # ✅ Минимум для НОРМАЛИЗОВАННЫХ (или вообще убрать)
    h = np.maximum(h, eps)
    w = np.maximum(w, eps)
    d = np.maximum(d, eps)

    y1 = cy - 0.5 * h
    x1 = cx - 0.5 * w
    z1 = cz - 0.5 * d
    y2 = y1 + h
    x2 = x1 + w
    z2 = z1 + d

    return np.stack([y1, x1, z1, y2, x2, z2], axis=1)



def denorm_boxes_3d_graph(boxes, image_shape):

    boxes = tf.cast(boxes, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)  # [H,W,D]
    h, w, d = image_shape[0], image_shape[1], image_shape[2]
    scale = tf.stack([h, w, d, h, w, d])
    out = boxes * scale
    out.set_shape([None, 6])
    return out

# === SAFE-реализация применения 3D-дельт (полная замена) ===
def apply_box_deltas_3d_graph(boxes, deltas, bbox_std_dev):
    """
    TF-версия. Применяет дельты к 3D боксам (px-координаты).
    boxes : [N,6]  (y1,x1,z1,y2,x2,z2)  в пикселях
    deltas: [N,6]  (dy,dx,dz, log(dh),log(dw),log(dd)) — НОРМАЛИЗОВАНЫ (умножим ниже на std)
    bbox_std_dev: [6]
    return: [N,6] в пикселях
    """
    boxes = tf.cast(boxes, tf.float32)
    deltas = tf.cast(deltas, tf.float32)
    bbox_std_dev = tf.cast(bbox_std_dev, tf.float32)

    # гарантируем известную последнюю ось
    boxes.set_shape([None, 6])
    deltas.set_shape([None, 6])

    # денормализация дельт по std чтобы совпадать с тренировкой
    deltas = deltas * bbox_std_dev

    # разбивка без требований к статической форме
    y1, x1, z1, y2, x2, z2 = [tf.squeeze(t, axis=1) for t in tf.split(boxes, 6, axis=1)]
    dy, dx, dz, dh, dw, dd = [tf.squeeze(t, axis=1) for t in tf.split(deltas, 6, axis=1)]

    h  = y2 - y1
    w  = x2 - x1
    d  = z2 - z1
    cy = y1 + 0.5 * h
    cx = x1 + 0.5 * w
    cz = z1 + 0.5 * d

    # стабилизация масштабов (как у Matterport)
    LOG_SCALE_LIMIT = tf.math.log(1000.0 / 16.0)
    dh = tf.clip_by_value(dh, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
    dw = tf.clip_by_value(dw, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
    dd = tf.clip_by_value(dd, -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)

    cy2 = cy + dy * h
    cx2 = cx + dx * w
    cz2 = cz + dz * d
    h2  = h * tf.exp(dh)
    w2  = w * tf.exp(dw)
    d2  = d * tf.exp(dd)

    y1n = cy2 - 0.5 * h2
    x1n = cx2 - 0.5 * w2
    z1n = cz2 - 0.5 * d2
    y2n = y1n + h2
    x2n = x1n + w2
    z2n = z1n + d2

    out = tf.stack([y1n, x1n, z1n, y2n, x2n, z2n], axis=1)
    out.set_shape([None, 6])
    return out


def non_max_suppression_3d_graph(boxes, scores, threshold, max_boxes):
    """
    Графовый враппер над custom_op.non_max_suppression_3d.
    Вход:
      boxes   : [N,6] float32 (y1,x1,z1,y2,x2,z2)
      scores  : [N]   float32
      threshold: float (IoU)   <-- Python float
      max_boxes: int           <-- Python int
    Выход:
      selected_boxes: [M,6] float32
      keep_indices  : [M] int32
    """
    import tensorflow as tf

    # гарантируем тензорные типы для boxes/scores
    boxes  = tf.cast(boxes,  tf.float32)
    scores = tf.cast(scores, tf.float32)

    # custom op ожидает ПИТОНОВСКИЕ скаляры для порога и лимита
    try:
        thr = float(threshold)
    except Exception:
        v = tf.get_static_value(threshold)
        thr = float(v) if v is not None else 0.3

    try:
        mxb = int(max_boxes)
    except Exception:
        v = tf.get_static_value(max_boxes)
        mxb = int(v) if v is not None else 200

    keep = custom_op.non_max_suppression_3d(
        boxes, scores, mxb, thr, name="nms3d_graph"
    )
    selected = tf.gather(boxes, keep)
    selected.set_shape([None, 6])
    return selected, keep

def non_max_suppression_3d(boxes, scores, threshold, max_boxes=2000):
    """Улучшенная функция NMS для 3D детекции.

    boxes: [N, (y1, x1, z1, y2, x2, z2)]
    scores: [N] уверенности для каждого бокса
    threshold: порог IoU
    max_boxes: максимальное количество боксов для возврата

    Возвращает:
    boxes: [M, (y1, x1, z1, y2, x2, z2)] отобранные боксы
    indices: [M] индексы выбранных боксов
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0), dtype=np.int32)

    # Координаты
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    z1 = boxes[:, 2]
    y2 = boxes[:, 3]
    x2 = boxes[:, 4]
    z2 = boxes[:, 5]

    # Объемы боксов
    volumes = (y2 - y1) * (x2 - x1) * (z2 - z1)

    # Сортируем боксы по уверенности
    order = scores.argsort()[::-1]

    # Выбираем до max_boxes лучших боксов
    max_boxes = min(max_boxes, order.shape[0])
    order = order[:max_boxes]

    keep = []
    while order.size > 0:
        # Добавляем бокс с наивысшей уверенностью
        i = order[0]
        keep.append(i)

        # Если остался только один бокс, завершаем
        if order.size == 1:
            break

        # Вычисляем IoU с оставшимися боксами
        # Общий объем пересечения
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])

        # Проверяем, есть ли пересечение
        inter_height = np.maximum(0.0, yy2 - yy1)
        inter_width = np.maximum(0.0, xx2 - xx1)
        inter_depth = np.maximum(0.0, zz2 - zz1)

        inter_vol = inter_height * inter_width * inter_depth
        union_vol = volumes[i] + volumes[order[1:]] - inter_vol

        # Предотвращаем деление на ноль
        union_vol = np.maximum(union_vol, np.finfo(float).eps)

        # IoU
        iou = inter_vol / union_vol

        # Отбрасываем все боксы с высоким IoU
        to_keep = np.where(iou <= threshold)[0]

        # Обновляем индексы
        order = order[to_keep + 1]

    # Возвращаем отобранные боксы и их индексы
    return boxes[keep], np.array(keep, dtype=np.int32)


def compute_detection_score(proposals, gt_boxes, threshold=0.5):
    """Улучшенная функция вычисления Detection Score.

    proposals: [N, (y1, x1, z1, y2, x2, z2)] предложения RPN
    gt_boxes: [M, (y1, x1, z1, y2, x2, z2)] ground truth боксы
    threshold: порог IoU для успешной детекции

    Возвращает: Оценка качества детекции от 0.0 до 100.0
    """
    if len(proposals) == 0 or len(gt_boxes) == 0:
        return 0.0

    # Вычисляем перекрытия всех proposals со всеми GT boxes
    overlaps = compute_overlaps(proposals, gt_boxes)

    # Для каждого GT box найдем proposal с максимальным IoU
    max_iou_per_gt = np.max(overlaps, axis=0)

    # Количество "хороших" детекций (с IoU > threshold)
    good_detections = np.sum(max_iou_per_gt >= threshold)

    # Нормализуем по количеству GT boxes
    recall = good_detections / len(gt_boxes)

    # Штраф за избыточные proposals
    if len(proposals) > len(gt_boxes):
        precision = min(1.0, len(gt_boxes) / len(proposals))
        f1_score = 2 * precision * recall / (precision + recall + 1e-7)
        score = f1_score * 100.0  # Шкала от 0 до 100
    else:
        score = recall * 100.0  # Шкала от 0 до 100

    return score


def box_refinement_graph(box, gt_box):
    """
    Вычисляет дельты для перехода от box к gt_box (3D).
    box:    [N,6] (y1,x1,z1,y2,x2,z2) - НОРМАЛИЗОВАННЫЕ [0,1]
    gt_box: [N,6] (y1,x1,z1,y2,x2,z2) - НОРМАЛИЗОВАННЫЕ [0,1]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    eps = 1e-6  # ✅ защита от деления на 0

    h = box[:, 3] - box[:, 0]
    w = box[:, 4] - box[:, 1]
    d = box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * h
    center_x = box[:, 1] + 0.5 * w
    center_z = box[:, 2] + 0.5 * d

    gt_h = gt_box[:, 3] - gt_box[:, 0]
    gt_w = gt_box[:, 4] - gt_box[:, 1]
    gt_d = gt_box[:, 5] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_h
    gt_center_x = gt_box[:, 1] + 0.5 * gt_w
    gt_center_z = gt_box[:, 2] + 0.5 * gt_d

    # ✅ Используем eps вместо больших значений
    dy = (gt_center_y - center_y) / tf.maximum(h, eps)
    dx = (gt_center_x - center_x) / tf.maximum(w, eps)
    dz = (gt_center_z - center_z) / tf.maximum(d, eps)

    # ✅ Клиппинг перед log
    dh = tf.math.log(tf.maximum(gt_h, eps) / tf.maximum(h, eps))
    dw = tf.math.log(tf.maximum(gt_w, eps) / tf.maximum(w, eps))
    dd = tf.math.log(tf.maximum(gt_d, eps) / tf.maximum(d, eps))

    return tf.stack([dy, dx, dz, dh, dw, dd], axis=1)


def box_refinement(box, gt_box):
    """
    box и gt_box: [N, (y1, x1, z1, y2, x2, z2)] - НОРМАЛИЗОВАННЫЕ [0,1]
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    eps = 1e-6  # ✅ защита от деления на 0

    height = box[:, 3] - box[:, 0]
    width = box[:, 4] - box[:, 1]
    depth = box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
    center_z = box[:, 2] + 0.5 * depth

    gt_height = gt_box[:, 3] - gt_box[:, 0]
    gt_width = gt_box[:, 4] - gt_box[:, 1]
    gt_depth = gt_box[:, 5] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    gt_center_z = gt_box[:, 2] + 0.5 * gt_depth

    # ✅ Используем eps
    dy = (gt_center_y - center_y) / np.maximum(height, eps)
    dx = (gt_center_x - center_x) / np.maximum(width, eps)
    dz = (gt_center_z - center_z) / np.maximum(depth, eps)

    # ✅ Клиппинг перед log
    dh = np.log(np.maximum(gt_height, eps) / np.maximum(height, eps))
    dw = np.log(np.maximum(gt_width, eps) / np.maximum(width, eps))
    dd = np.log(np.maximum(gt_depth, eps) / np.maximum(depth, eps))

    return np.stack([dy, dx, dz, dh, dw, dd], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, z1, y2, x2, z2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w, d = image.shape[:3]
    window = (0, 0, 0, h, w, d)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop
    #TODO: adapt to 3D

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=-1)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, scale, 1], order=0)
    if crop is not None:
        y, x, z, h, w, d = crop
        mask = mask[y:y + h, x:x + w, z:z + d]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[..., i].astype(bool)
        y1, x1, z1, y2, x2, z2 = bbox[i][:6]
        m = m[y1:y2, x1:x2, z1:z2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[..., i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:3] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[..., i]
        y1, x1, z1, y2, x2, z2 = bbox[i][:6]
        h = y2 - y1
        w = x2 - x1
        d = z2 - z1
        # Resize with bilinear interpolation
        m = resize(m, (h, w, d))
        mask[y1:y2, x1:x2, z1:z2, i] = np.around(m).astype(np.bool)
    return mask



def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, z1, y2, x2, z2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1, z2 - z1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:3], dtype=np.bool)
    full_mask[y1:y2, x1:x2, z1:z2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride, max_depth=None):
    import numpy as np

    if isinstance(feature_stride, (list, tuple)):
        if len(feature_stride) == 3:
            sy, sx, sz = feature_stride
        elif len(feature_stride) == 2:
            sy = sx = feature_stride[0]
            sz = feature_stride[1]
        else:
            sy = sx = sz = int(feature_stride[0])
    else:
        sy = sx = sz = int(feature_stride)

    shifts_y = np.arange(0, shape[0], anchor_stride) * sy
    shifts_x = np.arange(0, shape[1], anchor_stride) * sx
    shifts_z = np.arange(0, shape[2], anchor_stride) * sz

    shifts_y, shifts_x, shifts_z = np.meshgrid(shifts_y, shifts_x, shifts_z, indexing='ij')

    if isinstance(scales, (int, float)):
        scales = [scales]
    if isinstance(ratios, (int, float)):
        ratios = [ratios]

    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            height = width = scale
            depth = scale * ratio

            if max_depth is not None:
                depth = np.clip(depth, 0.5, max_depth)
            else:
                depth = max(0.5, depth)

            base_anchors.append([
                -height / 2, -width / 2, -depth / 2,
                height / 2, width / 2, depth / 2
            ])

    base_anchors = np.array(base_anchors, dtype=np.float32)

    shifts_y_flat = shifts_y.ravel()
    shifts_x_flat = shifts_x.ravel()
    shifts_z_flat = shifts_z.ravel()

    shifts = np.stack([
        shifts_y_flat, shifts_x_flat, shifts_z_flat,
        shifts_y_flat, shifts_x_flat, shifts_z_flat
    ], axis=1)

    anchors = base_anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
    anchors = anchors.reshape(-1, 6)

    return anchors.astype(np.float32)


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride, config=None):
    import numpy as np

    L = len(feature_shapes)
    scales = sorted(list(scales))
    n_scales = len(scales)

    max_depth = None
    if config is not None:
        max_depth = getattr(config, 'IMAGE_DEPTH', None)
        if max_depth is None:
            max_depth = getattr(config, 'IMAGE_SHAPE', (0, 0, 16))[2]

    print(f"[ANCHORS] {n_scales} scales, {L} levels, max_depth={max_depth}")

    level_scales = []
    if n_scales >= L:
        scales_per_level = n_scales // L
        extra = n_scales % L
        start = 0
        for i in range(L):
            end = start + scales_per_level + (1 if i < extra else 0)
            level_scales.append(scales[start:end])
            start = end
    else:
        for i in range(L):
            level_scales.append([scales[min(i, n_scales - 1)]])

    anchors = []
    total_per_level = []

    for level_idx in range(L):
        stride = feature_strides[level_idx]
        if isinstance(stride, (list, tuple)):
            if len(stride) == 3:
                stride_3d = [stride[0], stride[1], stride[2]]
            elif len(stride) == 2:
                stride_3d = [stride[0], stride[0], stride[1]]
            else:
                stride_3d = [stride[0], stride[0], stride[0]]
        else:
            stride_3d = [stride, stride, stride]

        level_count = 0
        for scale in level_scales[level_idx]:
            level_anchors = generate_anchors(
                scale, ratios, feature_shapes[level_idx],
                stride_3d, anchor_stride, max_depth
            )
            anchors.append(level_anchors)
            level_count += len(level_anchors)

        total_per_level.append(level_count)
        print(f"  P{level_idx + 2}: shape={feature_shapes[level_idx]}, stride={stride_3d}, "
              f"scales={level_scales[level_idx]}, anchors={level_count}")

    result = np.concatenate(anchors, axis=0)
    print(f"[ANCHORS] Total: {len(result)}")
    return result



############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    # print(overlaps)
    ious = []
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                ious.append(overlaps[i, j])
                break

    return gt_match, pred_match, ious


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, ious = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    
    precision_score = np.sum(pred_match > -1) / len(pred_match)
    recall_score = np.sum(pred_match > -1) / len(gt_match)

    return mAP, precision_score, recall_score, ious


def rpn_evaluation(model, config, subsets, datasets, check_boxes):
    IOU_THRESH = float(getattr(config, "EVAL_MATCH_IOU", 0.50))
    IOU_GRID = list(getattr(config, "EVAL_MATCH_IOU_GRID", [0.30, 0.40, 0.50]))
    topk = int(getattr(config, "EVAL_TOPK_RPN", 10000))
    det_at = {}

    for subset, dataset in zip(subsets, datasets):
        print(subset)
        generator = RPNGenerator(dataset=dataset, config=config, shuffle=False)
        bbox_errors = []
        detection_scores = []
        class_loss = []
        bbox_loss = []
        checked = 0
        steps = min(config.EVALUATION_STEPS, len(generator))
        subset_key = subset.strip().lower().replace(" ", "_")
        det_at[subset_key] = {f"{thr:.2f}": [] for thr in IOU_GRID}

        for k in range(steps):
            inputs, _ = generator.__getitem__(k)
            _, _, _, batch_rpn_rois, rpn_class_loss, rpn_bbox_loss = model.predict(inputs)

            # ✅ ИСПРАВЛЕНИЕ: правильный расчёт размера батча
            batch_start = k * generator.batch_size
            batch_end = min(batch_start + generator.batch_size, len(generator.image_ids))
            actual_batch_size = batch_end - batch_start

            for m in range(actual_batch_size):  # ✅ используем actual_batch_size
                ds_id = generator.image_ids[batch_start + m]  # ✅ безопасный индекс
                _, boxes, _ = generator.load_image_gt(ds_id)

                # если GT пустой — только учтём лоссы
                if boxes is None or boxes.shape[0] == 0:
                    class_loss.append(rpn_class_loss[m])
                    bbox_loss.append(rpn_bbox_loss[m])
                    continue

                # денорм и отбор первых topk ROI
                rpn_rois = denorm_boxes(batch_rpn_rois[m, :topk], config.IMAGE_SHAPE[:3])
                if rpn_rois is None or rpn_rois.shape[0] == 0:
                    class_loss.append(rpn_class_loss[m])
                    bbox_loss.append(rpn_bbox_loss[m])
                    continue

                # обрезаем по размеру изображения и фильтруем
                H, W, D = config.IMAGE_SHAPE[:3]
                rpn_rois[:, 0] = np.clip(rpn_rois[:, 0], 0, H)
                rpn_rois[:, 1] = np.clip(rpn_rois[:, 1], 0, W)
                rpn_rois[:, 2] = np.clip(rpn_rois[:, 2], 0, D)
                rpn_rois[:, 3] = np.clip(rpn_rois[:, 3], 1, H)
                rpn_rois[:, 4] = np.clip(rpn_rois[:, 4], 1, W)
                rpn_rois[:, 5] = np.clip(rpn_rois[:, 5], 1, D)

                valid = ((rpn_rois[:, 3] > rpn_rois[:, 0]) &
                         (rpn_rois[:, 4] > rpn_rois[:, 1]) &
                         (rpn_rois[:, 5] > rpn_rois[:, 2]))
                rpn_rois = rpn_rois[valid]

                if rpn_rois.shape[0] == 0:
                    class_loss.append(rpn_class_loss[m])
                    bbox_loss.append(rpn_bbox_loss[m])
                    continue

                # Телеметрия
                try:
                    Telemetry.update_rpn_proposals(rpn_rois, boxes, config)
                except Exception:
                    pass

                # Вычисление IoU между GT boxes и RPN proposals
                overlaps = compute_overlaps(boxes, rpn_rois)  # [G x R]

                # Detection@IoU по разным topK
                TOPK_LIST = list(getattr(config, "EVAL_TOPK_GRID", [topk]))
                try:
                    for K in TOPK_LIST:
                        rpn_rois_k = denorm_boxes(batch_rpn_rois[m, :int(K)], config.IMAGE_SHAPE[:3])
                        if rpn_rois_k is None or rpn_rois_k.shape[0] == 0:
                            continue

                        rpn_rois_k[:, 0] = np.clip(rpn_rois_k[:, 0], 0, H)
                        rpn_rois_k[:, 1] = np.clip(rpn_rois_k[:, 1], 0, W)
                        rpn_rois_k[:, 2] = np.clip(rpn_rois_k[:, 2], 0, D)
                        rpn_rois_k[:, 3] = np.clip(rpn_rois_k[:, 3], 1, H)
                        rpn_rois_k[:, 4] = np.clip(rpn_rois_k[:, 4], 1, W)
                        rpn_rois_k[:, 5] = np.clip(rpn_rois_k[:, 5], 1, D)

                        valid_k = ((rpn_rois_k[:, 3] > rpn_rois_k[:, 0]) &
                                   (rpn_rois_k[:, 4] > rpn_rois_k[:, 1]) &
                                   (rpn_rois_k[:, 5] > rpn_rois_k[:, 2]))
                        rpn_rois_k = rpn_rois_k[valid_k]

                        if rpn_rois_k.shape[0] == 0:
                            continue

                        overlaps_k = compute_overlaps(boxes, rpn_rois_k)
                        for thr in IOU_GRID:
                            hits = (overlaps_k >= float(thr)).any(axis=1).sum()
                            det_at[subset_key].setdefault(f"{thr:.2f}@{int(K)}", []).append(
                                100.0 * float(hits) / max(1, boxes.shape[0])
                            )
                except Exception as e:
                    print(f"[rpn_eval] topk_grid error: {e}")

                # Matching boxes<->rois с правильной логикой
                roi_association = -1 * np.ones(boxes.shape[0], dtype=np.int32)
                box_association = -1 * np.ones(rpn_rois.shape[0], dtype=np.int32)

                for j in range(rpn_rois.shape[0]):
                    roi_overlaps = overlaps[:, j]
                    argmax = np.argmax(roi_overlaps)

                    if roi_association[argmax] == -1:
                        if roi_overlaps[argmax] > IOU_THRESH:
                            roi_association[argmax] = j
                            box_association[j] = argmax

                # Собираем matched пары
                matched_pairs = []
                for gt_idx in range(boxes.shape[0]):
                    if roi_association[gt_idx] != -1:
                        matched_pairs.append((gt_idx, roi_association[gt_idx]))

                if len(matched_pairs) == 0:
                    detection_scores.append(0.0)
                    class_loss.append(rpn_class_loss[m])
                    bbox_loss.append(rpn_bbox_loss[m])
                    continue

                gt_indices = [pair[0] for pair in matched_pairs]
                roi_indices = [pair[1] for pair in matched_pairs]

                positive_boxes = boxes[gt_indices]
                positive_rois = rpn_rois[roi_indices]

                if check_boxes and checked < 10:
                    print("GT:", positive_boxes)
                    print("Pred:", positive_rois)
                    checked += 1

                bbox_errors.append(float(np.mean(np.abs(positive_rois - positive_boxes))))
                detection_scores.append(100.0 * len(matched_pairs) / max(1, boxes.shape[0]))

                class_loss.append(rpn_class_loss[m])
                bbox_loss.append(rpn_bbox_loss[m])

        mean_class_loss = float(np.mean(class_loss)) if len(class_loss) else 0.0
        std_class_loss = float(np.std(class_loss)) if len(class_loss) else 0.0
        mean_bbox_loss = float(np.mean(bbox_loss)) if len(bbox_loss) else 0.0
        std_bbox_loss = float(np.std(bbox_loss)) if len(bbox_loss) else 0.0
        mean_bbox_error = float(np.mean(bbox_errors)) if len(bbox_errors) else 0.0
        mean_detection_score = float(np.mean(detection_scores)) if len(detection_scores) else 0.0

        print("CLASS:", mean_class_loss, "+/-", std_class_loss,
              "BBOX:", mean_bbox_loss, "+/-", std_bbox_loss)
        print("Mean Coordinate Error:", mean_bbox_error,
              "Detection score:", mean_detection_score)

        try:
            for thr_str, values in det_at[subset_key].items():
                if values:
                    mean_det_at = float(np.mean(values))
                    print(f"Detection@{thr_str}:", mean_det_at)
        except Exception:
            pass

def head_evaluation(model, config, subsets, datasets):

    for subset, dataset in zip(subsets, datasets):

        print(subset)

        data_generator = HeadGenerator(dataset=dataset, config=config, shuffle=False)
        class_losses = []
        bbox_losses = []
        mask_losses = []
        steps = min(config.EVALUATION_STEPS, len(data_generator))
        if steps == 0:
            print("No positive samples after filtering. Skipping evaluation for this subset.")
            continue
        for id in range(steps):

            inputs, _ = data_generator.__getitem__(id)
            outputs = model.predict(inputs)
            class_loss, bbox_loss, mask_loss = outputs[-3], outputs[-2], outputs[-1]
            class_losses.append(class_loss[0])
            bbox_losses.append(bbox_loss[0])
            mask_losses.append(mask_loss[0])

        mean_class_loss = float(np.mean(class_losses)) if class_losses else 0.0
        std_class_loss = float(np.std(class_losses)) if class_losses else 0.0
        mean_bbox_loss = float(np.mean(bbox_losses)) if bbox_losses else 0.0
        std_bbox_loss = float(np.std(bbox_losses)) if bbox_losses else 0.0
        mean_mask_loss = float(np.mean(mask_losses)) if mask_losses else 0.0
        std_mask_loss = float(np.std(mask_losses)) if mask_losses else 0.0

        print("CLASS:", mean_class_loss, "+/-", std_class_loss, 
              "BBOX:", mean_bbox_loss, "+/-", std_bbox_loss, 
              "MASK:", mean_mask_loss, "+/-", std_mask_loss)


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results.

    ФИНАЛЬНОЕ РЕШЕНИЕ с безопасным доступом к элементам.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    import tensorflow as tf

    if not isinstance(inputs, list):
        inputs = [inputs]

    # Паддим все входы до batch_size
    padded_inputs = []
    for inp in inputs:
        current_size = tf.shape(inp)[0]
        pad_size = tf.maximum(0, batch_size - current_size)

        # Получаем форму для паддинга
        inp_shape = tf.shape(inp)
        pad_shape = tf.concat([[pad_size], inp_shape[1:]], axis=0)
        padding = tf.zeros(pad_shape, dtype=inp.dtype)

        # Конкатенируем и обрезаем
        padded = tf.concat([inp, padding], axis=0)[:batch_size]
        padded_inputs.append(padded)

    # Создаем маску валидности (какие элементы реальные, а какие паддинг)
    actual_batch_size = tf.shape(inputs[0])[0]
    indices = tf.range(batch_size)
    valid_mask = tf.less(indices, actual_batch_size)

    # Обрабатываем первый элемент для получения структуры
    inputs_slice_0 = [inp[0] for inp in padded_inputs]
    output_slice_0 = graph_fn(*inputs_slice_0)

    if not isinstance(output_slice_0, (tuple, list)):
        output_slice_0 = [output_slice_0]
    elif isinstance(output_slice_0, tuple):
        output_slice_0 = list(output_slice_0)

    # Создаем шаблоны нулевых выходов
    zero_outputs = [tf.zeros_like(out) for out in output_slice_0]

    # Функция для обработки одного элемента
    def process_single(i):
        """Обрабатываем i-й элемент батча"""
        inputs_slice = [tf.gather(inp, i) for inp in padded_inputs]

        # Проверяем валидность
        is_valid = tf.gather(valid_mask, i)

        def real_process():
            result = graph_fn(*inputs_slice)
            if not isinstance(result, (tuple, list)):
                result = [result]
            elif isinstance(result, tuple):
                result = list(result)
            return result

        def zero_process():
            return zero_outputs

        return tf.cond(is_valid, real_process, zero_process)

    # Обрабатываем все элементы
    all_outputs = [process_single(i) for i in range(batch_size)]

    # Транспонируем структуру: из списка кортежей в кортеж списков
    outputs = list(zip(*all_outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]

    if len(result) == 1:
        result = result[0]

    return result


def norm_boxes(boxes, shape):
    """
    Нормализует боксы из пикселей в [0,1].
    boxes: [N, (y1, x1, z1, y2, x2, z2)] в пикселях
    shape: (H, W, D) размер изображения

    Returns: [N, (y1, x1, z1, y2, x2, z2)] нормализованные [0,1]
    """
    h, w, d = int(shape[0]), int(shape[1]), int(shape[2])
    scale = np.array([h, w, d, h, w, d], dtype=np.float32)

    # ✅ Простая нормализация без shift
    return (boxes.astype(np.float32) / scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """
    Денормализует боксы из [0,1] в пиксели.
    boxes: [N, (y1, x1, z1, y2, x2, z2)] нормализованные [0,1]
    shape: (H, W, D) размер изображения в пикселях

    Returns: [N, (y1, x1, z1, y2, x2, z2)] в пикселях (float32, НЕ округлённые!)
    """
    h, w, d = int(shape[0]), int(shape[1]), int(shape[2])
    scale = np.array([h, w, d, h, w, d], dtype=np.float32)

    # ✅ КРИТИЧНО: НЕ округляем! Оставляем float для точности IoU
    return (boxes.astype(np.float32) * scale).astype(np.float32)

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

class Telemetry:
    """Ultra-light counters/histograms for anchor/IoU/proposal stats.
       Updates are probabilistically sampled to keep overhead negligible."""
    enabled: bool = True
    sample: float = 0.05          # ~2% батчей обновляют телеметрию
    _cnt   = _dd(int)
    _hist  = _dd(list)
    _save_dir = None

    @staticmethod
    def set_save_dir(path):
        Telemetry._save_dir = path
    @staticmethod
    def reset():
        Telemetry._cnt  = _dd(int)
        Telemetry._hist = _dd(list)

    @staticmethod
    def update_gt_stats(gt_boxes, config):
        """Записать геометрию GT: XY, DZ и ratio≈dz/sqrt(dx*dy)."""
        if not getattr(config, "TELEMETRY", True):
            return
        import numpy as _np
        if gt_boxes is None or not hasattr(gt_boxes, "shape") or gt_boxes.shape[0] == 0:
            return
        g = _np.asarray(gt_boxes, dtype=_np.float32)
        dy = g[:, 3] - g[:, 0]
        dx = g[:, 4] - g[:, 1]
        dz = g[:, 5] - g[:, 2]
        xy = _np.sqrt(_np.maximum(1.0, dx * dy))
        Telemetry._hist['gt_xy'].extend([float(v) for v in xy[:128]])
        Telemetry._hist['gt_dz'].extend([float(v) for v in dz[:128]])
        Telemetry._hist['gt_ratio_est'].extend([float(dz_i / max(1.0, xy_i)) for dz_i, xy_i in zip(dz[:128], xy[:128])])

    @staticmethod
    def _snap_vals(vals, step, lo, hi, ndigits=3):
        import numpy as _np
        xs = []
        for v in vals:
            if v is None or not _np.isfinite(v):
                continue
            v = float(v)
            v = max(lo, min(hi, v))
            v = round(round(v / step) * step, ndigits)
            xs.append(v)
        return sorted(set(xs))

    @staticmethod
    def _quantize_scales(vals, step=8, lo=8, hi=256, limit=8):
        xs = Telemetry._snap_vals(vals, step=step, lo=lo, hi=hi, ndigits=0)
        return xs[:limit]

    @staticmethod
    def update_rpn_targets(anchors, iou_max, match, config):
        """Call inside build_rpn_targets after rpn_match computed."""
        import numpy as _np
        if not getattr(config, "TELEMETRY", True): return
        if _np.random.rand() > float(getattr(config, "TELEMETRY_SAMPLE", Telemetry.sample)): return

        Telemetry._save_dir = getattr(config, "WEIGHT_DIR", Telemetry._save_dir)
        Telemetry._cnt['rpn_pos'] += int(_np.sum(match == 1))
        Telemetry._cnt['rpn_neg'] += int(_np.sum(match == -1))
        Telemetry._cnt['rpn_neu'] += int(_np.sum(match == 0))

        # ИСПРАВЛЕНИЕ: берём IoU ТОЛЬКО от позитивов (не нейтралов/негативов!)
        pos_mask = (match == 1)
        if _np.any(pos_mask):
            vals = iou_max[pos_mask]  # ТОЛЬКО ПОЗИТИВЫ!
            # Фильтруем слишком малые значения (артефакты)
            vals = vals[vals > 0.05]  # порог 0.05 для фильтрации шума
            if vals.size > 0:
                # Семплируем до 256 для экономии памяти
                if vals.size > 256:
                    idx = _np.random.choice(vals.size, 256, replace=False)
                    vals = vals[idx]
                Telemetry._hist['rpn_iou_max'].extend([float(v) for v in vals])

        # Геометрия позитивных якорей (как было)
        pos_idx = _np.where(match == 1)[0]
        if pos_idx.size > 0:
            if pos_idx.size > 256:
                pos_idx = _np.random.choice(pos_idx, 256, replace=False)
            a = anchors[pos_idx]
            dy = a[:, 3] - a[:, 0]
            dx = a[:, 4] - a[:, 1]
            dz = a[:, 5] - a[:, 2]
            xy_geom = _np.sqrt(_np.maximum(1.0, dy * dx))
            Telemetry._hist['pos_dz'].extend([float(v) for v in dz])
            Telemetry._hist['pos_xy'].extend([float(v) for v in xy_geom])

            scales = _np.array(getattr(config, "RPN_ANCHOR_SCALES", [32, 64, 96, 128, 160]), dtype=_np.float32)
            ratios = _np.array(getattr(config, "RPN_ANCHOR_RATIOS", [0.1, 0.2, 0.3]), dtype=_np.float32)
            s_idx = _np.argmin(_np.abs(xy_geom[:, None] - scales[None, :]), axis=1)
            est_ratio = dz / _np.maximum(1.0, scales[s_idx])
            r_idx = _np.argmin(_np.abs(est_ratio[:, None] - ratios[None, :]), axis=1)
            for v in scales[s_idx]:
                Telemetry._cnt[f"pos_scale_{int(v)}"] += 1
            for v in ratios[r_idx]:
                Telemetry._cnt[f"pos_ratio_{v:.3f}"] += 1

    @staticmethod
    def update_rpn_proposals(rois, gt_boxes, config):
        """Лёгкая метрика совпадений RPN ROI с GT (IoU>=τ) + распределение размеров ROI.
           Делает рандомный семплинг, чтобы не было смещения «первые N».
        """
        if not getattr(config, "TELEMETRY", True):
            return
        if rois is None or gt_boxes is None or rois.size == 0 or gt_boxes.size == 0:
            return

        import numpy as _np
        # рандомная подвыборка
        R = min(rois.shape[0], 256)
        G = min(gt_boxes.shape[0], 64)
        idx_r = _np.random.choice(rois.shape[0], R, replace=False) if rois.shape[0] > R else _np.arange(rois.shape[0])
        idx_g = _np.random.choice(gt_boxes.shape[0], G, replace=False) if gt_boxes.shape[0] > G else _np.arange(
            gt_boxes.shape[0])
        r = rois[idx_r]
        g = gt_boxes[idx_g]

        inter_min = _np.maximum(r[:, None, :3], g[None, :, :3])
        inter_max = _np.minimum(r[:, None, 3:], g[None, :, 3:])
        inter_sz = _np.maximum(0.0, inter_max - inter_min)
        inter_vol = inter_sz[:, :, 0] * inter_sz[:, :, 1] * inter_sz[:, :, 2]
        vol_r = _np.prod(r[:, 3:] - r[:, :3], axis=1)[:, None]
        vol_g = _np.prod(g[:, 3:] - g[:, :3], axis=1)[None, :]
        iou = inter_vol / (vol_r + vol_g - inter_vol + 1e-9)

        thr = float(getattr(config, "EVAL_DET_IOU", 0.40))
        hits_mask = (iou >= thr)
        hits = hits_mask.any(axis=0).sum()
        Telemetry._cnt['prop_hits'] += int(hits)
        Telemetry._cnt['prop_total'] += int(G)

        # геометрия ROI (для прикидки нужных scale/ratio)
        dz = r[:, 5] - r[:, 2]
        dx = r[:, 4] - r[:, 1]
        dy = r[:, 3] - r[:, 0]
        xy = _np.sqrt(_np.maximum(1.0, dx * dy))
        Telemetry._hist['roi_dz'].extend([float(v) for v in dz[:64]])
        Telemetry._hist['roi_xy'].extend([float(v) for v in xy[:64]])

        if hits > 0:
            # для каждого GT найдём лучший ROI (макс IoU по столбцу)
            best_roi_idx_per_gt = _np.argmax(iou, axis=0)  # shape: [G]
            best_hits = _np.where(
                hits_mask[_np.arange(best_roi_idx_per_gt.shape[0]), _np.arange(best_roi_idx_per_gt.shape[0])],
                best_roi_idx_per_gt, -_np.ones_like(best_roi_idx_per_gt))
            best_hits = best_hits[best_hits >= 0]
            if best_hits.size:
                scales = _np.array(getattr(config, "RPN_ANCHOR_SCALES", [32, 64, 96, 128, 160]), dtype=_np.float32)
                ratios = _np.array(getattr(config, "RPN_ANCHOR_RATIOS", [0.1, 0.2, 0.3]), dtype=_np.float32)
                roi_xy_sel = xy[best_hits]
                roi_dz_sel = dz[best_hits]

                # маппим к ближайшему scale по XY
                s_idx = _np.argmin(_np.abs(roi_xy_sel[:, None] - scales[None, :]), axis=1)
                # оценим ratio как dz/scale выбранного масштаба и привяжем к ближайшему ratio из конфига
                est_ratio = roi_dz_sel / _np.maximum(1.0, scales[s_idx])
                r_idx = _np.argmin(_np.abs(est_ratio[:, None] - ratios[None, :]), axis=1)

                for v in scales[s_idx]:
                    Telemetry._cnt[f"pos_scale_{int(v)}"] += 1
                for v in ratios[r_idx]:
                    Telemetry._cnt[f"pos_ratio_{v:.3f}"] += 1
    @staticmethod
    def _topk_from_cnt(prefix, k=5):
        """Вернуть top-k значений по Telemetry._cnt с ключами вида f'{prefix}{value}'."""
        items = [(k, v) for k, v in Telemetry._cnt.items() if k.startswith(prefix)]
        if not items:
            return []
        items.sort(key=lambda x: -x[1])
        vals = []
        for key, _ in items[:k]:
            try:
                vals.append(float(key.replace(prefix, "")))
            except Exception:
                pass
        return vals

    @staticmethod
    def _quantize_scales(vals, step=8, lo=8, hi=256):
        """Округлить размеры под сетку якорей."""
        out = []
        for v in vals:
            if v and v > 0:
                q = int(max(lo, min(hi, round(float(v) / step) * step)))
                out.append(q)
        # uniq + сорт
        out = sorted(list({int(x) for x in out}))
        return out

    @staticmethod
    def _unique_sorted(lst, ndigits=3):
        """Уникальные отсортированные float с округлением."""
        s = sorted({round(float(x), ndigits) for x in lst if x and x > 0})
        return s

    @staticmethod
    def snapshot_and_reset(epoch, save_dir=None, extra=None):
        """Снимок телеметрии + подсказки scales/ratios в JSONL. Абсолютно безопасно к типам/ошибкам."""
        import os, json
        import numpy as _np
        def _py(x):
            # Преобразуем всё потенциально «не-JSON-овое» в питоновские типы
            try:

                if isinstance(x, (_np.integer, _np.floating)):
                    return float(x)
                if isinstance(x, _np.ndarray):
                    # режем длину, чтобы не распухал jsonl
                    return [float(v) for v in x.reshape(-1).tolist()[:32]]
            except Exception:
                pass
            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode("utf-8", "ignore")
                except Exception:
                    return str(x)
            if isinstance(x, (dict, list, tuple)):
                try:
                    # рекурсивная очистка
                    if isinstance(x, dict):
                        return {str(k): _py(v) for k, v in x.items()}
                    else:
                        return [_py(v) for v in x]
                except Exception:
                    return str(x)
            return x

        # --- подготовка снимка ---
        snap = {
            "epoch": int(epoch),
            "cnt": {str(k): int(v) for k, v in Telemetry._cnt.items()},
            "hist": {k: Telemetry._percentiles(v) for k, v in Telemetry._hist.items()},
        }
        if isinstance(extra, dict):
            snap["extra"] = {str(k): _py(v) for k, v in extra.items()}

        if not save_dir:
            save_dir = Telemetry._save_dir or "./weights"
        os.makedirs(save_dir, exist_ok=True)

        # --- top по scales/ratios из счётчиков, «привязанных к конфигу» ---
        cnt = snap["cnt"]
        scale_items = [(k, v) for k, v in cnt.items() if k.startswith("pos_scale_")]
        ratio_items = [(k, v) for k, v in cnt.items() if k.startswith("pos_ratio_")]

        def _topN(items, key_is_scale=False, N=10):
            if not items:
                return []
            if key_is_scale:
                items = [(int(k.split("pos_scale_")[1]), int(v)) for k, v in items]
            else:
                items = [(float(k.split("pos_ratio_")[1]), int(v)) for k, v in items]
            items.sort(key=lambda kv: (-kv[1], kv[0]))
            return items[:N]

        top_scales = _topN(scale_items, key_is_scale=True, N=10)
        top_ratios = _topN(ratio_items, key_is_scale=False, N=10)
        snap["top"] = {
            "scales": [{"value": int(s), "count": int(c)} for s, c in top_scales],
            "ratios": [{"value": float(r), "count": int(c)} for r, c in top_ratios],
        }

        # --- SUGGEST: готовые списки для конфига (квантизация по перцентилям) ---
        SCALE_STEP = float(getattr(Telemetry, "SCALE_STEP", 8))
        SCALE_MIN, SCALE_MAX_DEFAULT = 8, 256
        RATIO_STEP = float(getattr(Telemetry, "RATIO_STEP", 0.02))
        RATIO_MIN, RATIO_MAX = 0.04, 0.30
        LIMIT = 8

        # XY → scales
        xy_vals = []
        for key in ("gt_xy", "pos_xy", "roi_xy"):
            h = snap["hist"].get(key, {})
            if "p50" in h:
                xy_vals += [h.get("p25", 0.0), h.get("p50", 0.0), h.get("p75", 0.0)]
        hi_xy = int(max(SCALE_MAX_DEFAULT, snap["hist"].get("roi_xy", {}).get("max", SCALE_MAX_DEFAULT)))
        scales_suggest = Telemetry._quantize_scales(xy_vals, step=SCALE_STEP, lo=SCALE_MIN, hi=hi_xy)[:LIMIT]

        # Ratio → ratios
        gt_rat = snap["hist"].get("gt_ratio_est", {})
        roi_xy = snap["hist"].get("roi_xy", {})
        roi_dz = snap["hist"].get("roi_dz", {})

        est_candidates = []
        # 1) Берём gt_ratio_est p25/p50/p75, если есть
        for k in ("p25", "p50", "p75"):
            if k in gt_rat:
                est_candidates.append(float(gt_rat[k]))
        # 2) Если есть ROI-оценки xy и dz, строим dz/xy по тем же перцентилям
        if all(k in roi_xy for k in ("p25", "p50", "p75")) and all(k in roi_dz for k in ("p25", "p50", "p75")):
            for k in ("p25", "p50", "p75"):
                denom = max(1e-6, float(roi_xy[k]))
                est_candidates.append(float(roi_dz[k]) / denom)

        # Фильтрация/квантизация
        def _snap_ratio(vals, step=RATIO_STEP, lo=RATIO_MIN, hi=RATIO_MAX, limit=LIMIT):
            return Telemetry._snap_vals(vals, step=step, lo=lo, hi=hi, ndigits=3)[:limit]

        ratios_suggest = _snap_ratio(est_candidates, step=RATIO_STEP, lo=RATIO_MIN, hi=RATIO_MAX, limit=LIMIT)

        snap["suggest"] = {
            "scales": [int(s) for s in scales_suggest],
            "ratios": [float(r) for r in ratios_suggest],
        }

        # --- запись JSONL (атомарно и без падений)
        try:
            path = os.path.join(save_dir, "telemetry.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_py(snap), ensure_ascii=False) + "\n")
        except Exception as e:
            # Логируем, но не мешаем обучению
            print(f"[Telemetry] write failed: {e}")

        # Сброс счётчиков
        Telemetry.reset()

    @staticmethod
    def _percentiles(arr):
        if not arr: return {}
        a = np.asarray(arr, dtype=np.float32)  # ← ИЗМЕНЕНО: _np → np
        return {
            "count": int(a.size),
            "min": float(a.min()),
            "p25": float(np.percentile(a, 25)),  # ← ИЗМЕНЕНО: _np → np
            "p50": float(np.percentile(a, 50)),  # ← ИЗМЕНЕНО: _np → np
            "p75": float(np.percentile(a, 75)),  # ← ИЗМЕНЕНО: _np → np
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std())
        }

    @staticmethod
    def log_config_params(config):
        """Логирует ключевые параметры конфига для анализа."""
        params = {
            "IMAGE_SHAPE": tuple(getattr(config, "IMAGE_SHAPE", (0, 0, 0))),
            "RPN_ANCHOR_SCALES": list(getattr(config, "RPN_ANCHOR_SCALES", [])),
            "RPN_ANCHOR_RATIOS": list(getattr(config, "RPN_ANCHOR_RATIOS", [])),
            "RPN_BBOX_STD_DEV": list(getattr(config, "RPN_BBOX_STD_DEV", [])),
            "BBOX_STD_DEV": list(getattr(config, "BBOX_STD_DEV", [])),
            "RPN_POSITIVE_IOU": float(getattr(config, "RPN_POSITIVE_IOU", 0.0)),
            "RPN_NEGATIVE_IOU": float(getattr(config, "RPN_NEGATIVE_IOU", 0.0)),
            "VOXEL_Z_OVER_Y": float(getattr(config, "VOXEL_Z_OVER_Y", 0.0)),
            "RPN_TRAIN_ANCHORS_PER_IMAGE": int(getattr(config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 0)),
            "PRE_NMS_LIMIT": int(getattr(config, "PRE_NMS_LIMIT", 0)),
            "POST_NMS_ROIS_TRAINING": int(getattr(config, "POST_NMS_ROIS_TRAINING", 0)),
            "ANCHOR_NB": int(getattr(config, "ANCHOR_NB", 0)),
        }
        print("\n" + "=" * 60)
        print("CONFIG PARAMETERS:")
        print("=" * 60)
        for k, v in params.items():
            print(f"  {k}: {v}")
        print("=" * 60 + "\n")
        return params
