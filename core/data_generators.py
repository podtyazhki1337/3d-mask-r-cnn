import math
import bz2
import _pickle as cPickle
import numpy as np
import pandas as pd
from skimage.io import imread
import keras
from core import utils

def apply_minimal_augs_3d(image, boxes, masks, config):
    """
    Безопасные для якорей 3D-аугментации:
      - flips по Y/X/Z с корректировкой боксов и масок
      - лёгкий jitter по интенсивности (яркость/шум)
    image: [Y, X, Z]
    boxes: [N, 6] (y1,x1,z1,y2,x2,z2) — ВАЖНО: y2/x2/z2 эксклюзивные!
    masks: [Y, X, Z, N] или None
    """
    if image is None:
        return image, boxes, masks

    import numpy as np

    image = image.copy()
    boxes = None if boxes is None else np.asarray(boxes, np.float32).copy()
    Y, X, Z = image.shape[:3]
    rng = np.random.RandomState(None)
    p = float(getattr(config, "AUG_PROB", 0.5))

    def flip_y():
        nonlocal image, masks, boxes
        image = image[::-1, :, :]
        if masks is not None:
            masks = masks[::-1, :, :]
        if boxes is not None and boxes.size:
            # для эксклюзивных координат: new_y1 = Y - old_y2, new_y2 = Y - old_y1
            y1 = Y - boxes[:, 3]
            y2 = Y - boxes[:, 0]
            boxes[:, 0], boxes[:, 3] = y1, y2

    def flip_x():
        nonlocal image, masks, boxes
        image = image[:, ::-1, :]
        if masks is not None:
            masks = masks[:, ::-1, :]
        if boxes is not None and boxes.size:
            # new_x1 = X - old_x2, new_x2 = X - old_x1
            x1 = X - boxes[:, 4]
            x2 = X - boxes[:, 1]
            boxes[:, 1], boxes[:, 4] = x1, x2

    def flip_z():
        nonlocal image, masks, boxes
        image = image[:, :, ::-1]
        if masks is not None:
            masks = masks[:, :, ::-1]
        if boxes is not None and boxes.size:
            # new_z1 = Z - old_z2, new_z2 = Z - old_z1
            z1 = Z - boxes[:, 5]
            z2 = Z - boxes[:, 2]
            boxes[:, 2], boxes[:, 5] = z1, z2

    # Геометрия
    if bool(getattr(config, "AUG_FLIP_Y", True)) and rng.rand() < p:
        flip_y()
    if bool(getattr(config, "AUG_FLIP_X", True)) and rng.rand() < p:
        flip_x()
    if bool(getattr(config, "AUG_FLIP_Z", False)) and rng.rand() < p:
        flip_z()

    # Яркость
    bd = float(getattr(config, "AUG_BRIGHTNESS_DELTA", 0.0))
    if bd > 0:
        vmin, vmax = np.min(image), np.max(image)
        scale = np.float32(bd) * (vmax - vmin + 1e-6)
        image = np.clip(image + rng.uniform(-scale, scale, size=image.shape).astype(image.dtype), vmin, vmax)

    # Лёгкий шум
    ns = float(getattr(config, "AUG_GAUSS_NOISE_STD", 0.0))
    if ns > 0:
        image = image + rng.normal(0.0, ns, size=image.shape).astype(image.dtype)

    return image, boxes, masks

def jitter_boxes_3d(boxes, count=3, scale_sigma=0.10, trans=(2, 2, 1),
                    img_shape=None, iou_thr=0.40, max_keep=None):
    """
    Генерирует вокруг каждого GT 'count' вариаций боксов + фильтрует по IoU.
    boxes: [N, (y1,x1,z1,y2,x2,z2)] (эксклюзивные правые/верхние/передние края)
    Возвращает concat[boxes, kept_jitters].
    max_keep: int или None — максимум ДОБАВОК на каждый GT-бокс (после IoU-фильтра).
    """
    import numpy as np

    if boxes is None:
        return boxes
    B = np.asarray(boxes, np.float32)
    if B.size == 0 or count <= 0:
        return B

    H = W = D = None
    if img_shape is not None:
        H, W, D = img_shape

    def iou_one_to_many(b, C):
        if C.size == 0:
            return np.zeros((0,), dtype=np.float32)
        y1 = np.maximum(b[0], C[:, 0]);  y2 = np.minimum(b[3], C[:, 3])
        x1 = np.maximum(b[1], C[:, 1]);  x2 = np.minimum(b[4], C[:, 4])
        z1 = np.maximum(b[2], C[:, 2]);  z2 = np.minimum(b[5], C[:, 5])
        inter = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0) * np.maximum(z2 - z1, 0)
        ab   = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-6)
        aC   = np.maximum((C[:,3]-C[:,0])*(C[:,4]-C[:,1])*(C[:,5]-C[:,2]), 1e-6)
        union = ab + aC - inter
        return inter / np.maximum(union, 1e-6)

    out = []
    for b in B:
        y1, x1, z1, y2, x2, z2 = b
        h = max(1.0, y2 - y1); w = max(1.0, x2 - x1); d = max(1.0, z2 - z1)
        cy, cx, cz = (y1 + y2) / 2.0, (x1 + x2) / 2.0, (z1 + z2) / 2.0

        cand = []
        for _ in range(int(count)):
            sh = 1.0 + np.random.randn() * float(scale_sigma)
            sw = 1.0 + np.random.randn() * float(scale_sigma)
            sd = 1.0 + np.random.randn() * float(scale_sigma)
            nh, nw, nd = max(1.0, h * sh), max(1.0, w * sw), max(1.0, d * sd)

            ty = cy + np.random.randint(-trans[0], trans[0] + 1)
            tx = cx + np.random.randint(-trans[1], trans[1] + 1)
            tz = cz + np.random.randint(-trans[2], trans[2] + 1)

            ny1, ny2 = ty - nh / 2.0, ty + nh / 2.0
            nx1, nx2 = tx - nw / 2.0, tx + nw / 2.0
            nz1, nz2 = tz - nd / 2.0, tz + nd / 2.0

            if img_shape is not None:
                ny1 = np.clip(ny1, 0, H - 1); ny2 = np.clip(ny2, 1, H)
                nx1 = np.clip(nx1, 0, W - 1); nx2 = np.clip(nx2, 1, W)
                nz1 = np.clip(nz1, 0, D - 1); nz2 = np.clip(nz2, 1, D)
                if ny2 <= ny1 or nx2 <= nx1 or nz2 <= nz1:
                    continue

            cand.append([ny1, nx1, nz1, ny2, nx2, nz2])

        if not cand:
            continue

        cand = np.asarray(cand, dtype=np.float32)
        ious = iou_one_to_many(b, cand)
        keep = cand[ious >= float(iou_thr)]

        if keep.size:
            if isinstance(max_keep, (int, np.integer)) and max_keep > 0 and keep.shape[0] > max_keep:
                # берём top-K по IoU к исходному боксу
                topk = np.argsort(ious[ious >= float(iou_thr)])[::-1][:int(max_keep)]
                keep = keep[topk]
            out.append(keep)

    if not out:
        return B
    aug = np.vstack(out).astype(np.float32)
    return np.vstack([B, aug]).astype(np.float32)
############################################################
#  Data Generator
############################################################

# core/data_generators.py
import numpy as np
import keras
from core import utils
 # уже используется ниже в других генераторах

# core/data_generators.py

class HeadGenerator(keras.utils.Sequence):
    """
    Генератор для обучения/валидации HEAD.
    Возвращает ровно то, что ожидают входы модели головы:
      inputs : [input_rois_aligned, input_mask_aligned, input_image_meta,
                input_target_class_ids, input_target_bbox, input_target_mask]
      targets: []  (лоссы добавлены в граф через add_loss)
    Все тензоры имеют внешнюю батч-ось (B=1).
    """

    def __init__(self, dataset, config, shuffle=True, training=True, batch_size=1):
        # базовые поля
        self.dataset = dataset
        self.config = config
        self.shuffle = bool(shuffle)
        self.training = bool(training)
        self.batch_size = int(batch_size) if batch_size is not None else 1  # здесь ожидается B=1

        # список id изображений
        self.image_ids = np.copy(self.dataset.image_ids).astype(np.int64)
        self._call_count = 0  # для чанкинга по ROI при режиме без балансировки

        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        # количество батчей на эпоху (по одному image_id на батч)
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    @staticmethod
    def _downsample_2x_mean(x):
        """
        x: [N, H, W, D, C] -> усреднение блоков 2x2x2 -> [N, H/2, W/2, D/2, C]
        Требует чётные H, W, D.
        """
        N, H, W, D, C = x.shape
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, "Downsample expects even spatial dims"
        x = x.reshape(N, H // 2, 2, W // 2, 2, D // 2, 2, C)
        return x.mean(axis=(2, 4, 6))

    def load_image_gt(self, image_id):
        """
        Возвращает РОВНО 5 массивов для головы (без image_meta):
          rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask
        Ничего не форсит и не балансирует.
        """
        import numpy as np
        # Dataset обязан реализовывать .load_data(image_id) и возвращать 5 массивов.
        rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = self.dataset.load_data(image_id)

        # типы/формы приводим здесь, но БЕЗ image_meta
        rois_aligned = rois_aligned.astype(np.float32, copy=False)
        mask_aligned = mask_aligned.astype(np.float32, copy=False)
        target_class_ids = target_class_ids.astype(np.int32, copy=False)
        target_bbox = target_bbox.astype(np.float32, copy=False)

        # target_mask ожидаем [T,mH,mW,mD] или [T,mH,mW,mD,1] -> сделаем [T,mH,mW,mD,1] и {0,1}
        if target_mask.ndim == 4:
            target_mask = target_mask[..., np.newaxis]
        elif target_mask.ndim != 5:
            raise ValueError(f"[HeadGenerator.load_image_gt] target_mask rank={target_mask.ndim}, need 4 or 5")
        target_mask = (target_mask > 0.5).astype(np.float32, copy=False)

        return rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask

    def __getitem__(self, idx):
        """
        Батч для головы с правильными размерностями всех тензоров.
        """
        import numpy as np

        image_id = int(self.image_ids[idx])

        # 1) Загружаем данные (уже с правильными размерностями после исправления ToyHeadDataset)
        rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = self.dataset.load_data(image_id)

        # 2) Конфиг/параметры
        cfg = self.config
        T = int(getattr(cfg, "TRAIN_ROIS_PER_IMAGE", 128))
        M_feat_mask = int(getattr(cfg, "MASK_POOL_SIZE", 14))

        # Размер таргет-маски из MASK_SHAPE
        assert hasattr(cfg, "MASK_SHAPE") and len(cfg.MASK_SHAPE) == 3
        M_tgt_mask = int(cfg.MASK_SHAPE[0])
        assert tuple(cfg.MASK_SHAPE) == (M_tgt_mask, M_tgt_mask, M_tgt_mask)

        H, W, D = int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1]), int(cfg.IMAGE_SHAPE[2])
        num_classes = int(getattr(self.dataset, "num_classes", getattr(cfg, "NUM_CLASSES", 2)))

        # 3) Типы
        rois_aligned = rois_aligned.astype(np.float32, copy=False)
        mask_aligned = mask_aligned.astype(np.float32, copy=False)
        target_class_ids = target_class_ids.astype(np.int32, copy=False)  # ОБЯЗАТЕЛЬНО int32
        target_bbox = target_bbox.astype(np.float32, copy=False)
        target_mask = target_mask.astype(np.float32, copy=False)

        # target_mask уже должен быть [T, mH, mW, mD, 1]
        if target_mask.ndim == 4:
            target_mask = target_mask[..., np.newaxis]
        target_mask = (target_mask > 0.5).astype(np.float32, copy=False)

        # 4) Ресайз функция (улучшенная)
        def _resize_spatial(x, M):
            """x: [N,Mh,Mw,Md,C] -> [N,M,M,M,C] или [N,Mh,Mw,Md] -> [N,M,M,M]"""
            if x is None:
                return None

            if x.ndim == 4:  # [N,Mh,Mw,Md]
                N, Mh, Mw, Md = x.shape
                has_ch = False
            elif x.ndim == 5:  # [N,Mh,Mw,Md,C]
                N, Mh, Mw, Md, C = x.shape
                has_ch = True
            else:
                raise ValueError(f"Unexpected x.ndim={x.ndim}")

            if (Mh, Mw, Md) == (M, M, M):
                return x.astype(np.float32, copy=False)

            # Простая стратегия ресайза - равномерная дискретизация
            ih = np.linspace(0, Mh - 1, M).astype(np.int64)
            iw = np.linspace(0, Mw - 1, M).astype(np.int64)
            iz = np.linspace(0, Md - 1, M).astype(np.int64)

            if has_ch:
                out = x[:, ih][:, :, iw][:, :, :, iz]  # [N,M,M,M,C]
            else:
                out = x[:, ih][:, :, iw][:, :, :, iz]  # [N,M,M,M]

            return out.astype(np.float32, copy=False)

        # 5) Ресайз масок
        # mask_aligned: должен остаться [T, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, C]
        if mask_aligned.ndim == 5:
            # Ресайзим пространственные размеры, сохраняя канальную ось
            mask_aligned = _resize_spatial(mask_aligned, M_feat_mask)
        else:
            # Если вдруг получили 4D, добавляем канальную ось
            mask_aligned = _resize_spatial(mask_aligned, M_feat_mask)
            if mask_aligned.ndim == 4:
                mask_aligned = mask_aligned[..., np.newaxis]

        # target_mask: [T, *MASK_SHAPE, 1]
        if target_mask.ndim == 5:
            # Убираем канальную ось для ресайза, потом возвращаем
            tm_no_ch = target_mask[..., 0]  # [T, m, m, m]
            tm_resized = _resize_spatial(tm_no_ch, M_tgt_mask)  # [T, M_tgt, M_tgt, M_tgt]
            target_mask = tm_resized[..., np.newaxis]  # [T, M_tgt, M_tgt, M_tgt, 1]

        # 6) Выборка ROI и паддинг
        total = int(rois_aligned.shape[0])
        order = np.arange(total, dtype=np.int32)
        if self.training and bool(getattr(cfg, "HEAD_SHUFFLE_ROIS", True)) and total > 0:
            np.random.shuffle(order)

        if total <= T:
            sel = order
        else:
            if self.training and bool(getattr(cfg, "HEAD_BALANCE_POS", True)):
                pos_idx = order[target_class_ids[order] > 0]
                neg_idx = order[target_class_ids[order] <= 0]
                pos_frac = float(getattr(cfg, "HEAD_POS_FRAC", 0.33))
                pos_cnt = int(round(T * pos_frac))
                neg_cnt = T - pos_cnt
                if pos_idx.size == 0:
                    sel = np.random.choice(neg_idx, size=T, replace=neg_idx.size < T)
                elif neg_idx.size == 0:
                    sel = np.random.choice(pos_idx, size=T, replace=pos_idx.size < T)
                else:
                    sel = np.concatenate([
                        np.random.choice(pos_idx, size=pos_cnt, replace=pos_idx.size < pos_cnt),
                        np.random.choice(neg_idx, size=neg_cnt, replace=neg_idx.size < neg_cnt)
                    ])
                    np.random.shuffle(sel)
            else:
                sel = np.random.choice(order, size=T, replace=False)

        def _take_or_pad(a, pad_val=0):
            if a is None:
                return None
            out = a[sel] if sel.size else a[:0]
            if out.shape[0] < T:
                pad = T - out.shape[0]
                pad_shape = (pad,) + out.shape[1:]
                out = np.concatenate([out, np.full(pad_shape, pad_val, dtype=out.dtype)], axis=0)
            return out

        rois_aligned = _take_or_pad(rois_aligned, pad_val=0)
        mask_aligned = _take_or_pad(mask_aligned, pad_val=0)
        target_class_ids = _take_or_pad(target_class_ids, pad_val=0).astype(np.int32, copy=False)
        target_bbox = _take_or_pad(target_bbox, pad_val=0)
        target_mask = _take_or_pad(target_mask, pad_val=0)

        # # 7) Финальные проверки форм
        # print(f"[DEBUG] HeadGenerator batch shapes:")
        # print(f"  rois_aligned: {rois_aligned.shape}")
        # print(f"  mask_aligned: {mask_aligned.shape}")
        # print(f"  target_class_ids: {target_class_ids.shape}")
        # print(f"  target_bbox: {target_bbox.shape}")
        # print(f"  target_mask: {target_mask.shape}")

        # 8) image_meta
        window = (0, 0, 0, H, W, D)
        active_class_ids = np.ones([num_classes], dtype=np.int32)
        image_meta = compose_image_meta(
            image_id=image_id,
            original_image_shape=(H, W, D, 1),
            image_shape=(H, W, D, 1),
            window=window,
            scale=1.0,
            active_class_ids=active_class_ids
        ).astype(np.float32)

        # 9) Батчевые оси (B=1)
        x = [
            np.expand_dims(rois_aligned, 0).astype(np.float32),
            np.expand_dims(mask_aligned, 0).astype(np.float32),
            np.expand_dims(image_meta, 0).astype(np.float32),
            np.expand_dims(target_class_ids, 0).astype(np.int32),  # ВАЖНО!
            np.expand_dims(target_bbox, 0).astype(np.float32),
            np.expand_dims(target_mask, 0).astype(np.float32),
        ]
        y = []
        return x, y


class RPNGenerator(keras.utils.Sequence):

    def __init__(self, dataset, config, shuffle=True):

        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = self.config.BATCH_SIZE
        backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes,
                                   config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)
        self.anchor_nb = self.anchors.shape[0]
        self.config.ANCHOR_NB = self.anchor_nb


    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(ids) == 0:
            raise IndexError("Empty batch requested — adjust evaluation steps or dataset size")
        return self.data_generator(ids)

    def rebuild_anchors(self):
        """Пересобрать якоря из self.config (вызов из коллбэка автотюнера)."""
        backbone_shapes = compute_backbone_shapes(self.config, self.config.IMAGE_SHAPE)
        self.anchors = utils.generate_pyramid_anchors(
            self.config.RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS,
            backbone_shapes, self.config.BACKBONE_STRIDES, self.config.RPN_ANCHOR_STRIDE
        )
        self.anchor_nb = self.anchors.shape[0]
        self.config.ANCHOR_NB = self.anchor_nb

    def data_generator(self, image_ids):
        """
        Генерирует батчи для RPN-тренировки.
        - Аугментации: только если self.training == True
        - GT-джиттер:   мягкий, опциональный через конфиг (по умолчанию выключен)
        """
        import numpy as np

        # Keras Sequence совместима с вызовом одним id — поддержим и это
        if len(image_ids) == 0:
            raise IndexError("Empty image_ids passed to data_generator.")
        b = 0
        batch_images = None
        batch_rpn_match = None
        batch_rpn_bbox = None

        # режим targeting — один пример «на показ»
        if getattr(self.config, "MODE", "training") == "targeting":
            image_id = image_ids[0]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.load_image_gt(image_id)

            # batch-оси
            batch_images = image[np.newaxis, ...].astype(np.float32)
            batch_image_meta = image_meta[np.newaxis, ...]

            # GT как ожидает модель в targeting-режиме
            gt_class_ids = np.asarray(gt_class_ids, dtype=np.int32)
            gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
            gt_masks = gt_masks.astype(bool)  # full-size или mini — согласно config.USE_MINI_MASK

            batch_gt_class_ids = gt_class_ids[np.newaxis, ...]  # [1, N]
            batch_gt_boxes = gt_boxes[np.newaxis, ...] if gt_boxes.ndim == 2 else gt_boxes[np.newaxis, :,
                                                                                  :]  # [1, N, 6]
            # gt_masks: [H, W, D, N] -> [1, H, W, D, N]
            if gt_masks.ndim == 4:
                batch_gt_masks = gt_masks[np.newaxis, ...]
            else:
                # на всякий случай, если пришло [H, W, D] без инстанс-оси
                batch_gt_masks = gt_masks[np.newaxis, ..., np.newaxis]

            # anchors в targeting НЕ подаются — они строятся внутри модели
            inputs = [batch_images, batch_image_meta, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            name = f"targeting_{int(image_id)}"
            return name, inputs

        # обычная тренировка
        batch_size = int(self.batch_size)
        batch_images = np.zeros((batch_size,) + tuple(self.config.IMAGE_SHAPE), dtype=np.float32)
        batch_rpn_match = np.zeros((batch_size, self.anchor_nb, 1), dtype=np.int8)
        batch_rpn_bbox = np.zeros((batch_size, int(getattr(self.config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 256)), 6),
                                  dtype=np.float32)

        for bi in range(batch_size):
            # вытаскиваем id и грузим с аугментациями (если training)
            image_id = image_ids[min(bi, len(image_ids) - 1)]
            image, boxes, class_ids = self.load_image_gt(image_id)

            # защищаемся от пустых GT — пересэмплим пару раз
            tries = 0
            while (boxes is None or np.size(boxes) == 0) and tries < 5:
                new_id = int(np.random.choice(self.image_ids))
                image, boxes, class_ids = self.load_image_gt(new_id)
                tries += 1

            # мягкий джиттер GT (по желанию)
            boxes_for_rpn = boxes
            if bool(getattr(self.config, "RPN_AUGMENT_GT", False)) and boxes is not None and np.size(boxes) > 0:
                boxes_for_rpn = jitter_boxes_3d(
                    boxes,
                    count=int(getattr(self.config, "RPN_GT_JITTER_PER_BOX", 1)),
                    scale_sigma=float(getattr(self.config, "RPN_GT_JITTER_SCALE_SIGMA", 0.05)),
                    trans=tuple(getattr(self.config, "RPN_GT_JITTER_TRANS", (1, 1, 0))),
                    img_shape=self.config.IMAGE_SHAPE[:3],
                    iou_thr=float(getattr(self.config, "RPN_GT_JITTER_IOU_THR", 0.70)),
                    max_keep=int(getattr(self.config, "RPN_GT_JITTER_MAX_KEEP", 1))
                )

            # таргеты RPN
            rpn_match, rpn_bbox = build_rpn_targets(self.anchors, class_ids, boxes_for_rpn, self.config)

            # заполняем батч
            batch_images[bi] = image.astype(np.float32)
            batch_rpn_match[bi] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[bi] = rpn_bbox

        # Keras-совместимый вывод
        inputs = [batch_images, batch_rpn_match, batch_rpn_bbox]
        outputs = []
        return inputs, outputs




    def load_image_gt(self, image_id):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        boxes, class_ids, masks = self.dataset.load_data(image_id)

        use_augs = bool(getattr(self.config, "AUGMENT", True)) and getattr(self, "training", True)
        if use_augs:
            image, boxes, masks = apply_minimal_augs_3d(image, boxes, masks, self.config)

        if self.config.MODE == "targeting":
            boxes, class_ids, masks = self.dataset.load_data(image_id)
            # active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
            # source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
            # active_class_ids[source_class_ids] = 1
            active_class_ids = np.ones([self.dataset.num_classes], dtype=np.int32)
            image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), 
                                            (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1, active_class_ids)
            return image, image_meta, class_ids, boxes, masks

        return image, boxes, class_ids

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)


class MrcnnGenerator(keras.utils.Sequence):

    def __init__(self, dataset, config, shuffle=True, batch_size=1, training=True):

        self.image_ids = np.copy(dataset.image_ids)
        np.random.shuffle(self.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.training = training
        self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors_pix = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            self.backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE
        )
        self.anchors = utils.norm_boxes(
            self.anchors_pix,
            tuple(self.config.IMAGE_SHAPE[:-1])  # (H, W, D)
        )

        self.config.ANCHOR_NB = self.anchors_pix.shape[0]
    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(ids) == 0:
            raise IndexError("Empty batch requested — adjust evaluation steps or dataset size")
        return self.data_generator(ids)

    def data_generator(self, image_ids):
        b = 0
        batch_size = len(image_ids)
        while b < self.batch_size:
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[b]
            if self.training:
                image, image_meta, gt_boxes, gt_class_ids, gt_masks = self.load_image_gt(image_id)
                rpn_match, rpn_bbox = build_rpn_targets(self.anchors_pix, gt_class_ids, gt_boxes, self.config)
                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 6), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.batch_size, gt_masks.shape[0], gt_masks.shape[1], gt_masks.shape[2],
                         self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    batch_rpn_match = np.zeros([self.batch_size, self.config.ANCHOR_NB, 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros([self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6],
                                              dtype=rpn_bbox.dtype)

                batch_image_meta[b] = image_meta
                batch_images[b] = image.astype(np.float32)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, ..., :gt_masks.shape[-1]] = gt_masks
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                b += 1

                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta,  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks,
                              batch_rpn_match, batch_rpn_bbox]
                    outputs = []

                    return inputs, outputs
            else:
                image, image_meta = self.load_image_gt(image_id)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_anchors = self.anchors[np.newaxis]

                batch_image_meta[b] = image_meta
                batch_images[b] = image.astype(np.float32)
                b += 1

                if b >= self.batch_size:
                    inputs = [batch_images, batch_image_meta, batch_anchors]
                    outputs = []

                    return inputs, outputs

    def load_image_gt(self, image_id):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        image_meta = compose_image_meta(image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE), 
                                        (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1, active_class_ids)
        if self.training:
            boxes, class_ids, masks = self.dataset.load_data(image_id)
            return image, image_meta, boxes, class_ids, masks
        else:
            return image, image_meta

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def get_input_prediction(self, image_id):
        """
        Готовит входы для инференса ТОЧНО как в тренировке:
          - изображение через dataset.load_image(image_id)
          - active_class_ids безопасно приводим к длине config.NUM_CLASSES (fallback если mapping пуст)
          - корректная image_meta через compose_image_meta(...)
          - anchors
        Возвращает:
          name, inputs: [images, image_meta, anchors]
        """
        import numpy as np

        info = self.dataset.image_info[image_id]
        image = self.dataset.load_image(image_id).astype(np.float32)
        name = info["path"].split("/")[-1].rsplit(".", 1)[0]
        if bool(getattr(self.config, "INFERENCE_MATCH_TRAIN_NORMALIZATION", False)):
            mu = float(image.mean());
            sd = float(image.std())
            if sd > 0.0:
                image = (image - mu) / sd
        # robust active_class_ids
        num = int(getattr(self.config, "NUM_CLASSES", getattr(self.dataset, "num_classes", 1)))
        active_class_ids = np.zeros([num], dtype=np.int32)
        try:
            src_ids = list(self.dataset.source_class_ids.get(info["source"], []))
            src_ids = [int(i) for i in src_ids if 0 <= int(i) < num]
            if src_ids:
                active_class_ids[src_ids] = 1
            else:
                if num > 1:
                    active_class_ids[1:] = 1
        except Exception:
            if num > 1:
                active_class_ids[1:] = 1

        # ВАЖНО: фон всегда активен (BG=1), иначе лосс/логика класса 0 деградирует
        if active_class_ids.shape[0] > 0:
            active_class_ids[0] = 1

        image_meta = compose_image_meta(
            image_id=image_id,
            original_image_shape=tuple(self.config.IMAGE_SHAPE),
            image_shape=tuple(self.config.IMAGE_SHAPE),
            window=(0, 0, 0, *self.config.IMAGE_SHAPE[:-1]),
            scale=1.0,
            active_class_ids=active_class_ids
        ).astype(np.float32)

        batch_images = np.zeros((1,) + image.shape, dtype=np.float32);
        batch_images[0] = image
        batch_meta = np.zeros((1,) + image_meta.shape, dtype=np.float32);
        batch_meta[0] = image_meta
        batch_anchors = self.anchors[np.newaxis].astype(np.float32)

        try:
            act_idx = np.nonzero(active_class_ids)[0]
            print(f"[DEBUG] active_class_ids: total_active={int(active_class_ids.sum())}, sample={act_idx[:8]}")
        except Exception:
            pass

        return name, [batch_images, batch_meta, batch_anchors]

    def load_image_gt(self, image_id):
        """Load image and (если training) GT, с корректным active_class_ids под NUM_CLASSES."""
        import numpy as np

        # Load image
        image = self.dataset.load_image(image_id)

        # Собираем meta и active_class_ids одинаково для train/infer
        num = int(getattr(self.config, "NUM_CLASSES", getattr(self.dataset, "num_classes", 1)))
        active_class_ids = np.zeros([num], dtype=np.int32)
        try:
            src = self.dataset.image_info[image_id]["source"]
            src_ids = list(self.dataset.source_class_ids.get(src, []))
            src_ids = [int(i) for i in src_ids if 0 <= int(i) < num]
            if src_ids:
                active_class_ids[src_ids] = 1
            else:
                if num > 1:
                    active_class_ids[1:] = 1
        except Exception:
            if num > 1:
                active_class_ids[1:] = 1

        image_meta = compose_image_meta(
            image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE),
            (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1.0, active_class_ids
        )

        if self.training:
            boxes, class_ids, masks = self.dataset.load_data(image_id)
            return image, image_meta, boxes, class_ids, masks
        else:
            return image, image_meta


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, D, C] before resizing or padding.
    image_shape: [H, W, D, C] after resizing and padding
    window: (y1, x1, z1, y2, x2, z2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [int(image_id)] +  # size=1
        list(original_image_shape) +  # size=4
        list(image_shape) +  # size=4
        list(window) +  # size=6 (y1, x1, z1, y2, x2, z2) in image coordinates
        [float(scale)] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta #18 for num_classes = 2


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    shapes = []
    for stride in config.BACKBONE_STRIDES:
        if isinstance(stride, (int, np.integer)):
            sy = sx = sz = int(stride)
        else:
            sy, sx, sz = stride
        shapes.append([
            int(math.ceil(image_shape[0] / sy)),
            int(math.ceil(image_shape[1] / sx)),
            int(math.ceil(image_shape[2] / sz)),
        ])
    return np.array(shapes, dtype=np.int32)


############################################################
#  Dataset Objects
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

    def subset(self, image_ids):
        import numpy as np, copy
        view = copy.copy(self)  # неглубокая копия объекта
        view._image_ids = np.asarray(image_ids, dtype=np.int32)
        return view
    def filter_positive(self):
        """Оставляет только те image_ids, где есть хотя бы один позитивный объект.
        RPN: по boxes; HEAD: по target_class_ids>0.
        Сделано максимально дёшево по I/O.
        """
        keep = []
        for i in self._image_ids:
            info = self.image_info[i]

            # RPN-датасет (есть cab_path → читаем только боксы через load_data(..., masks_needed=False))
            if "cab_path" in info:
                boxes, _, _ = self.load_data(i, masks_needed=False)
                has_pos = boxes is not None and boxes.shape[0] > 0

            # HEAD-датасет (после target generation)
            elif ("tci_path" in info) or ("ra_path" in info) or ("target_class_ids" in info):
                tci_path = info.get("tci_path")
                if tci_path is None:
                    # fallback — вдруг структура иная: пробуем load_data, но это дороже
                    try:
                        _, _, target_class_ids, _, _ = self.load_data(i)
                        has_pos = target_class_ids is not None and np.any(target_class_ids > 0)
                    except Exception:
                        has_pos = True  # не рискуем выкинуть; лучше оставить
                else:
                    try:
                        # читаем ТОЛЬКО target_class_ids, mmap для дешёвого доступа
                        tci = np.load(tci_path, mmap_mode='r', allow_pickle=False)
                        # Важно: не принуждаем к .copy(), просто проверка >0
                        has_pos = tci.size > 0 and np.any(tci > 0)
                    except Exception:
                        # если файл битый/недоступен — лучше оставить, чем потерять сэмпл молча
                        has_pos = True

            else:
                # неизвестный тип записи — не фильтруем
                has_pos = True

            if has_pos:
                keep.append(i)

        self._image_ids = np.asarray(keep, dtype=np.int32)
        print(f"[Dataset] positive patches: {len(keep)} / {self.num_images}", flush=True)
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
        self._image_ids = np.arange(self.num_images, dtype=np.int32)
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


class ToyDataset(Dataset):
    """Улучшенный датасет для нейронных данных с лучшей предобработкой."""

    def load_dataset(self, data_dir, is_train=True):
        import os, pandas as pd
        self.add_class("dataset", 1, "neuron")

        split = "train" if is_train else "test"
        csv_path = os.path.join(data_dir, "datasets", f"{split}.csv")
        td = pd.read_csv(csv_path, sep=None, engine="python")

        cols = {c.lower(): c for c in td.columns}

        def pick(*cands, required=True):
            for c in cands:
                k = c.lower()
                if k in cols: return cols[k]
                for lc, orig in cols.items():
                    if k in lc: return orig
            if required:
                raise KeyError(f"[Dataset.load_dataset] none of columns {cands} found. Available: {list(td.columns)}")
            return None

        col_images = pick("images", "image", "img", "path", "image_path")
        col_segs = pick("segs", "seg", "seg_path", "labels", "label_path", required=False)
        col_cabs = pick("cabs", "cab", "boxes", "cab_path")
        col_masks = pick("masks", "mask", "masks_path", "mask_path")

        print(f"[Dataset] Using columns -> images:'{col_images}', segs:'{col_segs}', "
              f"cabs:'{col_cabs}', masks:'{col_masks}'", flush=True)

        for i in range(len(td)):
            img_path = td.at[i, col_images]
            seg_path = td.at[i, col_segs] if col_segs is not None else None
            cab_path = td.at[i, col_cabs]
            m_path = td.at[i, col_masks]
            if not isinstance(img_path, str): raise ValueError(f"[load_dataset] bad 'images' at row {i}")
            if not isinstance(cab_path, str): raise ValueError(f"[load_dataset] bad 'cabs' at row {i}")
            if not isinstance(m_path, str):   raise ValueError(f"[load_dataset] bad 'masks' at row {i}")
            self.add_image('dataset', image_id=i, path=img_path, seg_path=seg_path, cab_path=cab_path, m_path=m_path)

        print('Training dataset is loaded.' if is_train else 'Validation dataset is loaded.', flush=True)

    def load_image(self, image_id, z_slice=None):
        """
        Улучшенная загрузка для нейронных данных с z-score нормализацией.
        Возвращает [H, W, D, 1] float32.
        """
        info = self.image_info[image_id]
        image = imread(info["path"])  # ожидается (Z, Y, X)
        image = np.transpose(image, (1, 2, 0))  # -> (Y, X, Z)

        # Z-score нормализация лучше для нейронных данных
        image = image.astype(np.float32)

        # Убираем выбросы (улучшает контраст нейронов)
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)

        # Z-score нормализация
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val > 0:
            image = (image - mean_val) / std_val
        else:
            image = image - mean_val

        # Дополнительное масштабирование для стабильности
        image = np.tanh(image * 0.5)  # мягкое ограничение [-1, 1]

        return image[..., np.newaxis].astype(np.float32, copy=False)

    def load_data(self, image_id, masks_needed=True):
        """
        Улучшенная загрузка данных с валидацией боксов для нейронов.
        """
        import bz2, _pickle as cPickle, numpy as np
        info = self.image_info[image_id]

        # Загрузка и валидация боксов
        cabs = np.loadtxt(info["cab_path"], ndmin=2, dtype=np.int32)
        if cabs.size:
            if cabs.ndim == 1:
                cabs = cabs.reshape((1, -1))
            boxes = cabs[:, [2, 3, 1, 5, 6, 4]]  # (z,y,x,z,y,x) -> (y1,x1,z1,y2,x2,z2)
            class_ids = cabs[:, 0]

            # Валидация боксов для нейронных структур
            valid_mask = (
                    (boxes[:, 3] > boxes[:, 0]) &  # y2 > y1
                    (boxes[:, 4] > boxes[:, 1]) &  # x2 > x1
                    (boxes[:, 5] > boxes[:, 2]) &  # z2 > z1
                    (boxes[:, 0] >= 0) &  # y1 >= 0
                    (boxes[:, 1] >= 0) &  # x1 >= 0
                    (boxes[:, 2] >= 0)  # z1 >= 0
            )

            if not np.all(valid_mask):
                print(f"[Dataset][{image_id}] Warning: {np.sum(~valid_mask)} invalid boxes removed")
                boxes = boxes[valid_mask]
                class_ids = class_ids[valid_mask]
        else:
            boxes = np.zeros((0, 6), dtype=np.int32)
            class_ids = np.zeros((0,), dtype=np.int32)

        if not masks_needed:
            return boxes, class_ids, None

        if boxes.shape[0] == 0:
            img = imread(info["path"])
            H, W, D = img.shape[1], img.shape[2], img.shape[0]
            masks = np.zeros((H, W, D, 0), dtype=bool)
        else:
            try:
                with bz2.BZ2File(info["m_path"], 'rb') as f:
                    m = cPickle.load(f)  # (Z, Y, X, N)
                masks = np.transpose(m, (1, 2, 0, 3))  # -> (Y, X, Z, N)

                # Обеспечиваем соответствие количества масок и боксов
                if masks.shape[-1] != boxes.shape[0]:
                    min_count = min(masks.shape[-1], boxes.shape[0])
                    if min_count > 0:
                        masks = masks[..., :min_count]
                        boxes = boxes[:min_count]
                        class_ids = class_ids[:min_count]
                    else:
                        H, W, D = masks.shape[:3]
                        masks = np.zeros((H, W, D, 0), dtype=bool)
                        boxes = np.zeros((0, 6), dtype=np.int32)
                        class_ids = np.zeros((0,), dtype=np.int32)

            except Exception as e:
                print(f"[Dataset][{image_id}] Mask loading failed: {e}")
                img = imread(info["path"])
                H, W, D = img.shape[1], img.shape[2], img.shape[0]
                masks = np.zeros((H, W, D, 0), dtype=bool)

        return boxes, class_ids, masks



class ToyHeadDataset(Dataset):
    """Head-датасет после TARGET_GENERATION: гибко читаем CSV c rois_aligned/target_*."""

    def __init__(self, *args, **kwargs):
        try: super(ToyHeadDataset, self).__init__()
        except Exception: pass
        if not hasattr(self, "image_info") or self.image_info is None:
            self.image_info = []
        cfg = kwargs.get("config", None)
        if cfg is not None: self.config = cfg
        if not hasattr(self, "num_classes") or self.num_classes is None:
            try:    self.num_classes = int(getattr(self.config, "NUM_CLASSES", 2))
            except Exception: self.num_classes = 2

    def load_dataset(self, data_dir, is_train=True):
        """Обязательные поля: rois_aligned/ra_path И target_class_ids/tci_path (названия гибкие)."""
        import os, pandas as pd
        self.add_class("dataset", 1, "neuron")

        split = "train" if is_train else "test"
        csv_path = os.path.join(data_dir, "datasets", f"{split}.csv")
        td = pd.read_csv(csv_path, sep=None, engine="python")
        cols = {c.lower(): c for c in td.columns}

        def pick(*cands, required=True):
            for c in cands:
                k = c.lower()
                if k in cols: return cols[k]
                for lc, orig in cols.items():
                    if k in lc: return orig
            if required:
                raise KeyError(f"[ToyHeadDataset.load_dataset] none of columns {cands} found. Available: {list(td.columns)}")
            return None

        col_rois = pick("rois", "rois_path", required=False)
        col_ra   = pick("rois_aligned", "ra_path", "aligned_rois", "roisAligned", required=True)
        col_ma   = pick("mask_aligned", "ma_path", "aligned_mask", required=False)
        col_tci  = pick("target_class_ids", "tci_path", "tci", required=True)
        col_tb   = pick("target_bbox", "tb_path", "bbox", required=False)
        col_tm   = pick("target_mask", "tm_path", "tm", required=False)

        print(f"[ToyHeadDataset] Using columns -> rois:'{col_rois}', ra:'{col_ra}', ma:'{col_ma}', "
              f"tci:'{col_tci}', tb:'{col_tb}', tm:'{col_tm}'", flush=True)

        for i in range(len(td)):
            r_path   = td.at[i, col_rois] if col_rois is not None else None
            ra_path  = td.at[i, col_ra]
            ma_path  = td.at[i, col_ma] if col_ma is not None else None
            tci_path = td.at[i, col_tci]
            tb_path  = td.at[i, col_tb] if col_tb is not None else None
            tm_path  = td.at[i, col_tm] if col_tm is not None else None

            # обязательный image_info['path'] — rois, иначе ra, иначе ma
            path = r_path if (isinstance(r_path, str) and r_path) else (ra_path if isinstance(ra_path, str) and ra_path else ma_path)
            if not isinstance(path, str) or not path:
                raise ValueError(f"[ToyHeadDataset] bad path fallback at row {i} (no rois/ra/ma string value)")

            self.add_image('dataset', image_id=i, path=path,
                           ra_path=ra_path, ma_path=ma_path,
                           tci_path=tci_path, tb_path=tb_path, tm_path=tm_path)

        print('Training dataset is loaded.' if is_train else 'Validation dataset is loaded.', flush=True)

    def load_image(self, image_id, z_slice=None):
        """Нужен только для совместимости проверок. Возвращаем dummy [H,W,D,1] из конфига."""
        H, W, D = tuple(getattr(self, "config", None).IMAGE_SHAPE[:3])
        return np.zeros((H, W, D, 1), dtype=np.float32)

    def load_data(self, image_id):
        """Возвращает РОВНО 5 массивов для головы:
        rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask
        (все — numpy.ndarray; маски распаковываются из packbits; при отсутствии масок возвращаются нули)."""
        import numpy as np
        info = self.image_info[image_id]

        def _load_any(path):
            if path is None:
                return {"arr_0": None}
            p = str(path)
            if p.endswith(".npz"):
                return np.load(p, allow_pickle=False)
            arr = np.load(p, allow_pickle=False)
            return {"arr_0": arr}

        def _pick(z, *keys, fallback="arr_0"):
            for k in keys:
                if z is not None and k in z:
                    return z[k]
            if z is None:
                return None
            if hasattr(z, "files") and z.files:
                return z[z.files[0]]
            return z[fallback]

        def _unbit(z, bits_key, shape_key, fallback_key):
            """Распаковка битовых масок: (bits, shape) -> ndarray по shape."""
            if z is None:
                return None
            if bits_key in z and shape_key in z:
                bits = z[bits_key]
                shape = z[shape_key].astype(np.int64)
                flat = np.unpackbits(bits)
                need = int(np.prod(shape))
                if flat.shape[0] < need:
                    flat = np.pad(flat, (0, need - flat.shape[0]), mode="constant")
                arr = flat[:need].reshape(tuple(shape)).astype(np.uint8, copy=False)
                return arr
            return _pick(z, fallback_key)

        # читаем все артефакты
        z_ra = _load_any(info.get("ra_path"))
        z_ma = _load_any(info.get("ma_path"))
        z_tci = _load_any(info.get("tci_path"))
        z_tb = _load_any(info.get("tb_path"))
        z_tm = _load_any(info.get("tm_path"))

        rois_aligned = _pick(z_ra, "rois_aligned", "ra", "arr_0")
        mask_aligned = _unbit(z_ma, "mask_bits", "mask_shape", "mask_aligned")
        target_class_ids = _pick(z_tci, "tci", "target_class_ids", "arr_0")
        target_bbox = _pick(z_tb, "bbox", "target_bbox", "arr_0")
        target_mask = _unbit(z_tm, "tm_bits", "tm_shape", "tm")

        # приведение форм и типов
        if rois_aligned is not None:
            rois_aligned = np.asarray(rois_aligned, dtype=np.float32)
        if target_class_ids is not None:
            target_class_ids = np.asarray(target_class_ids, dtype=np.int32)
        if target_bbox is not None:
            target_bbox = np.asarray(target_bbox, dtype=np.float32)

        # ИСПРАВЛЕНИЕ: НЕ убираем канальную ось, а гарантируем её наличие
        def _ensure_5d_mask(x):
            """Гарантируем форму [T, mH, mW, mD, C]"""
            if x is None:
                return None
            x = np.asarray(x)
            if x.ndim == 4:  # [T, mH, mW, mD] -> [T, mH, mW, mD, 1]
                return x[..., np.newaxis]
            elif x.ndim == 5:  # уже [T, mH, mW, mD, C]
                return x
            else:
                return None

        mask_aligned = _ensure_5d_mask(mask_aligned)
        target_mask = _ensure_5d_mask(target_mask)

        # инференция размеров для фолбэков
        def _infer_T():
            if rois_aligned is not None and rois_aligned.ndim >= 1:
                return int(rois_aligned.shape[0])
            if target_class_ids is not None and target_class_ids.ndim >= 1:
                return int(target_class_ids.shape[0])
            return 0

        T = _infer_T()

        # ИСПРАВЛЕННЫЕ фолбэки с правильными размерностями
        if mask_aligned is None:
            # Используем размеры из конфига
            cfg = getattr(self, "config", None)
            if cfg:
                M = int(getattr(cfg, "MASK_POOL_SIZE", 14))
                C = int(getattr(cfg, "TOP_DOWN_PYRAMID_SIZE", 256))
            else:
                M, C = 14, 256
            mask_aligned = np.zeros((T, M, M, M, C), dtype=np.float32)
        else:
            mask_aligned = mask_aligned.astype(np.float32, copy=False)

        if target_mask is None:
            # Используем MASK_SHAPE из конфига
            cfg = getattr(self, "config", None)
            if cfg and hasattr(cfg, "MASK_SHAPE"):
                M = int(cfg.MASK_SHAPE[0])
            else:
                M = 28  # значение по умолчанию
            target_mask = np.zeros((T, M, M, M, 1), dtype=np.float32)
        else:
            target_mask = target_mask.astype(np.float32, copy=False)

        return rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask


############################################################
#  Target Generators
############################################################


def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    """
    anchors: [N, (y1,x1,z1,y2,x2,z2)]
    gt_class_ids: [M]
    gt_boxes: [M, (y1,x1,z1,y2,x2,z2)]
    Returns:
      rpn_match: [N] in {-1,0,1}
      rpn_bbox:  [RPN_TRAIN_ANCHORS_PER_IMAGE, 6] — только для pos, остальное нули
    """
    import numpy as np

    # --- телеметрия GT (без падений, если Telemetry не доступен)
    try:
        from core.utils import Telemetry
        Telemetry.update_gt_stats(gt_boxes, config)
    except Exception:
        pass

    N = anchors.shape[0]
    rpn_match = np.zeros((N,), dtype=np.int8)
    rpn_bbox  = np.zeros((int(getattr(config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 256)), 6), dtype=np.float32)

    if gt_boxes is None or gt_boxes.size == 0:
        # все негативы
        rpn_match[:] = -1
        try:
            from core.utils import Telemetry
            Telemetry.update_rpn_targets(anchors, np.zeros((N,), dtype=np.float32), rpn_match, config)
        except Exception:
            pass
        return rpn_match, rpn_bbox

    # IoU между anchors и GT (векторизовано)
    def _overlaps(A, B):
        # A: [N,6], B: [M,6]  ->  [N,M]
        Ay1, Ax1, Az1, Ay2, Ax2, Az2 = [A[:,i] for i in range(6)]
        By1, Bx1, Bz1, By2, Bx2, Bz2 = [B[:,i] for i in range(6)]
        # broadcasting
        y1 = np.maximum(Ay1[:,None], By1[None,:]);  y2 = np.minimum(Ay2[:,None], By2[None,:])
        x1 = np.maximum(Ax1[:,None], Bx1[None,:]);  x2 = np.minimum(Ax2[:,None], Bx2[None,:])
        z1 = np.maximum(Az1[:,None], Bz1[None,:]);  z2 = np.minimum(Az2[:,None], Bz2[None,:])
        inter = np.maximum(y2-y1,0)*np.maximum(x2-x1,0)*np.maximum(z2-z1,0)
        aA = np.maximum((Ay2-Ay1)*(Ax2-Ax1)*(Az2-Az1), 1e-6)[:,None]
        aB = np.maximum((By2-By1)*(Bx2-Bx1)*(Bz2-Bz1), 1e-6)[None,:]
        union = aA + aB - inter
        return inter / np.maximum(union, 1e-6)

    overlaps = _overlaps(anchors.astype(np.float32), gt_boxes.astype(np.float32))

    # Центры
    a_c = 0.5 * (anchors[:, :3] + anchors[:, 3:])
    g_c = 0.5 * (gt_boxes[:, :3] + gt_boxes[:, 3:])

    # Гиперпараметры
    topk = int(getattr(config, "ATSS_TOPK", 12))
    min_pos_per_gt = int(getattr(config, "ATSS_MIN_POS_PER_GT", 3))
    pos_iou = float(getattr(config, "POSITIVE_IOU", 0.3))
    neg_iou = float(getattr(config, "NEGATIVE_IOU", 0.1))
    total_target_anchors = int(getattr(config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 256))
    pos_ratio = float(getattr(config, "RPN_POSITIVE_RATIO", 0.5))
    target_pos = max(1, int(total_target_anchors * pos_ratio))

    pos_idx = set()
    # ATSS-лайт по каждому GT
    for j in range(gt_boxes.shape[0]):
        d = np.linalg.norm(a_c - g_c[j], axis=1)
        cand = np.argsort(d)[:max(topk,1)]
        ious = overlaps[cand, j]
        mu, sigma = float(np.mean(ious)), float(np.std(ious))
        thr = max(pos_iou, mu + sigma)
        j_pos = cand[ious >= thr]
        if j_pos.size == 0:
            k = max(min_pos_per_gt, 1)
            j_pos = cand[np.argsort(-ious)[:k]]
        # гарантируем минимум на GT
        if j_pos.size < min_pos_per_gt:
            extra = cand[np.argsort(-ious)[:min_pos_per_gt]]
            j_pos = np.unique(np.concatenate([j_pos, extra]))
        for i in j_pos:
            pos_idx.add(int(i))

    pos_idx = np.array(sorted(list(pos_idx)), dtype=np.int64)
    rpn_match[pos_idx] = 1

    # Негативы: max IoU < neg_iou
    anchor_iou_max = overlaps.max(axis=1)
    rpn_match[(anchor_iou_max < neg_iou) & (rpn_match != 1)] = -1

    # --- телеметрия до балансировки
    try:
        from core.utils import Telemetry
        Telemetry.update_rpn_targets(anchors, anchor_iou_max, rpn_match, config)
        Telemetry._cnt['rpn_pos_total'] += int(np.sum(rpn_match == 1))
        Telemetry._cnt['rpn_neg_total'] += int(np.sum(rpn_match == -1))
    except Exception:
        pass

    # Баланс: ограничиваем положительные и негативы под батч
    pos_count = int(np.sum(rpn_match == 1))
    keep_pos = min(pos_count, target_pos)
    if pos_count > keep_pos:
        pos_ids = np.where(rpn_match == 1)[0]
        drop = np.random.choice(pos_ids, size=pos_count - keep_pos, replace=False)
        rpn_match[drop] = 0

    neg_ids = np.where(rpn_match == -1)[0]
    target_neg = int(min(len(neg_ids), total_target_anchors - int(np.sum(rpn_match == 1))))
    if len(neg_ids) > target_neg:
        drop = np.random.choice(neg_ids, size=len(neg_ids) - target_neg, replace=False)
        rpn_match[drop] = 0

    # --- телеметрия после балансировки
    try:
        Telemetry._cnt['rpn_pos_kept'] += int(np.sum(rpn_match == 1))
        Telemetry._cnt['rpn_neg_kept'] += int(np.sum(rpn_match == -1))
    except Exception:
        pass

    # Заполняем rpn_bbox (только для pos)
    std = np.array(getattr(config, "RPN_BBOX_STD_DEV", [0.1,0.1,0.1,0.2,0.2,0.2]), dtype=np.float32)
    ids = np.where(rpn_match == 1)[0]
    if ids.size:
        # сопоставляем каждому anchor лучший GT по IoU
        gt_of_anchor = np.argmax(overlaps, axis=1)
        ix = 0
        for i in ids:
            a = anchors[i].astype(np.float32)
            g = gt_boxes[gt_of_anchor[i]].astype(np.float32)
            ah, aw, ad = max(a[3]-a[0],1e-6), max(a[4]-a[1],1e-6), max(a[5]-a[2],1e-6)
            gh, gw, gd = max(g[3]-g[0],1e-6), max(g[4]-g[1],1e-6), max(g[5]-g[2],1e-6)
            acy, acx, acz = (a[0]+a[3])/2, (a[1]+a[4])/2, (a[2]+a[5])/2
            gcy, gcx, gcz = (g[0]+g[3])/2, (g[1]+g[4])/2, (g[2]+g[5])/2
            dy, dx, dz = (gcy-acy)/ah, (gcx-acx)/aw, (gcz-acz)/ad
            dh, dw, dd = np.log(gh/ah), np.log(gw/aw), np.log(gd/ad)
            rpn_bbox[ix] = np.array([dy, dx, dz, dh, dw, dd], dtype=np.float32) / std
            ix += 1
            if ix >= rpn_bbox.shape[0]:
                break

    return rpn_match, rpn_bbox