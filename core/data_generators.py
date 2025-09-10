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

    def __init__(self, dataset, config, shuffle=True, training=True):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.training = training
        self.batch_size = int(getattr(self.config, "BATCH_SIZE", 1))
        assert self.batch_size == 1, "HeadGenerator рассчитан на BATCH_SIZE==1 для предсказуемого VRAM"
        self._call_count = 0
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    @staticmethod
    def _downsample_2x_mean(x):
        """
        x: [T, H, W, D, C], H=W=D кратны 2 -> усреднение 2x2x2 -> [T, H/2, W/2, D/2, C]
        """
        T, H, W, D, C = x.shape
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, "Downsample expects even spatial dims"
        x = x.reshape(T, H//2, 2, W//2, 2, D//2, 2, C)
        return x.mean(axis=(2, 4, 6))

    def __getitem__(self, idx):
        # один image_id (B=1)
        image_id = int(self.image_ids[idx])

        # dataset обязан вернуть:
        # rois_aligned [N, P,P,P, C], mask_aligned [N, M,M,M, C],
        # target_class_ids [N], target_bbox [N, 6], target_mask [N, mH,mW,mD]
        rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = \
            self.dataset.load_data(image_id)

        H, W, D = int(self.config.IMAGE_SHAPE[0]), int(self.config.IMAGE_SHAPE[1]), int(self.config.IMAGE_SHAPE[2])

        # ЕСЛИ нужно ровно так же, как в RPN "targeting":
        # window = (0, 0, 0, *self.config.IMAGE_SHAPE[:-1])  # (y1,x1,z1,y2,x2,z2) без -1
        window = (0, 0, 0, H, W, D)

        active_class_ids = np.ones([self.dataset.num_classes], dtype=np.int32)


        image_meta = compose_image_meta(
            image_id=image_id,
            original_image_shape=(H, W, D, 1),
            image_shape=(H, W, D, 1),
            window=window,
            scale=1.0,
            active_class_ids=active_class_ids
        ).astype(np.float32)  # см. аналогичную сборку мета в твоей RPN "targeting" ветке :contentReference[oaicite:0]{index=0}

        # Типы
        rois_aligned = rois_aligned.astype(np.float32, copy=False)
        mask_aligned = mask_aligned.astype(np.float32, copy=False)
        target_class_ids = target_class_ids.astype(np.int32, copy=False)
        target_bbox = target_bbox.astype(np.float32, copy=False)

        # target_mask: [T, mH, mW, mD] -> [T, mH, mW, mD, 1]
        if target_mask.ndim == 4:
            target_mask = target_mask[..., np.newaxis]
        elif target_mask.ndim != 5:
            raise ValueError(f"[HeadGenerator] target_mask rank={target_mask.ndim}, need [T,mH,mW,mD,1]")
        target_mask = target_mask.astype(np.float32, copy=False)

        # Балансировка/чанк T ROI
        T = int(getattr(self.config, "TRAIN_ROIS_PER_IMAGE", 128))
        total = int(rois_aligned.shape[0])
        order = np.arange(total, dtype=np.int32)
        if self.training and bool(getattr(self.config, "HEAD_SHUFFLE_ROIS", True)):
            np.random.shuffle(order)
        pos_idx = order[target_class_ids[order] > 0]
        neg_idx = order[target_class_ids[order] <= 0]

        if self.training and bool(getattr(self.config, "HEAD_BALANCE_POS", True)):
            pos_frac = float(getattr(self.config, "HEAD_POS_FRAC", 0.33))
            pos_cnt = int(round(T * pos_frac))
            neg_cnt = T - pos_cnt
            if pos_idx.size == 0:
                batch_idx = np.random.choice(neg_idx, size=T, replace=neg_idx.size < T)
            elif neg_idx.size == 0:
                batch_idx = np.random.choice(pos_idx, size=T, replace=pos_idx.size < T)
            else:
                pos_pick = np.random.choice(pos_idx, size=pos_cnt, replace=pos_idx.size < pos_cnt)
                neg_pick = np.random.choice(neg_idx, size=neg_cnt, replace=neg_idx.size < neg_cnt)
                batch_idx = np.concatenate([pos_pick, neg_pick])
                np.random.shuffle(batch_idx)
        else:
            num_chunks = int(np.ceil(total / float(T))) if T > 0 else 1
            chunk_id = self._call_count % max(1, num_chunks)
            self._call_count += 1
            start, end = int(chunk_id * T), min(int(chunk_id * T) + T, total)
            batch_idx = order[start:end]
            if len(batch_idx) < T:
                pad = T - len(batch_idx)
                batch_idx = np.pad(batch_idx, (0, pad), mode='edge')

        # Собираем батч ROI
        rois_aligned = rois_aligned[batch_idx]
        mask_aligned = mask_aligned[batch_idx]
        target_class_ids = target_class_ids[batch_idx]
        target_bbox = target_bbox[batch_idx]
        target_mask = target_mask[batch_idx]

        # Приводим mask_aligned к config.MASK_POOL_SIZE (допускается только аккуратный 2x даунсемпл)
        Mconf = int(getattr(self.config, "MASK_POOL_SIZE", 7))
        Mh, Mw, Md = mask_aligned.shape[1:4]
        if (Mh, Mw, Md) != (Mconf, Mconf, Mconf):
            if (Mh, Mw, Md) == (2*Mconf, 2*Mconf, 2*Mconf):
                mask_aligned = self._downsample_2x_mean(mask_aligned)
            else:
                raise ValueError(f"[HeadGenerator] mask_aligned={Mh}x{Mw}x{Md}, but MASK_POOL_SIZE={Mconf}")

        # Оборачиваем всё в батч-ось
        x = [
            np.expand_dims(rois_aligned, 0),     # [1, T, P, P, P, C]
            np.expand_dims(mask_aligned, 0),     # [1, T, M, M, M, C]
            np.expand_dims(image_meta, 0),       # [1, META]
            np.expand_dims(target_class_ids, 0), # [1, T]
            np.expand_dims(target_bbox, 0),      # [1, T, 6]
            np.expand_dims(target_mask, 0),      # [1, T, mH, mW, mD, 1]
        ]
        y = []  # лоссы через add_loss
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)







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
          - изображение через dataset.load_image(image_id) (у вас там уже шкалирование в [-1, 1])
          - active_class_ids как в __getitem__: включены классы текущего source
          - корректная image_meta через compose_image_meta(...)
          - anchors: нормированные self.anchors
        Возвращает:
          name  : базовое имя файла (без пути и расширения)
          inputs: [images, image_meta, anchors] с батч-измерением
        """
        import numpy as np

        info = self.dataset.image_info[image_id]
        image = self.dataset.load_image(image_id).astype(np.float32)  # (H, W, D, 1) — как в train
        name = info["path"].split("/")[-1].rsplit(".", 1)[0]

        # active_class_ids — как в __getitem__
        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[info["source"]]
        active_class_ids[source_class_ids] = 1

        # meta: original=image_shape=CONFIG.IMAGE_SHAPE, окно — вся картинка, scale=1.0
        image_meta = compose_image_meta(
            image_id=image_id,
            original_image_shape=tuple(self.config.IMAGE_SHAPE),
            image_shape=tuple(self.config.IMAGE_SHAPE),
            window=(0, 0, 0, *self.config.IMAGE_SHAPE[:-1]),  # (y1,x1,z1,y2,x2,z2)
            scale=1.0,
            active_class_ids=active_class_ids
        ).astype(np.float32)

        # батч=1
        batch_images = np.zeros((1,) + image.shape, dtype=np.float32)
        batch_images[0] = image
        batch_meta = np.zeros((1,) + image_meta.shape, dtype=np.float32)
        batch_meta[0] = image_meta
        batch_anchors = self.anchors[np.newaxis].astype(np.float32)

        return name, [batch_images, batch_meta, batch_anchors]

    def load_image_gt(self, image_id):
        """Load image and ground truth data with consistent processing."""
        # Load image with proper preprocessing
        image = self.dataset.load_image(image_id)

        # In inference mode, we only need image and meta
        if not self.training:
            active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
            source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
            active_class_ids[source_class_ids] = 1
            image_meta = compose_image_meta(
                image_id,
                tuple(self.config.IMAGE_SHAPE),
                tuple(self.config.IMAGE_SHAPE),
                (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]),  # Full image window
                1.0,  # Scale = 1.0
                active_class_ids
            )

            # Also load GT data for metrics calculation during evaluation
            try:
                boxes, class_ids, masks = self.dataset.load_data(image_id)
                return image, image_meta, boxes, class_ids, masks
            except:
                return image, image_meta

        # For training, load all GT data
        boxes, class_ids, masks = self.dataset.load_data(image_id)
        return image, image_meta, boxes, class_ids, masks


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

    def load_dataset(self, data_dir, is_train=True):
        """
        Универсальный лоадер:
        - поддерживает названия колонок: images|image|img|path|image_path,
                                         segs|seg|seg_path|labels|label_path,
                                         cabs|cab|boxes|cab_path,
                                         masks|mask|masks_path|mask_path
        - автоматически определяет разделитель CSV (',' или ';')
        - печатает, какие колонки реально использованы
        """
        import pandas as pd
        import numpy as np
        import os

        # объявим классы как в исходнике (оставляю как было)
        cfg = getattr(self, "config", None)
        if cfg and hasattr(cfg, "CLASS_NAMES"):
            class_names = list(cfg.CLASS_NAMES)  # напр. ["neuron"] или ["rat_cell"]
        elif cfg and hasattr(cfg, "NUM_CLASSES") and int(cfg.NUM_CLASSES) == 2:
            class_names = ["neuron"]  # один foreground (+ BG)
        else:
            class_names = ["ellipsoid", "cuboid", "pyramid"]  # бэкап как в исходнике

        for i, name in enumerate(class_names, start=1):
            self.add_class("dataset", i, name)

        split = "train" if is_train else "test"
        csv_path = os.path.join(data_dir, "datasets", f"{split}.csv")

        # читаем CSV с авто-определением разделителя
        try:
            td = pd.read_csv(csv_path, sep=None, engine="python")
        except Exception as e:
            raise RuntimeError(f"[load_dataset] failed to read CSV: {csv_path} ({type(e).__name__}: {e})")

        cols = {c.lower(): c for c in td.columns}  # lower->original

        def pick(*candidates, required=True):
            """берём первую существующую колонку из списка кандидатов (по lowercase-совпадению)"""
            for cand in candidates:
                key = cand.lower()
                if key in cols:
                    return cols[key]
                # мягкий поиск по подстроке
                for lc, orig in cols.items():
                    if key in lc:
                        return orig
            if required:
                raise KeyError(f"[load_dataset] none of columns {candidates} found in CSV. "
                               f"Available: {list(td.columns)}")
            return None

        # подбираем реальные имена колонок
        col_images = pick("images", "image", "img", "path", "image_path")
        col_segs   = pick("segs", "seg", "seg_path", "labels", "label_path", required=False)
        col_cabs   = pick("cabs", "cab", "boxes", "cab_path")
        col_masks  = pick("masks", "mask", "masks_path", "mask_path")

        print(f"[Dataset] Using columns -> images:'{col_images}', segs:'{col_segs}', "
              f"cabs:'{col_cabs}', masks:'{col_masks}'")

        # добавляем изображения
        n = len(td)
        for i in range(n):
            img_path = td.at[i, col_images]
            seg_path = td.at[i, col_segs]  if col_segs  is not None else None
            cab_path = td.at[i, col_cabs]
            m_path   = td.at[i, col_masks]

            if not isinstance(img_path, str):
                raise ValueError(f"[load_dataset] bad 'images' value at row {i}: {img_path}")
            if not isinstance(cab_path, str):
                raise ValueError(f"[load_dataset] bad 'cabs' value at row {i}: {cab_path}")
            if not isinstance(m_path, str):
                raise ValueError(f"[load_dataset] bad 'masks' value at row {i}: {m_path}")

            self.add_image(
                'dataset',
                image_id=i,
                path=img_path,
                seg_path=seg_path,
                cab_path=cab_path,
                m_path=m_path
            )

        print('Training dataset is loaded.' if is_train else 'Validation dataset is loaded.')

    def load_image(self, image_id, z_slice=None):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        image = imread(info["path"])
        image = np.transpose(image, (1, 2, 0))
        image = 2 * (image / 255) - 1
        image = image[..., np.newaxis]
        # print(image.shape)
        return image


    def load_data(self, image_id, masks_needed=True):
        info = self.image_info[image_id]
        cabs = np.loadtxt(info["cab_path"], ndmin=2, dtype=np.int32)
        if cabs.size:
            if cabs.ndim == 1:
                cabs = cabs.reshape((1, -1))
            # z y x z y x  →  y x z y x z
            boxes = cabs[:, [2, 3, 1, 5, 6, 4]]
            class_ids = cabs[:, 0]
        else:
            boxes = np.zeros((0, 6), dtype=np.int32)
            class_ids = np.zeros((0,), dtype=np.int32)

            # --- маски --------------------------------------------
        if masks_needed:
            if boxes.shape[0] == 0:
                img = imread(info["path"])
                H, W, D = img.shape[1], img.shape[2], img.shape[0]  # (Z,Y,X)→(Y,X,Z)
                masks = np.zeros((H, W, D, 0), dtype=bool)
            else:
                with bz2.BZ2File(info["m_path"], 'rb') as f:
                    m = cPickle.load(f)  # (Z, Y, X, N)
                masks = np.transpose(m, (1, 2, 0, 3))  # → (Y, X, Z, N)
        else:
            masks = None

        return boxes, class_ids, masks


class ToyHeadDataset(Dataset):

    def load_dataset(self, data_dir, is_train=True):
        '''
        An image is defined in the dataset by:
        # image_id, an image id [type: str of size 7]
        # path, the path of its origin 3D image [type: str]
        # seg_path, the path of the segmented 3D image [type: str]
        # ann_time_step, the time step of the origin 3D image (for validation test) [type: str of size 3]
        # ann_section_type, the type of section of the origin 3D image (x, y or z) [type: str of form 'x-section']
        # ann_section_id, the index of the section in regard to the origin image [type: str of size 3]
        # ann_mean = the mean of the origin 3D image pixel values [type: float]
        # ann_std = the standard deviation of the origin 3D image pixel values
        '''
        self.add_class("dataset", 1, "neuron")


        if is_train:
            td = pd.read_csv(f"{data_dir}datasets/train.csv", header=[0])
            for i in range(len(td)):
                r_path = td["rois"][i]
                ra_path = td["rois_aligned"][i]
                ma_path = td["mask_aligned"][i]
                tci_path = td["target_class_ids"][i]
                tb_path = td["target_bbox"][i]
                tm_path = td["target_mask"][i]
                self.add_image('dataset', image_id=i, path=r_path, ra_path=ra_path, ma_path=ma_path, tci_path=tci_path,
                               tb_path=tb_path, tm_path=tm_path)
            print('Training dataset is loaded.')
        else:
            td = pd.read_csv(f"{data_dir}datasets/test.csv", header=[0])
            for i in range(len(td)):
                r_path = td["rois"][i]
                ra_path = td["rois_aligned"][i]
                ma_path = td["mask_aligned"][i]
                tci_path = td["target_class_ids"][i]
                tb_path = td["target_bbox"][i]
                tm_path = td["target_mask"][i]
                self.add_image('dataset', image_id=i, path=r_path, ra_path=ra_path, ma_path=ma_path, tci_path=tci_path,
                               tb_path=tb_path, tm_path=tm_path)
            print('Validation dataset is loaded.')

    def load_data(self, image_id):
        info = self.image_info[image_id]
        rois_aligned = np.load(info["ra_path"])
        mask_aligned = np.load(info["ma_path"])
        target_class_ids = np.load(info["tci_path"])
        target_bbox = np.load(info["tb_path"])
        target_mask = np.load(info["tm_path"])
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