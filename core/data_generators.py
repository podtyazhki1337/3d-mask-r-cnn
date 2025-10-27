import math
import bz2
import _pickle as cPickle
import numpy as np
import pandas as pd
from skimage.io import imread
import keras
from core import utils
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

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
    Генератор для обучения HEAD с защитой от переполнения памяти.

    КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
    - Использует HEAD_TRAIN_ROIS_PER_IMAGE (32-64) вместо TRAIN_ROIS_PER_IMAGE (128)
    - Жёсткое ограничение на загрузку ROI ДО попадания в память
    - Фильтрация происходит в load_data(), а не в __getitem__
    """

    def __init__(self, dataset, config, shuffle=True, training=True, batch_size=1):
        self.dataset = dataset
        self.config = config
        self.shuffle = bool(shuffle)
        self.training = bool(training)
        self.batch_size = int(batch_size) if batch_size is not None else 1

        # КРИТИЧНО: используем отдельный параметр для HEAD (меньше чем для генерации таргетов)
        self.head_train_rois = int(getattr(config, "TRAIN_ROIS_PER_IMAGE", 64))

        self.image_ids = np.copy(self.dataset.image_ids).astype(np.int64)
        self._call_count = 0

        if self.shuffle:
            np.random.shuffle(self.image_ids)

        print(f"[HeadGenerator] Using HEAD_TRAIN_ROIS_PER_IMAGE={self.head_train_rois}")

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def load_image_gt(self, image_id):
        """
        Загружает данные с ЖЁСТКИМ ограничением на количество ROI.
        КРИТИЧНО: ограничение применяется В ДАТАСЕТЕ, до загрузки в память!
        """
        import numpy as np

        # ✅ КРИТИЧЕСКАЯ ЗАЩИТА: ограничиваем ПЕРЕД загрузкой
        MAX_LOAD_ROIS = 200  # Жёсткий лимит для защиты от багов

        # Если датасет поддерживает параметр max_rois, используем его
        if hasattr(self.dataset, 'load_data_with_limit'):
            rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = \
                self.dataset.load_data_with_limit(image_id, max_rois=MAX_LOAD_ROIS)
        else:
            # Старый метод - загружаем и сразу обрезаем
            rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = \
                self.dataset.load_data(image_id)

            # ✅ АВАРИЙНАЯ ЗАЩИТА: если загрузили слишком много - обрезаем СРАЗУ
            if rois_aligned.shape[0] > MAX_LOAD_ROIS:
                print(
                    f"[EMERGENCY] Image {image_id}: {rois_aligned.shape[0]} ROI loaded, truncating to {MAX_LOAD_ROIS}")
                rois_aligned = rois_aligned[:MAX_LOAD_ROIS]
                mask_aligned = mask_aligned[:MAX_LOAD_ROIS]
                target_class_ids = target_class_ids[:MAX_LOAD_ROIS]
                target_bbox = target_bbox[:MAX_LOAD_ROIS]
                target_mask = target_mask[:MAX_LOAD_ROIS]

        # Типы
        rois_aligned = rois_aligned.astype(np.float32, copy=False)
        mask_aligned = mask_aligned.astype(np.float32, copy=False)
        target_class_ids = target_class_ids.astype(np.int32, copy=False)
        target_bbox = target_bbox.astype(np.float32, copy=False)

        # Маски: [T,H,W,D] -> [T,H,W,D,1]
        if target_mask.ndim == 4:
            target_mask = target_mask[..., np.newaxis]
        target_mask = (target_mask > 0.5).astype(np.float32, copy=False)

        return rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask

    def __getitem__(self, idx):
        """
        Батч для головы с правильными размерностями всех тензоров.
        Включает фильтрацию слабых positives и диагностику качества ROI.
        """
        import numpy as np

        image_id = int(self.image_ids[idx])

        # 1) Загружаем данные (уже с правильными размерностями после исправления ToyHeadDataset)
        rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = self.dataset.load_data(image_id)

        # ========== КРИТИЧЕСКАЯ ДИАГНОСТИКА RAW DATA ==========
        if idx < 1:
            print(f"\n{'=' * 70}")
            print(f"[RAW DATA CHECK] Image {image_id}")
            print(f"{'=' * 70}")

            print(f"\n1. SHAPES:")
            print(f"   rois_aligned: {rois_aligned.shape}")
            print(f"   mask_aligned: {mask_aligned.shape}")
            print(f"   target_class_ids: {target_class_ids.shape}")
            print(f"   target_bbox: {target_bbox.shape}")
            print(f"   target_mask: {target_mask.shape}")

            print(f"\n2. DATA TYPES:")
            print(f"   rois_aligned: {rois_aligned.dtype}")
            print(f"   mask_aligned: {mask_aligned.dtype}")
            print(f"   target_class_ids: {target_class_ids.dtype}")
            print(f"   target_bbox: {target_bbox.dtype}")
            print(f"   target_mask: {target_mask.dtype}")

            print(f"\n3. ROIS (bbox coordinates):")
            pos_mask = target_class_ids > 0
            if np.any(pos_mask):
                pos_rois = rois[pos_mask]
                print(f"   Positive ROIs shape: {pos_rois.shape}")
                print(f"   First 3 positive ROIs:")
                for i, roi in enumerate(pos_rois[:3]):
                    print(f"     ROI {i}: y1={roi[0]:.4f}, x1={roi[1]:.4f}, z1={roi[2]:.4f}, "
                          f"y2={roi[3]:.4f}, x2={roi[4]:.4f}, z2={roi[5]:.4f}")
                    h = roi[3] - roi[0]
                    w = roi[4] - roi[1]
                    d = roi[5] - roi[2]
                    print(f"            h={h:.4f}, w={w:.4f}, d={d:.4f}, vol={h * w * d:.6f}")

                print(f"   Stats (all positive ROIs):")
                print(f"     y1: min={pos_rois[:, 0].min():.4f}, max={pos_rois[:, 0].max():.4f}")
                print(f"     x1: min={pos_rois[:, 1].min():.4f}, max={pos_rois[:, 1].max():.4f}")
                print(f"     z1: min={pos_rois[:, 2].min():.4f}, max={pos_rois[:, 2].max():.4f}")
                print(f"     y2: min={pos_rois[:, 3].min():.4f}, max={pos_rois[:, 3].max():.4f}")
                print(f"     x2: min={pos_rois[:, 4].min():.4f}, max={pos_rois[:, 4].max():.4f}")
                print(f"     z2: min={pos_rois[:, 5].min():.4f}, max={pos_rois[:, 5].max():.4f}")

            print(f"\n4. ROIS_ALIGNED (feature maps - NOT coordinates!):")
            print(f"   Shape: {rois_aligned.shape}")
            print(f"   Stats: min={rois_aligned.min():.4f}, max={rois_aligned.max():.4f}, "
                  f"mean={rois_aligned.mean():.4f}")

            print(f"\n5. TARGET_MASK (mask content):")
            if np.any(pos_mask):
                pos_indices = np.where(pos_mask)[0]
                print(f"   Checking first 3 positive masks:")
                for i in pos_indices[:3]:
                    mask = target_mask[i]
                    if mask.ndim == 4:
                        mask = mask[..., 0]
                    print(f"     Mask {i}: shape={mask.shape}, "
                          f"min={mask.min():.4f}, max={mask.max():.4f}, "
                          f"mean={mask.mean():.4f}, "
                          f"nonzero={(mask > 0.5).sum()}/{mask.size} "
                          f"({100 * (mask > 0.5).sum() / mask.size:.1f}%)")

            print(f"\n6. MASK_ALIGNED (feature maps):")
            if np.any(pos_mask):
                pos_masks_aligned = mask_aligned[pos_mask]
                print(f"   Positive masks_aligned shape: {pos_masks_aligned.shape}")
                print(f"   Stats (all positive):")
                print(f"     min={pos_masks_aligned.min():.4f}")
                print(f"     max={pos_masks_aligned.max():.4f}")
                print(f"     mean={pos_masks_aligned.mean():.4f}")
                print(f"     std={pos_masks_aligned.std():.4f}")

            print(f"\n7. TARGET_BBOX (deltas):")
            if np.any(pos_mask):
                pos_deltas = target_bbox[pos_mask]
                print(f"   Positive deltas shape: {pos_deltas.shape}")
                print(f"   First 3 positive deltas:")
                for i, delta in enumerate(pos_deltas[:3]):
                    print(f"     Delta {i}: [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}, "
                          f"{delta[3]:.4f}, {delta[4]:.4f}, {delta[5]:.4f}]")

                print(f"   Stats (all positive):")
                for coord in range(6):
                    vals = pos_deltas[:, coord]
                    print(f"     coord{coord}: min={vals.min():.4f}, "
                          f"max={vals.max():.4f}, mean={vals.mean():.4f}")

            print(f"{'=' * 70}\n")
        # ========== КОНЕЦ RAW DATA CHECK ==========

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

        # target_mask = (target_mask > 0.5).astype(np.float32, copy=False)

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

        # ========== ДИАГНОСТИКА КАЧЕСТВА ROI (до фильтрации) ==========
        if self.training and idx < 3:  # первые 3 батча
            pos_mask = target_class_ids > 0
            neg_mask = target_class_ids == 0

            print(f"\n{'=' * 70}")
            print(f"[BATCH DIAGNOSTICS] Batch {idx}, Image {image_id}")
            print(f"{'=' * 70}")
            print(f"Total ROIs loaded: {len(target_class_ids)}")
            print(f"  Positives (class>0): {np.sum(pos_mask)}")
            print(f"  Negatives (class=0): {np.sum(neg_mask)}")

            if np.any(pos_mask):
                # Coverage для positives
                pos_indices = np.where(pos_mask)[0]
                coverages = np.array([
                    float(target_mask[i].mean())
                    for i in pos_indices
                ])

                print(f"\nPositive ROI mask coverage:")
                print(f"  min:  {np.min(coverages):.4f}")
                print(f"  mean: {np.mean(coverages):.4f}")
                print(f"  p25:  {np.percentile(coverages, 25):.4f}")
                print(f"  p50:  {np.percentile(coverages, 50):.4f}")
                print(f"  p75:  {np.percentile(coverages, 75):.4f}")
                print(f"  max:  {np.max(coverages):.4f}")

                # Гистограмма coverage
                weak = np.sum(coverages < 0.1)
                medium = np.sum((coverages >= 0.1) & (coverages < 0.3))
                strong = np.sum(coverages >= 0.3)
                print(f"\nCoverage distribution:")
                print(f"  Weak (<0.1):      {weak:3d} ({100 * weak / len(coverages):5.1f}%)")
                print(f"  Medium (0.1-0.3): {medium:3d} ({100 * medium / len(coverages):5.1f}%)")
                print(f"  Strong (>0.3):    {strong:3d} ({100 * strong / len(coverages):5.1f}%)")

                # Проверка bbox размеров (нормализованных)
                pos_boxes = rois[pos_mask]
                if len(pos_boxes) > 0:
                    heights = pos_boxes[:, 3] - pos_boxes[:, 0]
                    widths = pos_boxes[:, 4] - pos_boxes[:, 1]
                    depths = pos_boxes[:, 5] - pos_boxes[:, 2]

                    print(f"\nPositive bbox sizes (normalized [0,1]):")
                    print(f"  height: {np.mean(heights):.3f} ± {np.std(heights):.3f} "
                          f"[{np.min(heights):.3f}, {np.max(heights):.3f}]")
                    print(f"  width:  {np.mean(widths):.3f} ± {np.std(widths):.3f} "
                          f"[{np.min(widths):.3f}, {np.max(widths):.3f}]")
                    print(f"  depth:  {np.mean(depths):.3f} ± {np.std(depths):.3f} "
                          f"[{np.min(depths):.3f}, {np.max(depths):.3f}]")

                    # Объём боксов
                    volumes = heights * widths * depths
                    print(f"  volume: {np.mean(volumes):.4f} ± {np.std(volumes):.4f} "
                          f"[{np.min(volumes):.4f}, {np.max(volumes):.4f}]")

            print(f"{'=' * 70}\n")
        # ========== КОНЕЦ ДИАГНОСТИКИ ==========

        # 6) Выборка ROI с фильтрацией и балансировкой
        total = int(rois_aligned.shape[0])
        order = np.arange(total, dtype=np.int32)

        if self.training and bool(getattr(cfg, "HEAD_SHUFFLE_ROIS", True)) and total > 0:
            np.random.shuffle(order)

        if total <= T:
            sel = order
        else:
            if self.training and bool(getattr(cfg, "HEAD_BALANCE_POS", True)):
                all_pos = np.where(target_class_ids > 0)[0]
                all_neg = np.where(target_class_ids <= 0)[0]

                # ========== ФИЛЬТРАЦИЯ СЛАБЫХ POSITIVES ==========
                min_coverage = float(getattr(cfg, "HEAD_MIN_POSITIVE_COVERAGE", 0.06))

                if len(all_pos) > 0:
                    # Вычисляем coverage для каждого positive ROI
                    pos_coverages = np.array([
                        float(target_mask[i].mean())
                        for i in all_pos
                    ])

                    # Фильтруем слабые
                    strong_mask = pos_coverages >= min_coverage
                    strong_pos = all_pos[strong_mask]
                    weak_pos = all_pos[~strong_mask]

                    # Диагностика
                    if len(weak_pos) > 0 and idx < 5:
                        weak_cov = pos_coverages[~strong_mask]
                        print(f"[ROI FILTER] Image {image_id}: "
                              f"Filtered out {len(weak_pos)}/{len(all_pos)} weak positives "
                              f"(coverage<{min_coverage:.2f})")
                        print(f"  Weak positives coverage: "
                              f"min={weak_cov.min():.4f}, max={weak_cov.max():.4f}, mean={weak_cov.mean():.4f}")

                        if len(strong_pos) > 0:
                            strong_cov = pos_coverages[strong_mask]
                            print(f"  Strong positives coverage: "
                                  f"min={strong_cov.min():.4f}, max={strong_cov.max():.4f}, mean={strong_cov.mean():.4f}")

                    # Берём сильные, если есть; иначе лучшие из слабых
                    if len(strong_pos) > 0:
                        all_pos = strong_pos
                    else:
                        # Все слабые → берём топ-N лучших
                        n_take = min(len(all_pos), max(1, T // 2))
                        if idx < 5:
                            print(f"[ROI FILTER WARNING] All {len(all_pos)} positives are weak! "
                                  f"Taking best {n_take} anyway")
                        best_indices = np.argsort(pos_coverages)[::-1][:n_take]
                        all_pos = all_pos[best_indices]
                else:
                    # Нет позитивов вообще
                    if idx < 5:
                        print(f"[WARNING] Image {image_id} has NO positives after initial filtering!")
                    all_pos = np.array([], dtype=np.int32)
                # ========== КОНЕЦ ФИЛЬТРАЦИИ ==========

                # Пересечение с order (shuffled)
                pos_idx = np.intersect1d(order, all_pos, assume_unique=False) if len(all_pos) > 0 else np.array([],
                                                                                                                dtype=np.int32)
                neg_idx = np.intersect1d(order, all_neg, assume_unique=False)

                pos_frac = float(getattr(cfg, "HEAD_POS_FRAC", 0.33))
                pos_cnt = int(round(T * pos_frac))
                neg_cnt = T - pos_cnt

                # Если позитивы есть, но не попали в shuffled order → берём напрямую
                if pos_idx.size == 0 and len(all_pos) > 0:
                    pos_idx = all_pos[:min(len(all_pos), pos_cnt)]
                    if idx < 5:
                        print(f"[WARNING] Positives not in shuffled order, taking {len(pos_idx)} directly")

                # ========== ВЫБОРКА ROI С БАЛАНСИРОВКОЙ ==========
                if pos_idx.size == 0 and neg_idx.size == 0:
                    # Совсем нет данных → fallback на случайную выборку
                    if idx < 5:
                        print(f"[ERROR] Image {image_id}: No pos AND no neg indices! Using random fallback")
                    sel = np.random.choice(order, size=min(T, total), replace=False)

                elif pos_idx.size == 0:
                    # Нет позитивов → только негативы
                    if idx < 5:
                        print(f"[WARNING] Image {image_id}: NO positives available! Using {T} negatives")
                    sel = np.random.choice(neg_idx, size=T, replace=neg_idx.size < T)

                elif neg_idx.size == 0:
                    # Нет негативов → только позитивы
                    if idx < 5:
                        print(f"[WARNING] Image {image_id}: NO negatives available! Using {T} positives")
                    sel = np.random.choice(pos_idx, size=T, replace=pos_idx.size < T)

                else:
                    # ✅ НОРМАЛЬНЫЙ СЛУЧАЙ: есть и позитивы, и негативы
                    # Сэмплируем с учётом доступного количества
                    actual_pos_cnt = min(pos_cnt, pos_idx.size)
                    actual_neg_cnt = T - actual_pos_cnt

                    # Если негативов не хватает, добираем позитивами
                    if actual_neg_cnt > neg_idx.size:
                        actual_neg_cnt = neg_idx.size
                        actual_pos_cnt = T - actual_neg_cnt

                    sel_pos = np.random.choice(pos_idx, size=actual_pos_cnt, replace=pos_idx.size < actual_pos_cnt)
                    sel_neg = np.random.choice(neg_idx, size=actual_neg_cnt, replace=neg_idx.size < actual_neg_cnt)

                    sel = np.concatenate([sel_pos, sel_neg])
                    np.random.shuffle(sel)

                    # Финальная статистика
                    if idx < 5:
                        final_pos = np.sum(target_class_ids[sel] > 0)
                        final_neg = np.sum(target_class_ids[sel] == 0)
                        print(f"[ROI SELECTION] Image {image_id}: "
                              f"Selected {final_pos} pos + {final_neg} neg = {len(sel)} total "
                              f"(target: {pos_cnt} pos + {neg_cnt} neg)")

                        # Проверка coverage финального батча
                        if final_pos > 0:
                            final_coverages = np.array([
                                float(target_mask[i].mean())
                                for i in sel if target_class_ids[i] > 0
                            ])
                            print(f"  Final positive coverage: "
                                  f"min={final_coverages.min():.4f}, "
                                  f"mean={final_coverages.mean():.4f}, "
                                  f"max={final_coverages.max():.4f}")
                # ========== КОНЕЦ ВЫБОРКИ ==========
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

        # 7) Финальная диагностика после выборки
        if self.training and idx < 5:
            final_pos = np.sum(target_class_ids > 0)
            final_neg = np.sum(target_class_ids == 0)

            if final_pos > 0:
                final_coverages = np.array([
                    float(target_mask[i].mean())
                    for i in range(len(target_class_ids)) if target_class_ids[i] > 0
                ])

                print(f"[FINAL BATCH] Image {image_id} - After selection & padding:")
                print(f"  Batch size: {T}")
                print(f"  Positives: {final_pos} ({100 * final_pos / T:.1f}%)")
                print(f"  Negatives: {final_neg} ({100 * final_neg / T:.1f}%)")
                print(f"  Positive coverage: min={final_coverages.min():.4f}, "
                      f"mean={final_coverages.mean():.4f}, max={final_coverages.max():.4f}")
                print()

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
        self.anchors = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE
        )

        H, W, D = int(self.config.IMAGE_SHAPE[0]), int(self.config.IMAGE_SHAPE[1]), int(self.config.IMAGE_SHAPE[2])

        # КРИТИЧНО: НЕ ТРОГАЕМ Z-координаты! Они уже правильно сгенерированы с ratios
        a = self.anchors.copy()

        # Только клипируем в границы
        a[:, 0] = np.clip(a[:, 0], 0, H - 1)
        a[:, 1] = np.clip(a[:, 1], 0, W - 1)
        a[:, 2] = np.clip(a[:, 2], 0, D - 1)
        a[:, 3] = np.clip(a[:, 3], 1, H)
        a[:, 4] = np.clip(a[:, 4], 1, W)
        a[:, 5] = np.clip(a[:, 5], 1, D)

        # Гарантируем минимальный размер
        a[:, 3] = np.maximum(a[:, 3], a[:, 0] + 1)
        a[:, 4] = np.maximum(a[:, 4], a[:, 1] + 1)
        a[:, 5] = np.maximum(a[:, 5], a[:, 2] + 0.5)  # Минимальная толщина 0.5 для плоских

        # Простая нормализация
        scale = np.array([H, W, D, H, W, D], dtype=np.float32)
        self.anchors = (a / scale).astype(np.float32)
        self.anchor_nb = self.anchors.shape[0]

        # Детальная отладка
        anchors_px = a
        sizes_y = anchors_px[:, 3] - anchors_px[:, 0]
        sizes_x = anchors_px[:, 4] - anchors_px[:, 1]
        sizes_z = anchors_px[:, 5] - anchors_px[:, 2]
        xy_sizes = np.sqrt(sizes_y * sizes_x)
        z_ratios = sizes_z / (xy_sizes + 1e-8)

        print(f"\n[ANCHOR STATS]")
        print(f"  Total anchors: {len(self.anchors)}")
        print(f"  XY sizes: {np.percentile(xy_sizes, [10, 25, 50, 75, 90]).round(1)}")
        print(f"  Z sizes: {np.percentile(sizes_z, [10, 25, 50, 75, 90]).round(2)}")
        print(f"  Z/XY ratios: {np.percentile(z_ratios, [10, 25, 50, 75, 90]).round(3)}")

        # Проверка покрытия целевых ratios
        target_ratios = config.RPN_ANCHOR_RATIOS
        print(f"  Target ratios: {target_ratios}")
        print(f"  Actual ratio range: [{z_ratios.min():.3f}, {z_ratios.max():.3f}]")

        self.config.ANCHOR_NB = self.anchor_nb


    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(ids) == 0:
            raise IndexError("Empty batch requested — adjust evaluation steps or dataset size")
        return self.data_generator(ids)

    def rebuild_anchors(self):
        """
        Пересобирает anchors с простой нормализацией (деление на H,W,D без смещения).
        """
        H, W, D = int(self.config.IMAGE_SHAPE[0]), int(self.config.IMAGE_SHAPE[1]), int(self.config.IMAGE_SHAPE[2])

        # 1) Пирамидальные якоря в пикселях
        backbone_shapes = compute_backbone_shapes(self.config, self.config.IMAGE_SHAPE)
        a = utils.generate_pyramid_anchors(
            self.config.RPN_ANCHOR_SCALES,
            self.config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            self.config.BACKBONE_STRIDES,
            self.config.RPN_ANCHOR_STRIDE
        ).astype(np.float32)  # [N, (y1, x1, z1, y2, x2, z2)] в пикселях

        # 2) Толщина по Z
        VOXEL_Z_OVER_Y = float(getattr(self.config, "VOXEL_Z_OVER_Y", 4.0))
        MIN_Z_EXT = int(getattr(self.config, "RPN_MIN_Z_EXTENT", 2))
        MAX_Z_EXT = int(getattr(self.config, "RPN_MAX_Z_EXTENT", max(2, D)))

        h_pix = (a[:, 3] - a[:, 0]).astype(np.float32)
        z_extent = np.clip(np.round(h_pix / max(VOXEL_Z_OVER_Y, 1.0)), MIN_Z_EXT, MAX_Z_EXT).astype(np.float32)

        cz = (a[:, 2] + a[:, 5]) * 0.5
        hz = 0.5 * z_extent

        z1 = cz - hz
        z2 = cz + hz

        # 3) Клип внутри объёма
        z1 = np.clip(z1, 0.0, D - 1.0)
        z2 = np.clip(z2, 0.0, D - 1.0)
        same = (z2 <= z1)
        if np.any(same):
            z2[same] = np.minimum(D - 1.0, z1[same] + 1.0)

        a[:, 2] = z1
        a[:, 5] = z2
        a[:, 0] = np.clip(a[:, 0], 0.0, H - 1.0)
        a[:, 1] = np.clip(a[:, 1], 0.0, W - 1.0)
        a[:, 2] = np.clip(a[:, 2], 0.0, D - 1.0)
        a[:, 3] = np.clip(a[:, 3], 0.0, H)
        a[:, 4] = np.clip(a[:, 4], 0.0, W)
        a[:, 5] = np.clip(a[:, 5], 0.0, D)

        # 4) Простая нормализация: деление на H,W,D (БЕЗ H-1)
        self.anchors_pix = a.astype(np.float32)
        scale = np.array([H, W, D, H, W, D], dtype=np.float32)
        self.anchors = (self.anchors_pix / scale).astype(np.float32)
        self.anchor_nb = int(self.anchors_pix.shape[0])
        self.config.ANCHOR_NB = self.anchor_nb

    # В rpn_model.py, метод data_generator класса RPNGenerator:

    def data_generator(self, image_ids):
        """
        Генерирует батчи для RPN-тренировки или targeting.
        - MODE="training": [images, rpn_match, rpn_bbox] (3 входа)
        - MODE="targeting" или "training_head_e2e": [images, meta, gt_class_ids, gt_boxes, gt_masks] (5 входов)
        """
        import numpy as np

        if len(image_ids) == 0:
            raise IndexError("Empty image_ids passed to data_generator.")

        # ✅ КРИТИЧНО: проверяем режим работы
        mode = getattr(self.config, "MODE", "training")

        # режим targeting — один пример «на показ»
        if mode == "targeting":
            image_id = image_ids[0]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.load_image_gt(image_id)

            # batch-оси
            batch_images = image[np.newaxis, ...].astype(np.float32)
            batch_image_meta = image_meta[np.newaxis, ...]

            # GT как ожидает модель в targeting-режиме
            gt_class_ids = np.asarray(gt_class_ids, dtype=np.int32)
            gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
            gt_masks = gt_masks.astype(bool)

            batch_gt_class_ids = gt_class_ids[np.newaxis, ...]
            batch_gt_boxes = gt_boxes[np.newaxis, ...] if gt_boxes.ndim == 2 else gt_boxes[np.newaxis, :, :]
            if gt_masks.ndim == 4:
                batch_gt_masks = gt_masks[np.newaxis, ...]
            else:
                batch_gt_masks = gt_masks[np.newaxis, ..., np.newaxis]

            inputs = [batch_images, batch_image_meta, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            outputs = []
            return inputs, outputs

        # ✅ ДОБАВЛЕНО: режим training_head_e2e — такой же как targeting!
        if mode == "training_head_e2e":
            batch_size = int(self.batch_size)
            batch_images = np.zeros((batch_size,) + tuple(self.config.IMAGE_SHAPE), dtype=np.float32)

            batch_image_meta = []
            batch_gt_class_ids = []
            batch_gt_boxes = []
            batch_gt_masks = []

            for bi in range(batch_size):
                image_id = image_ids[min(bi, len(image_ids) - 1)]

                # ✅ ИСПРАВЛЕНО: load_image_gt теперь возвращает все нужные данные
                image, image_meta, class_ids, boxes, masks = self.load_image_gt(image_id)

                # Защита от пустых GT
                tries = 0
                while (boxes is None or np.size(boxes) == 0) and tries < 5:
                    new_id = int(np.random.choice(self.image_ids))
                    image, image_meta, class_ids, boxes, masks = self.load_image_gt(new_id)
                    tries += 1

                batch_images[bi] = image.astype(np.float32)
                batch_image_meta.append(image_meta)

                # ✅ GT данные (уже нормализованные!)
                batch_gt_class_ids.append(class_ids if class_ids is not None else np.array([], dtype=np.int32))
                batch_gt_boxes.append(boxes if boxes is not None else np.zeros((0, 6), dtype=np.float32))

                # ✅ Маски уже загружены!
                if masks is None or masks.size == 0:
                    H, W, D = self.config.IMAGE_SHAPE[:3]
                    masks = np.zeros((H, W, D, 0), dtype=bool)
                batch_gt_masks.append(masks)

            # Паддинг GT до одинаковой длины
            max_gt = max(len(gt) for gt in batch_gt_class_ids)
            if max_gt == 0:
                max_gt = 1

            # Паддим class_ids
            padded_class_ids = np.zeros((batch_size, max_gt), dtype=np.int32)
            for i, gt in enumerate(batch_gt_class_ids):
                if len(gt) > 0:
                    padded_class_ids[i, :len(gt)] = gt[:max_gt]

            # Паддим boxes
            padded_boxes = np.zeros((batch_size, max_gt, 6), dtype=np.float32)
            for i, gt in enumerate(batch_gt_boxes):
                if len(gt) > 0:
                    n = min(len(gt), max_gt)
                    padded_boxes[i, :n] = gt[:n]

            # ✅ ДИАГНОСТИКА: проверяем boxes перед паддингом масок
            if bi == 0 and not hasattr(self, '_e2e_boxes_checked'):
                self._e2e_boxes_checked = True
                print(f"\n[E2E_GEN_DEBUG] First batch boxes:")
                for i in range(min(2, batch_size)):
                    if len(batch_gt_boxes[i]) > 0:
                        print(f"  Image {i}: boxes shape={batch_gt_boxes[i].shape}, "
                              f"min={batch_gt_boxes[i].min():.4f}, max={batch_gt_boxes[i].max():.4f}")
                        print(f"    First box: {batch_gt_boxes[i][0]}")

            # Паддим masks
            H, W, D = self.config.IMAGE_SHAPE[:3]
            padded_masks = np.zeros((batch_size, H, W, D, max_gt), dtype=bool)
            for i, gt in enumerate(batch_gt_masks):
                if gt.shape[-1] > 0:
                    n = min(gt.shape[-1], max_gt)
                    padded_masks[i, ..., :n] = gt[..., :n]

            batch_image_meta = np.array(batch_image_meta, dtype=np.float32)

            inputs = [batch_images, batch_image_meta, padded_class_ids, padded_boxes, padded_masks]
            outputs = []
            return inputs, outputs

        # обычная тренировка RPN
        batch_size = int(self.batch_size)
        batch_images = np.zeros((batch_size,) + tuple(self.config.IMAGE_SHAPE), dtype=np.float32)
        batch_rpn_match = np.zeros((batch_size, self.anchor_nb, 1), dtype=np.int8)
        batch_rpn_bbox = np.zeros((batch_size, int(getattr(self.config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 256)), 6),
                                  dtype=np.float32)

        batch_gt_stats = []

        for bi in range(batch_size):
            image_id = image_ids[min(bi, len(image_ids) - 1)]
            image, boxes, class_ids = self.load_image_gt(image_id)

            # защищаемся от пустых GT
            tries = 0
            while (boxes is None or np.size(boxes) == 0) and tries < 5:
                new_id = int(np.random.choice(self.image_ids))
                image, boxes, class_ids = self.load_image_gt(new_id)
                tries += 1

            if boxes is not None and len(boxes) > 0:
                h, w, d = self.config.IMAGE_SHAPE[:3]
                if boxes.max() > 1.0:
                    boxes_px = boxes.copy()
                else:
                    boxes_px = boxes.copy()
                    boxes_px[:, [0, 3]] *= h
                    boxes_px[:, [1, 4]] *= w
                    boxes_px[:, [2, 5]] *= d

                gt_sizes_y = boxes_px[:, 3] - boxes_px[:, 0]
                gt_sizes_x = boxes_px[:, 4] - boxes_px[:, 1]
                gt_sizes_z = boxes_px[:, 5] - boxes_px[:, 2]
                gt_sizes_xy = (gt_sizes_y + gt_sizes_x) / 2.0

                batch_gt_stats.append({
                    'xy': gt_sizes_xy,
                    'z': gt_sizes_z
                })

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

        # Статистика GT
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1

        if self._batch_counter % 300 == 1 and len(batch_gt_stats) > 0:
            all_xy = np.concatenate([s['xy'] for s in batch_gt_stats])
            all_z = np.concatenate([s['z'] for s in batch_gt_stats])
            print(f"\n[GT DEBUG] Batch {self._batch_counter}, {len(batch_gt_stats)} images, {len(all_xy)} objects")
            print(
                f"  XY sizes (px): min={all_xy.min():.1f}, mean={all_xy.mean():.1f}, median={np.median(all_xy):.1f}, max={all_xy.max():.1f}")
            print(
                f"  Z sizes (px):  min={all_z.min():.1f}, mean={all_z.mean():.1f}, median={np.median(all_z):.1f}, max={all_z.max():.1f}\n")

        batch_anchors = np.tile(self.anchors[np.newaxis, :, :], (batch_size, 1, 1))

        # Keras-совместимый вывод
        inputs = [batch_images, batch_rpn_match, batch_rpn_bbox]
        outputs = []
        return inputs, outputs

    def load_image_gt(self, image_id):
        """
        Загрузка изображения и GT данных.

        Returns:
            - MODE="training": (image, boxes, class_ids)
            - MODE="targeting" или "training_head_e2e": (image, image_meta, class_ids, boxes, masks)
        """
        # Load image and mask
        image = self.dataset.load_image(image_id)
        boxes, class_ids, masks = self.dataset.load_data(image_id)
        # if boxes is not None and len(boxes) > 0:
        #     print(f"\n[RPN_GEN_DEBUG] Image {image_id}:")
        #     print(f"  boxes FROM dataset.load_data():")
        #     print(f"    shape={boxes.shape}")
        #     print(f"    min={boxes.min():.4f}, max={boxes.max():.4f}")
        #     print(f"    first box: {boxes[0]}")
        #
        #     # Проверка: если max > 1, то в пикселях
        #     if boxes.max() > 1.0:
        #         print(f"    ✅ IN PIXELS → will normalize")
        #     else:
        #         print(f"    ❌ ALREADY NORMALIZED → skip normalization!")
        # ✅ ПРОВЕРКА: применяем аугментации только для обычной RPN тренировки
        mode = getattr(self.config, "MODE", "training")
        use_augs = bool(getattr(self.config, "AUGMENT", True)) and mode == "training"

        if use_augs:
            image, boxes, masks = apply_minimal_augs_3d(image, boxes, masks, self.config)

        # ✅ ДЛЯ TARGETING и E2E возвращаем полный набор с image_meta
        if mode in ["targeting", "training_head_e2e"]:
            active_class_ids = np.ones([self.dataset.num_classes], dtype=np.int32)

            # ✅ КРИТИЧНО: проверяем, нормализованы ли boxes
            if boxes is not None and len(boxes) > 0:
                # Если boxes > 1, значит они в пикселях → нормализуем
                if boxes.max() > 1.0:
                    H, W, D = self.config.IMAGE_SHAPE[:3]
                    boxes = boxes.astype(np.float32)
                    boxes[:, [0, 3]] /= H  # y1, y2
                    boxes[:, [1, 4]] /= W  # x1, x2
                    boxes[:, [2, 5]] /= D  # z1, z2
                    boxes = np.clip(boxes, 0.0, 1.0)

            image_meta = compose_image_meta(
                image_id=image_id,
                original_image_shape=tuple(self.config.IMAGE_SHAPE),
                image_shape=tuple(self.config.IMAGE_SHAPE),
                window=(0, 0, 0, *self.config.IMAGE_SHAPE[:-1]),  # (0,0,0, H,W,D)
                scale=1.0,
                active_class_ids=active_class_ids
            )

            # ✅ ДИАГНОСТИКА (только первые 3 батча)
            if not hasattr(self, '_rpn_diag_count'):
                self._rpn_diag_count = 0

            if self._rpn_diag_count < 3:
                print(f"[RPN_GEN_{mode.upper()}] image_id={image_id}")
                print(f"  boxes: shape={boxes.shape if boxes is not None else None}, "
                      f"min={boxes.min():.4f}, max={boxes.max():.4f}" if boxes is not None and len(
                    boxes) > 0 else "  boxes: None")
                print(f"  class_ids: {class_ids}")
                print(f"  masks: shape={masks.shape}, dtype={masks.dtype}, sum={masks.sum():.0f}")
                self._rpn_diag_count += 1

            return image, image_meta, class_ids, boxes, masks

        # ✅ ДЛЯ ОБЫЧНОЙ RPN ТРЕНИРОВКИ возвращаем (image, boxes, class_ids)
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

        # 1) Якоря в пикселях
        self.anchors_pix = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            self.backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE
        )

        # 2) Привести толщину по Z
        H, W, D = int(self.config.IMAGE_SHAPE[0]), int(self.config.IMAGE_SHAPE[1]), int(self.config.IMAGE_SHAPE[2])
        VOXEL_Z_OVER_Y = float(getattr(self.config, "VOXEL_Z_OVER_Y", 4.0))
        MIN_Z_EXT = int(getattr(self.config, "RPN_MIN_Z_EXTENT", 2))
        MAX_Z_EXT = int(getattr(self.config, "RPN_MAX_Z_EXTENT", max(2, D)))

        a = self.anchors_pix
        h_pix = (a[:, 3] - a[:, 0]).astype(np.float32)
        z_extent = np.clip(np.round(h_pix / max(VOXEL_Z_OVER_Y, 1.0)), MIN_Z_EXT, MAX_Z_EXT).astype(np.float32)
        cz = (a[:, 2] + a[:, 5]) * 0.5
        hz = z_extent * 0.5
        z1 = np.clip(cz - hz, 0.0, D - 1.0)
        z2 = np.clip(cz + hz, 0.0, D - 1.0)
        same = (z2 <= z1)
        z2[same] = np.clip(z1[same] + 1.0, 0.0, D - 1.0)
        a[:, 2] = z1
        a[:, 5] = z2
        a[:, 0] = np.clip(a[:, 0], 0.0, H - 1.0)
        a[:, 1] = np.clip(a[:, 1], 0.0, W - 1.0)
        a[:, 2] = np.clip(a[:, 2], 0.0, D - 1.0)
        a[:, 3] = np.clip(a[:, 3], 0.0, H)
        a[:, 4] = np.clip(a[:, 4], 0.0, W)
        a[:, 5] = np.clip(a[:, 5], 0.0, D)
        self.anchors_pix = a

        # 3) Простая нормализация: деление на H,W,D (БЕЗ H-1)
        scale = np.array([H, W, D, H, W, D], dtype=np.float32)
        self.anchors = (self.anchors_pix / scale).astype(np.float32)

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


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def get_input_prediction(self, image_id):
        """
        Готовит входы для инференса ТОЧНО как в тренировке
        """
        import numpy as np

        info = self.dataset.image_info[image_id]
        image = self.dataset.load_image(image_id).astype(np.float32)

        # Применяем ТУ ЖЕ нормализацию, что была при обучении (из ToyDataset.load_image)
        # Z-score нормализация + процентильное клиппирование + tanh
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)

        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val > 0:
            image = (image - mean_val) / std_val
        else:
            image = image - mean_val

        # Дополнительное масштабирование для стабильности (как в ToyDataset)
        image = np.tanh(image * 0.5)

        name = info["path"].split("/")[-1].rsplit(".", 1)[0]

        # active_class_ids
        num = int(getattr(self.config, "NUM_CLASSES", getattr(self.dataset, "num_classes", 2)))
        active_class_ids = np.zeros([num], dtype=np.int32)

        try:
            src_ids = list(self.dataset.source_class_ids.get(info["source"], []))
            src_ids = [int(i) for i in src_ids if 0 <= int(i) < num]
            if src_ids:
                active_class_ids[src_ids] = 1
            else:
                # Если нет source_class_ids, активируем все кроме фона
                if num > 1:
                    active_class_ids[1:] = 1
        except Exception:
            # Фолбэк: активируем все кроме фона
            if num > 1:
                active_class_ids[1:] = 1

        # ВАЖНО: фон всегда активен
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

        batch_images = np.zeros((1,) + image.shape, dtype=np.float32)
        batch_images[0] = image
        batch_meta = np.zeros((1,) + image_meta.shape, dtype=np.float32)
        batch_meta[0] = image_meta
        batch_anchors = self.anchors[np.newaxis].astype(np.float32)

        return name, [batch_images, batch_meta, batch_anchors]

    def load_image_gt(self, image_id):
        """Load image and (если training) GT, с корректным active_class_ids под NUM_CLASSES."""
        import numpy as np

        # Load image
        image = self.dataset.load_image(image_id)

        # Собираем meta и active_class_ids одинаково для train/infer
        num = int(getattr(self.config, "NUM_CLASSES", getattr(self.dataset, "num_classes", 2)))
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

        if active_class_ids.shape[0] > 0:
            active_class_ids[0] = 1

        image_meta = compose_image_meta(
            image_id, tuple(self.config.IMAGE_SHAPE), tuple(self.config.IMAGE_SHAPE),
            (0, 0, 0, *self.config.IMAGE_SHAPE[:-1]), 1.0, active_class_ids
        )

        if self.training:
            # ✅ ИСПРАВЛЕНО: распаковываем все 6 значений!
            rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask = self.dataset.load_data(
                image_id)

            # ✅ ДИАГНОСТИКА
            if hasattr(self, '_diag_count'):
                self._diag_count += 1
            else:
                self._diag_count = 0

            if self._diag_count < 5:
                print(f"[MRCNN_GEN] image_id={image_id}")
                print(f"  rois.shape: {rois.shape}, dtype: {rois.dtype}")
                print(f"  rois_aligned.shape: {rois_aligned.shape}")
                print(f"  mask_aligned.shape: {mask_aligned.shape if mask_aligned is not None else None}")
                print(f"  target_class_ids.shape: {target_class_ids.shape}")
                print(f"  target_bbox.shape: {target_bbox.shape if target_bbox is not None else None}")
                print(f"  target_mask.shape: {target_mask.shape}, sum: {target_mask.sum():.0f}")

            # ✅ ВАЖНО: возвращаем rois (bbox), class_ids, masks для совместимости с остальным кодом
            # rois уже в правильном формате [T, 6]
            return image, image_meta, rois, target_class_ids, target_mask
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
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    assert config.BACKBONE in ["resnet50", "resnet101"]

    shapes = []
    for stride in config.BACKBONE_STRIDES:
        if isinstance(stride, (int, np.integer)):
            sy = sx = sz = int(stride)
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 3:
                sy, sx, sz = int(stride[0]), int(stride[1]), int(stride[2])
            elif len(stride) == 2:
                sy = sx = int(stride[0])
                sz = int(stride[1])
            else:
                sy = sx = sz = int(stride[0])
        else:
            sy = sx = sz = int(stride)

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
        Улучшенная загрузка данных с валидацией боксов и ПРАВИЛЬНОЙ обработкой масок.
        Returns:
            boxes: [N, 6] int32 (y1, x1, z1, y2, x2, z2) в пикселях
            class_ids: [N] int32
            masks: [H, W, D, N] float32 (0..1) — НЕ bool!
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

        # ✅ КРИТИЧНО: Загрузка масок с правильной обработкой формата
        if boxes.shape[0] == 0:
            img = imread(info["path"])
            H, W, D = img.shape[1], img.shape[2], img.shape[0]
            masks = np.zeros((H, W, D, 0), dtype=np.float32)  # ← float32, НЕ bool!
        else:
            try:
                with bz2.BZ2File(info["m_path"], 'rb') as f:
                    m = cPickle.load(f)  # (Z, Y, X, N)

                # ✅ ИСПРАВЛЕНИЕ: правильный transpose и dtype
                masks = np.transpose(m, (1, 2, 0, 3))  # -> (Y, X, Z, N)

                # ✅ КРИТИЧНО: приводим к float32 (не bool!)
                if masks.dtype == np.bool_ or masks.dtype == bool:
                    masks = masks.astype(np.uint8)  # bool -> uint8
                masks = masks.astype(np.float32, copy=False)  # -> float32

                # ✅ ДИАГНОСТИКА
                # print(f"[Dataset][{image_id}] Loaded masks: shape={masks.shape}, "
                #       f"dtype={masks.dtype}, sum={masks.sum():.0f}, "
                #       f"per_instance_sums={[masks[..., i].sum() for i in range(min(3, masks.shape[-1]))]}")

                # Обеспечиваем соответствие количества масок и боксов
                if masks.shape[-1] != boxes.shape[0]:
                    min_count = min(masks.shape[-1], boxes.shape[0])
                    print(f"[Dataset][{image_id}] Warning: masks count ({masks.shape[-1]}) != "
                          f"boxes count ({boxes.shape[0]}), truncating to {min_count}")
                    if min_count > 0:
                        masks = masks[..., :min_count]
                        boxes = boxes[:min_count]
                        class_ids = class_ids[:min_count]
                    else:
                        H, W, D = masks.shape[:3]
                        masks = np.zeros((H, W, D, 0), dtype=np.float32)
                        boxes = np.zeros((0, 6), dtype=np.int32)
                        class_ids = np.zeros((0,), dtype=np.int32)

            except Exception as e:
                print(f"[Dataset][{image_id}] Mask loading failed: {e}")
                img = imread(info["path"])
                H, W, D = img.shape[1], img.shape[2], img.shape[0]
                masks = np.zeros((H, W, D, 0), dtype=np.float32)

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

    def filter_by_positive_count(self, min_positive=20):
        """
        Удаляет изображения с малым количеством positive примеров.
        Вызывай ПОСЛЕ prepare().
        """
        import numpy as np

        filtered_ids = []
        skipped = []

        for image_id in self.image_ids:
            try:
                # Загружаем target_class_ids
                tci_path = self.image_info[image_id]["target_class_ids"]

                # Поддержка .npz
                if tci_path.endswith('.npz'):
                    with np.load(tci_path) as data:
                        tci = data['tci']
                else:
                    tci = np.load(tci_path, allow_pickle=False)

                pos_count = int(np.sum(tci > 0))

                if pos_count >= min_positive:
                    filtered_ids.append(image_id)
                else:
                    img_name = self.image_info[image_id].get("id", image_id)
                    skipped.append((img_name, pos_count))

            except Exception as e:
                print(f"[FILTER] Error loading image {image_id}: {e}")
                continue

        print(f"\n[FILTER] Keeping {len(filtered_ids)}/{len(self.image_ids)} images")
        print(f"[FILTER] Min positive threshold: {min_positive}")

        if skipped:
            print(f"[FILTER] Skipped {len(skipped)} images:")
            for name, cnt in skipped[:10]:  # показываем первые 10
                print(f"  - {name}: {cnt} positive")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")

        self.image_ids = np.array(filtered_ids, dtype=np.int64)
        print(f"[FILTER] Final dataset size: {len(self.image_ids)}\n")

    def load_dataset(self, data_dir, is_train=True):
        """
        Загружает HEAD датасет из CSV с колонками:
        rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask
        """
        import os
        import pandas as pd

        self.add_class("dataset", 1, "neuron")

        split = "train" if is_train else "test"
        csv_path = os.path.join(data_dir, "datasets", f"{split}.csv")
        td = pd.read_csv(csv_path, sep=None, engine="python")

        # Создаём lowercase-маппинг
        cols = {c.lower(): c for c in td.columns}

        def pick(*cands, required=True):
            """Ищет первую подходящую колонку"""
            for c in cands:
                k = c.lower()
                if k in cols:
                    return cols[k]
                # Частичное совпадение
                for lc, orig in cols.items():
                    if k in lc:
                        return orig
            if required:
                raise KeyError(
                    f"[ToyHeadDataset.load_dataset] None of columns {cands} found. "
                    f"Available: {list(td.columns)}"
                )
            return None

        # ✅ ИСПРАВЛЕНО: ищем все колонки включая rois
        col_rois = pick("rois", "rois_path", "r_path", required=True)  # ← ТЕПЕРЬ ОБЯЗАТЕЛЬНО
        col_ra = pick("rois_aligned", "ra_path", "aligned_rois", "roisAligned", required=True)
        col_ma = pick("mask_aligned", "ma_path", "aligned_mask", required=False)
        col_tci = pick("target_class_ids", "tci_path", "tci", required=True)
        col_tb = pick("target_bbox", "tb_path", "bbox", required=False)
        col_tm = pick("target_mask", "tm_path", "tm", required=False)

        print(f"[ToyHeadDataset] Using columns:")
        print(f"  rois: '{col_rois}'")
        print(f"  rois_aligned: '{col_ra}'")
        print(f"  mask_aligned: '{col_ma}'")
        print(f"  target_class_ids: '{col_tci}'")
        print(f"  target_bbox: '{col_tb}'")
        print(f"  target_mask: '{col_tm}'")

        # Загружаем каждую строку
        for i in range(len(td)):
            # ✅ ИСПРАВЛЕНО: читаем rois
            r_path = td.at[i, col_rois]
            ra_path = td.at[i, col_ra]
            ma_path = td.at[i, col_ma] if col_ma is not None else None
            tci_path = td.at[i, col_tci]
            tb_path = td.at[i, col_tb] if col_tb is not None else None
            tm_path = td.at[i, col_tm] if col_tm is not None else None

            # Валидация обязательных путей
            if not isinstance(r_path, str) or not r_path:
                raise ValueError(f"[ToyHeadDataset] Bad 'rois' at row {i}: {r_path}")
            if not isinstance(ra_path, str) or not ra_path:
                raise ValueError(f"[ToyHeadDataset] Bad 'rois_aligned' at row {i}: {ra_path}")
            if not isinstance(tci_path, str) or not tci_path:
                raise ValueError(f"[ToyHeadDataset] Bad 'target_class_ids' at row {i}: {tci_path}")

            # Извлекаем имя для path (используем basename от rois)
            path = os.path.basename(r_path)

            # ✅ ИСПРАВЛЕНО: добавляем rois в add_image
            self.add_image(
                source='dataset',
                image_id=i,
                path=path,
                rois=r_path,  # ← КРИТИЧНО: ДОБАВЛЕНО!
                ra_path=ra_path,
                ma_path=ma_path,
                tci_path=tci_path,
                tb_path=tb_path,
                tm_path=tm_path
            )

        print(f"[ToyHeadDataset] {'Training' if is_train else 'Validation'} dataset loaded: "
              f"{len(self.image_info)} images")

    def load_image(self, image_id, z_slice=None):
        """Нужен только для совместимости проверок. Возвращаем dummy [H,W,D,1] из конфига."""
        H, W, D = tuple(getattr(self, "config", None).IMAGE_SHAPE[:3])
        return np.zeros((H, W, D, 1), dtype=np.float32)

    def load_data(self, image_id):
        """
        Возвращает РОВНО 6 массивов для головы:
        rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask

        Returns:
            rois: [T, 6] float32 - bbox координаты в нормализованных [0,1]
            rois_aligned: [T, P, P, P, C] float32 - ROI features
            mask_aligned: [T, M, M, M, C] float32 - mask features
            target_class_ids: [T] int32 - class IDs
            target_bbox: [T, 6] float32 - bbox deltas
            target_mask: [T, mH, mW, mD, 1] float32 - GT masks
        """
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

        # ✅ ДОБАВЛЕНО: читаем rois
        z_rois = _load_any(info.get("rois")) # или info.get("rois") если в CSV так

        # читаем все артефакты
        z_ra = _load_any(info.get("ra_path"))
        z_ma = _load_any(info.get("ma_path"))
        z_tci = _load_any(info.get("tci_path"))
        z_tb = _load_any(info.get("tb_path"))
        z_tm = _load_any(info.get("tm_path"))

        # ✅ ДОБАВЛЕНО: извлекаем rois
        rois = _pick(z_rois, "rois", "roi", "arr_0")

        rois_aligned = _pick(z_ra, "rois_aligned", "ra", "arr_0")
        mask_aligned = _unbit(z_ma, "mask_bits", "mask_shape", "mask_aligned")
        target_class_ids = _pick(z_tci, "tci", "target_class_ids", "arr_0")
        target_bbox = _pick(z_tb, "bbox", "target_bbox", "arr_0")
        target_mask = _unbit(z_tm, "tm_bits", "tm_shape", "tm")

        # приведение форм и типов
        # ✅ ДОБАВЛЕНО: rois к float32
        if rois is not None:
            rois = np.asarray(rois, dtype=np.float32)

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
            # ✅ ИЗМЕНЕНО: проверяем rois первым
            if rois is not None and rois.ndim >= 1:
                return int(rois.shape[0])
            if rois_aligned is not None and rois_aligned.ndim >= 1:
                return int(rois_aligned.shape[0])
            if target_class_ids is not None and target_class_ids.ndim >= 1:
                return int(target_class_ids.shape[0])
            return 0

        T = _infer_T()

        # ✅ ДОБАВЛЕНО: фолбэк для rois
        if rois is None:
            rois = np.zeros((T, 6), dtype=np.float32)
            print(f"[ToyHeadDataset][{image_id}] Warning: rois missing, using zeros")

        if rois_aligned is None:
            cfg = getattr(self, "config", None)
            if cfg:
                P = int(getattr(cfg, "POOL_SIZE", 7))
                C = int(getattr(cfg, "TOP_DOWN_PYRAMID_SIZE", 256))
            else:
                P, C = 7, 256
            rois_aligned = np.zeros((T, P, P, P, C), dtype=np.float32)
            print(f"[ToyHeadDataset][{image_id}] Warning: rois_aligned missing, using zeros")

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

        # ✅ ИЗМЕНЕНО: возвращаем 6 элементов вместо 5
        return rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask


############################################################
#  Target Generators
############################################################


def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    """
    Генерирует таргеты для RPN с простой нормализацией (деление на H,W,D без смещения).

    Args:
        anchors: [A, 6] нормализованные якоря (y1,x1,z1,y2,x2,z2) / (H,W,D)
        gt_class_ids: [N] class IDs
        gt_boxes: [N, 6] боксы в пикселях (y1,x1,z1,y2,x2,z2)
        config: конфигурация

    Returns:
        rpn_match: [A] int32 (1=pos, -1=neg, 0=ignore)
        rpn_bbox:  [RPN_TRAIN_ANCHORS_PER_IMAGE, 6] float32 (дельты для POS)
    """
    import numpy as np
    from core.utils import compute_overlaps_3d as overlaps3d

    # Параметры из конфига
    pos_iou_thr = float(getattr(config, "RPN_POSITIVE_IOU", 0.15))
    neg_iou_thr = float(getattr(config, "RPN_NEGATIVE_IOU", 0.05))
    total_target_anchors = int(getattr(config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 2048))
    positive_ratio = float(getattr(config, "RPN_POSITIVE_RATIO", 0.5))
    atss_topk = int(getattr(config, "ATSS_TOPK", 24))
    atss_min_pos = int(getattr(config, "ATSS_MIN_POS_PER_GT", 4))

    A = int(anchors.shape[0]) if anchors is not None else 0
    G = int(gt_boxes.shape[0]) if gt_boxes is not None else 0

    rpn_match = np.zeros((A,), dtype=np.int32)
    rpn_bbox = np.zeros((total_target_anchors, 6), dtype=np.float32)

    # Пустые входы → все NEG
    if A == 0 or G == 0:
        rpn_match[:] = -1
        return rpn_match, rpn_bbox

    # Приводим к одной системе координат (нормализованные [0,1])
    anchors_work = anchors.astype(np.float32, copy=False)
    gt_boxes_work = gt_boxes.astype(np.float32, copy=False)

    # Определяем, нужна ли нормализация GT
    anchor_max = float(np.max(np.abs(anchors_work))) if anchors_work.size else 0.0
    gt_max = float(np.max(np.abs(gt_boxes_work))) if gt_boxes_work.size else 0.0

    H = int(getattr(config, "IMAGE_SIZE", getattr(config, "IMAGE_SHAPE", (0, 0, 0))[0]))
    W = int(getattr(config, "IMAGE_SIZE", getattr(config, "IMAGE_SHAPE", (0, 0, 0))[1]))
    D = int(getattr(config, "IMAGE_DEPTH", getattr(config, "IMAGE_SHAPE", (0, 0, 1))[2]))

    # Простая нормализация: деление на H,W,D (БЕЗ H-1)
    def _to_norm(b):
        scale = np.array([H, W, D, H, W, D], dtype=np.float32)
        return np.clip(b / scale, 0.0, 1.0)

    # Якоря norm, GT px → нормализуем GT
    if anchor_max <= 1.5 and gt_max > 2.0:
        gt_boxes_work = _to_norm(gt_boxes_work)

    # GT norm, якоря px → нормализуем anchors
    elif gt_max <= 1.5 and anchor_max > 2.0:
        anchors_work = _to_norm(anchors_work)

    # IoU якоря↔GT
    overlaps = overlaps3d(anchors_work, gt_boxes_work)  # [A, G]
    if overlaps.size == 0:
        rpn_match[:] = -1
        return rpn_match, rpn_bbox

    anchor_iou_max = overlaps.max(axis=1)  # [A]
    gt_argmax = overlaps.argmax(axis=0)  # [G]

    # Минимум: лучший для каждого GT → POS
    rpn_match[gt_argmax] = 1

    # Пороговая маркировка
    rpn_match[anchor_iou_max < neg_iou_thr] = -1
    rpn_match[anchor_iou_max >= pos_iou_thr] = 1

    # ATSS локальная адаптация порога
    for g in range(G):
        ious_g = overlaps[:, g]
        if not np.any(ious_g > 0.0):
            continue
        topk = min(atss_topk, ious_g.shape[0])
        idx = np.argpartition(-ious_g, topk - 1)[:topk]
        mu = float(np.mean(ious_g[idx]))
        sd = float(np.std(ious_g[idx]))
        thr = max(pos_iou_thr, mu + sd)
        cand = np.where(ious_g >= thr)[0]
        if cand.size < atss_min_pos:
            cand = idx[:atss_min_pos]
        rpn_match[cand] = 1

    # Балансировка под батч
    target_pos = int(round(total_target_anchors * positive_ratio))
    pos_ids = np.where(rpn_match == 1)[0]
    if pos_ids.size > target_pos:
        # Оставляем top-K по IoU
        order = np.argsort(-anchor_iou_max[pos_ids])
        drop = pos_ids[order[target_pos:]]
        rpn_match[drop] = 0

    neg_ids = np.where(rpn_match == -1)[0]
    target_neg = int(min(len(neg_ids), total_target_anchors - int(np.sum(rpn_match == 1))))
    if len(neg_ids) > target_neg:
        drop = np.random.choice(neg_ids, size=len(neg_ids) - target_neg, replace=False)
        rpn_match[drop] = 0

    # rpn_bbox: дельты только для POS
    pos_final = np.where(rpn_match == 1)[0]
    if pos_final.size > 0:
        gt_of_pos = np.argmax(overlaps[pos_final], axis=1)

        # Берём нормализованные версии
        anc_pos = anchors_work[pos_final].astype(np.float32, copy=False)
        gt_pos = gt_boxes_work[gt_of_pos].astype(np.float32, copy=False)

        # Конвертация в (cy,cx,cz,h,w,d)
        def _to_cywhd(b):
            y1, x1, z1, y2, x2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]
            h = y2 - y1
            w = x2 - x1
            d = z2 - z1
            cy = y1 + 0.5 * h
            cx = x1 + 0.5 * w
            cz = z1 + 0.5 * d
            return cy, cx, cz, h, w, d

        acy, acx, acz, ah, aw, ad = _to_cywhd(anc_pos)
        gcy, gcx, gcz, gh, gw, gd = _to_cywhd(gt_pos)

        eps = 1e-6
        dy = (gcy - acy) / np.maximum(ah, eps)
        dx = (gcx - acx) / np.maximum(aw, eps)
        dz = (gcz - acz) / np.maximum(ad, eps)
        dh = np.log(np.maximum(gh, eps) / np.maximum(ah, eps))
        dw = np.log(np.maximum(gw, eps) / np.maximum(aw, eps))
        dd = np.log(np.maximum(gd, eps) / np.maximum(ad, eps))

        deltas = np.stack([dy, dx, dz, dh, dw, dd], axis=1).astype(np.float32)

        # std-нормировка как в ProposalLayer
        std = np.array(getattr(config, "RPN_BBOX_STD_DEV", [0.10, 0.10, 0.16, 0.22, 0.22, 0.30]), dtype=np.float32)
        deltas = deltas / std[None, :]

        count = min(deltas.shape[0], total_target_anchors)
        rpn_bbox[:count, :] = deltas[:count, :]

    return rpn_match, rpn_bbox




