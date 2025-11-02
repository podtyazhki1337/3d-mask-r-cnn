import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import re
import math
import multiprocessing
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave

from core import utils
from core.utils import parse_image_meta_graph

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# ---- ЕДИНЫЙ TF1-стиль графа и один сеанс ----
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

K.clear_session()

_tf_config = tf.compat.v1.ConfigProto()
_tf_config.gpu_options.allow_growth = True
_TF_SESSION = tf.compat.v1.Session(config=_tf_config)

# Привязываем сессию к Keras (и к tf.compat)
try:
    if hasattr(K, "set_session"):
        K.set_session(_TF_SESSION)
    if hasattr(tf.compat.v1.keras.backend, "set_session"):
        tf.compat.v1.keras.backend.set_session(_TF_SESSION)
except Exception as e:
    print("[tf-compat] set_session warn:", e)

def _get_value_safe(x):
    import numpy as np
    # быстрый путь для скаляров/np
    if isinstance(x, (float, int, np.floating, np.integer)):
        return x
    if isinstance(x, np.ndarray):
        return x

    # TensorFlow объекты: ResourceVariable / Variable / Tensor
    try:
        import tensorflow as _tf
        try:
            from tensorflow.python.ops.resource_variable_ops import ResourceVariable as _RV
        except Exception:
            _RV = None
        if _RV is not None and isinstance(x, _RV):
            try:
                return _TF_SESSION.run(x.read_value())
            except Exception:
                try:
                    return _TF_SESSION.run(x)
                except Exception:
                    pass
        if isinstance(x, (_tf.Variable, _tf.Tensor)):
            try:
                return _TF_SESSION.run(x)
            except Exception:
                pass
    except Exception:
        pass

    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return x

# переназначаем бекенд
keras.backend.get_value = _get_value_safe
K.get_value = _get_value_safe

if platform.processor() == 'ppc64le':
    import core.custom_op.ppc64le_custom_op as custom_op
else:
    import core.custom_op.custom_op as custom_op

from core.utils import rpn_evaluation, head_evaluation, compute_ap
from core.data_generators import RPNGenerator, HeadGenerator, MrcnnGenerator, ToyDataset, ToyHeadDataset

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

class BatchNorm(KL.BatchNormalization):
    """
    Extends the Keras BatchNormalization class.

    CRITICAL: Respects layer.trainable flag!
    """

    def call(self, inputs, training=None):
        # ✅ КРИТИЧНО: если слой заморожен → training=False всегда!
        if not self.trainable:
            training = False

        return super(BatchNorm, self).call(inputs, training=training)


def _keras_opt_params(p):
    p = dict(p or {})
    if 'learning_rate' in p and 'lr' not in p:
        p['lr'] = p.pop('learning_rate')
    if 'beta1' in p and 'beta_1' not in p:
        p['beta_1'] = p.pop('beta1')
    if 'beta2' in p and 'beta_2' not in p:
        p['beta_2'] = p.pop('beta2')
    return p

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
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
    return np.array(shapes)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter1, (1, 1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               #strides=(2, 2, 2), use_bias=True, train_bn=True):
               strides=(2, 2, 1), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter1, (1, 1, 1), strides=strides, name=conv_name_base + '2a', 
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv3D(nb_filter3, (1, 1, 1), strides=strides, name=conv_name_base + '1', 
                         use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=False):
    """ResNet для переменной глубины"""

    assert architecture in ["resnet50", "resnet101"]

    # Stage 1 - НЕ уменьшаем глубину в начале
    x = KL.ZeroPadding3D((3, 3, 3))(input_image)
    x = KL.Conv3D(64, (7, 7, 7), strides=(2, 2, 1), name='conv1', use_bias=True)(x)  # (2,2,1)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding="same")(x)  # (2,2,1)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2, 1), train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2, 2, 1), train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(2, 2, 1), train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """
    Применяет регрессионные дельты к 3D-боксам в формате [y1, x1, z1, y2, x2, z2].
    Нормализация: boxes нормализованы делением на H,W,D (без H-1).
    """
    boxes = tf.cast(boxes, tf.float32)
    deltas = tf.cast(deltas, tf.float32)
    deltas = tf.clip_by_value(deltas, -3.0, 3.0)
    # ✅ ДИАГНОСТИКА ВХОДОВ (TF1 синтаксис)
    # boxes = tf.compat.v1.Print(boxes, [
    #     tf.shape(boxes),
    #     tf.reduce_min(boxes),
    #     tf.reduce_max(boxes),
    #     tf.reduce_mean(boxes),
    # ], message="[APPLY_DELTAS_IN] boxes shape/min/max/mean: ", summarize=10)
    #
    # deltas = tf.compat.v1.Print(deltas, [
    #     tf.shape(deltas),
    #     tf.reduce_min(deltas),
    #     tf.reduce_max(deltas),
    #     tf.reduce_mean(tf.abs(deltas)),
    # ], message="[APPLY_DELTAS_IN] deltas shape/min/max/mean_abs: ", summarize=10)

    # Высоты/ширины/глубины
    height = boxes[:, 3] - boxes[:, 0]
    width = boxes[:, 4] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 2]

    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 2] + 0.5 * depth

    # Применяем дельты
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= tf.exp(deltas[:, 3])
    width *= tf.exp(deltas[:, 4])
    depth *= tf.exp(deltas[:, 5])

    # Обратно к углам
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth

    result = tf.stack([y1, x1, z1, y2, x2, z2], axis=1)
    result = tf.clip_by_value(result, 0.0, 1.0)
    # ✅ ДИАГНОСТИКА ВЫХОДА
    # result = tf.compat.v1.Print(result, [
    #     tf.reduce_min(result),
    #     tf.reduce_max(result),
    #     tf.reduce_mean(result),
    # ], message="[APPLY_DELTAS_OUT] min/max/mean: ", summarize=10)

    return result





def clip_boxes_graph(boxes, window):
    """
    Clip the anchor boxes within the normalized image space.

    Args:
        boxes: [N, (y1, x1, z1, y2, x2, z2)]
        window: [6] in the form y1, x1, z1, y2, x2, z2

    Returns:
        clipped: the clipped boxes
    """
    # Split
    wy1, wx1, wz1, wy2, wx2, wz2 = tf.split(window, 6)
    y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    z1 = tf.maximum(tf.minimum(z1, wz2), wz1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    z2 = tf.maximum(tf.minimum(z2, wz2), wz1)
    clipped = tf.concat([y1, x1, z1, y2, x2, z2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 6))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals."""

    def __init__(self, proposal_count, nms_threshold, pre_nms_limit, images_per_gpu,
                 rpn_bbox_std_dev, image_depth, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.image_depth = int(image_depth)

    def call(self, inputs):
        """
        rpn_probs:  [B, N, 2]
        rpn_bbox:   [B, N, 6]  (dy,dx,dz,log(dh),log(dw),log(dd)) НОРМАЛИЗОВАННЫЕ по std
        anchors:    [B, N, 6]  НОРМАЛИЗОВАННЫЕ (y1,x1,z1,y2,x2,z2) делением на H,W,D
        -> proposals: [B, P, 6] (нормированные)
        """
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        anchors = inputs[2]

        scores = tf.cast(scores, tf.float32)
        deltas = tf.cast(deltas, tf.float32)
        anchors = tf.cast(anchors, tf.float32)

        # Де-нормализация дельт по std
        std = tf.constant(self.rpn_bbox_std_dev, dtype=tf.float32)
        deltas = deltas * tf.reshape(std, [1, 1, 6])
        deltas = tf.clip_by_value(deltas, -3.0, 3.0)

        # Top-K до NMS
        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        topk_idx = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="rpn_topk").indices

        scores = utils.batch_slice([scores, topk_idx],
                                   lambda s, i: tf.gather(s, i),
                                   self.images_per_gpu)
        deltas = utils.batch_slice([deltas, topk_idx],
                                   lambda d, i: tf.gather(d, i),
                                   self.images_per_gpu)
        pre_nms_anchors = utils.batch_slice([anchors, topk_idx],
                                            lambda a, i: tf.gather(a, i),
                                            self.images_per_gpu,
                                            names=["pre_nms_anchors"])

        # ✅ КРИТИЧНО: anchors УЖЕ нормализованы [0,1]
        # Применяем дельты → получаем boxes в ПИКСЕЛЯХ!
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda a, d: apply_box_deltas_graph(a, d),
                                  self.images_per_gpu,
                                  names=["refined_anchors"])

        # ❌ ПРОБЛЕМА: apply_box_deltas_graph денормализует anchors!
        # Нужно проверить эту функцию...

        # Клип в [0,1]
        window = tf.constant([0., 0., 0., 1., 1., 1.], dtype=tf.float32)
        boxes = utils.batch_slice([boxes],
                                  lambda b: clip_boxes_graph(b, window),
                                  self.images_per_gpu,
                                  names=["refined_anchors_clipped"])

        # Enforce min sizes
        eps = tf.constant(1e-6, dtype=tf.float32)
        img_depth = tf.maximum(tf.cast(self.image_depth, tf.float32), 1.0)
        min_d_norm = tf.maximum(1.0 / img_depth, 1e-4)

        def _enforce_min_size(b):
            y1, x1, z1, y2, x2, z2 = tf.split(b, 6, axis=1)
            y2 = tf.maximum(y2, y1 + eps)
            x2 = tf.maximum(x2, x1 + eps)
            z2 = tf.maximum(z2, z1 + min_d_norm)
            return tf.concat([y1, x1, z1, y2, x2, z2], axis=1)

        boxes = utils.batch_slice([boxes], lambda b: _enforce_min_size(b),
                                  self.images_per_gpu, names=["refined_min_size"])

        # NMS
        def _nms_single(b, s):
            def _try_custom_op():
                try:
                    idx = custom_op.non_max_suppression_3d(
                        b, s, self.proposal_count, self.nms_threshold, name="rpn_non_max_suppression"
                    )
                    return tf.gather(b, idx)
                except Exception:
                    return None

            def _try_utils_nms():
                try:
                    idx = utils.nms_3d(b, s, self.nms_threshold)
                    idx = idx[:self.proposal_count]
                    return tf.gather(b, idx)
                except Exception:
                    return None

            props = _try_custom_op()
            if props is None:
                props = _try_utils_nms()
            if props is None:
                k = tf.minimum(tf.shape(b)[0], tf.cast(self.proposal_count, tf.int32))
                idx = tf.nn.top_k(s, k, sorted=True).indices
                props = tf.gather(b, idx)

            cur = tf.shape(props)[0]
            need = self.proposal_count - cur

            def _pad():
                pad = tf.zeros((need, 6), dtype=props.dtype)
                return tf.concat([props, pad], axis=0)

            props = tf.cond(need > 0, _pad, lambda: props)
            props.set_shape([self.proposal_count, 6])
            return props

        proposals = utils.batch_slice([boxes, scores],
                                      lambda b, s: _nms_single(b, s),
                                      self.images_per_gpu,
                                      names=["rpn_nms"])

        # ✅ ДИАГНОСТИКА: проверяем финальные proposals
        # proposals = tf.compat.v1.Print(proposals, [
        #     tf.shape(proposals),
        #     tf.reduce_min(proposals),
        #     tf.reduce_max(proposals),
        #     tf.reduce_mean(proposals),
        # ], message="[PROPOSAL_LAYER] final proposals shape/min/max/mean: ", summarize=10)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 6)




############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Builds the computation graph of Region Proposal Network with improved architecture.

    Args:
        feature_map: backbone features [batch, height, width, depth, channels]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits, rpn_probs, rpn_bbox
    """

    # Слой нормализации для стабилизации обучения
    shared = feature_map

    # Улучшенная архитектура с двумя сверточными слоями
    shared = KL.Conv3D(512, (3, 3, 3), padding='same', activation='relu',
                       strides=anchor_stride, name='rpn_conv_shared1')(shared)

    # Dropout для регуляризации
    #shared = KL.Dropout(0.1, name='rpn_dropout')(shared)

    shared = KL.Conv3D(256, (1, 1, 1), padding='same', activation='relu',
                       name='rpn_conv_shared2')(shared)

    # Anchor Score. [batch, height, width, depth, anchors per location * 2].
    x = KL.Conv3D(2 * anchors_per_location, (1, 1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement
    x = KL.Conv3D(anchors_per_location * 6, (1, 1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred',
                  kernel_initializer=keras.initializers.RandomNormal(stddev=0.001))(shared)

    # Reshape to [batch, anchors, 6]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 6]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, channel):
    """
        Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        Args:
            anchors_per_location: number of anchors per pixel in the feature map
            anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                           every pixel in the feature map), or 2 (every other pixel).
            channels: number of channels of a backbone feature map.
            shared_layer_size: channel number of the RPN shared block

        Returns:
            rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to be
                        applied to anchors.
        """
    
    input_feature_map = KL.Input(shape=[None, None, None, channel], name="input_rpn_feature_map")

    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)

    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""

    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KE.Layer):
    """ИСПРАВЛЕННАЯ версия для переменной глубины"""

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        """
        PyramidROIAlign для 3D с правильной обработкой нормализованных координат
        """
        boxes, image_meta = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        eps = tf.constant(1e-6, dtype=boxes.dtype)
        y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=2)

        # Клип в [0,1]
        y1 = tf.clip_by_value(y1, 0.0, 1.0)
        x1 = tf.clip_by_value(x1, 0.0, 1.0)
        z1 = tf.clip_by_value(z1, 0.0, 1.0)
        y2 = tf.clip_by_value(y2, 0.0, 1.0)
        x2 = tf.clip_by_value(x2, 0.0, 1.0)
        z2 = tf.clip_by_value(z2, 0.0, 1.0)

        # Минимальные размеры
        y2 = tf.maximum(y2, y1 + eps)
        x2 = tf.maximum(x2, x1 + eps)

        # ✅ ИСПРАВЛЕНО: правильный минимум для Z
        meta = parse_image_meta_graph(image_meta)
        image_shape = meta['image_shape'][:, :3]  # [B,(H,W,D)]
        D = tf.cast(image_shape[:, 2], tf.float32)
        min_dz = 1.0 / tf.maximum(D, 1.0)  # ✅ без -1.0
        min_dz = tf.expand_dims(tf.expand_dims(min_dz, 1), 2)
        z2 = tf.maximum(z2, z1 + min_dz)

        boxes = tf.concat([y1, x1, z1, y2, x2, z2], axis=2)

        # Выбор уровня пирамиды
        hroi = y2 - y1
        wroi = x2 - x1
        droi = z2 - z1

        H = tf.cast(image_shape[:, 0], tf.float32)
        W = tf.cast(image_shape[:, 1], tf.float32)
        image_area = H * W * D

        roi_volume = hroi * wroi * droi
        roi_level = log2_graph(tf.pow(roi_volume, 1.0 / 3.0) /
                               (224.0 / tf.pow(image_area, 1.0 / 3.0)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Crop and resize по уровням
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_indices = tf.cast(ix[:, 0], tf.int32)
            box_to_level.append(ix)

            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(custom_op.crop_and_resize_3d(
                feature_maps[i], level_boxes, box_indices, self.pool_shape))

        # Восстановление порядка
        if len(pooled) > 0:
            pooled = tf.concat(pooled, axis=0)
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

            sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)
        else:
            pooled = tf.zeros([0] + list(self.pool_shape) + [feature_maps[0].shape[-1]],
                              dtype=feature_maps[0].dtype)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        pooled = tf.where(tf.math.is_finite(pooled), pooled, tf.zeros_like(pooled))

        return pooled
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)



############################################################
#  Detection Target Layer
############################################################
#
def overlaps_graph(boxes1, boxes2):
    """Вычисляет IoU между двумя наборами 3D боксов.

    boxes1, boxes2: [N, (y1, x1, z1, y2, x2, z2)]

    Returns: [N, M] IoU overlaps
    """
    # Преобразуем в float32
    b1 = tf.cast(boxes1, tf.float32)
    b2 = tf.cast(boxes2, tf.float32)

    # Expand dimensions для broadcasting
    b1 = tf.expand_dims(b1, 1)  # [N, 1, 6]
    b2 = tf.expand_dims(b2, 0)  # [1, M, 6]

    # Intersection
    y1 = tf.maximum(b1[..., 0], b2[..., 0])
    x1 = tf.maximum(b1[..., 1], b2[..., 1])
    z1 = tf.maximum(b1[..., 2], b2[..., 2])
    y2 = tf.minimum(b1[..., 3], b2[..., 3])
    x2 = tf.minimum(b1[..., 4], b2[..., 4])
    z2 = tf.minimum(b1[..., 5], b2[..., 5])

    intersection = tf.maximum(y2 - y1, 0) * \
                   tf.maximum(x2 - x1, 0) * \
                   tf.maximum(z2 - z1, 0)

    # Volumes
    b1_vol = (b1[..., 3] - b1[..., 0]) * \
             (b1[..., 4] - b1[..., 1]) * \
             (b1[..., 5] - b1[..., 2])
    b2_vol = (b2[..., 3] - b2[..., 0]) * \
             (b2[..., 4] - b2[..., 1]) * \
             (b2[..., 5] - b2[..., 2])

    union = b1_vol + b2_vol - intersection
    iou = intersection / tf.maximum(union, 1e-10)

    return iou


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks,
                            train_rois_per_image=None, roi_positive_ratio=None,
                            bbox_std_dev=None, use_mini_mask=None, mask_shape=None,
                            config=None):
    """
    ВЕРСИЯ С РАСШИРЕННЫМ ДЕБАГОМ через tf.Print + диагностика координат
    Использует пороги IoU из config для согласованности с RPN
    """
    import tensorflow as tf

    # Проверяем, передан ли config или используются старые аргументы
    if config is not None:
        train_rois_per_image = int(getattr(config, "TRAIN_ROIS_PER_IMAGE", 256))
        roi_positive_ratio = float(getattr(config, "ROI_POSITIVE_RATIO", 0.5))
        bbox_std_dev = tf.constant(getattr(config, "BBOX_STD_DEV", [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]), tf.float32)
        use_mini_mask = bool(getattr(config, "USE_MINI_MASK", True))
        mask_shape = tuple(getattr(config, "MASK_SHAPE", (28, 28, 28)))
        positive_iou_threshold = float(getattr(config, "RPN_POSITIVE_IOU", 0.25))
        negative_iou_threshold = float(getattr(config, "RPN_NEGATIVE_IOU", 0.15))
    else:
        if not isinstance(bbox_std_dev, tf.Tensor):
            bbox_std_dev = tf.constant(bbox_std_dev, tf.float32)
        positive_iou_threshold = 0.5
        negative_iou_threshold = 0.5

    # ✅ КРИТИЧЕСКАЯ ДИАГНОСТИКА ВХОДНЫХ ДАННЫХ
    def _get_first_proposals():
        return tf.cond(
            tf.greater(tf.shape(proposals)[0], 0),
            lambda: proposals[0, :3],
            lambda: tf.zeros([3], dtype=tf.float32)
        )

    first_3 = _get_first_proposals()

    # proposals = tf.compat.v1.Print(proposals, [
    #     tf.shape(proposals),
    #     tf.reduce_min(proposals),
    #     tf.reduce_max(proposals),
    #     tf.reduce_mean(proposals),
    # ], message="[DET_TGT_INPUT] proposals shape/min/max/mean: ", summarize=10)
    #
    # gt_boxes = tf.compat.v1.Print(gt_boxes, [
    #     tf.shape(gt_boxes),
    #     tf.reduce_min(gt_boxes),
    #     tf.reduce_max(gt_boxes),
    #     tf.reduce_mean(gt_boxes),
    # ], message="[DET_TGT_INPUT] gt_boxes shape/min/max/mean: ", summarize=10)

    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion")
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=3, name="trim_gt_masks")

    num_proposals = tf.shape(proposals)[0]
    num_gt = tf.shape(gt_boxes)[0]

    # ✅ ДИАГНОСТИКА ПОСЛЕ TRIM
    # proposals = tf.compat.v1.Print(proposals, [
    #     num_proposals,
    #     tf.reduce_min(proposals),
    #     tf.reduce_max(proposals),
    # ], message="[DET_TGT_TRIM] num_proposals/min/max: ", summarize=10)
    #
    # gt_boxes = tf.compat.v1.Print(gt_boxes, [
    #     num_gt,
    #     tf.reduce_min(gt_boxes),
    #     tf.reduce_max(gt_boxes),
    # ], message="[DET_TGT_TRIM] num_gt/min/max: ", summarize=10)

    def _empty_targets():
        """Возвращает пустые паддинговые тензоры"""
        P = train_rois_per_image
        mask_h, mask_w, mask_d = mask_shape

        rois = tf.zeros([P, 6], dtype=tf.float32)
        roi_gt_boxes = tf.zeros([P, 6], dtype=tf.float32)
        roi_gt_class_ids = tf.zeros([P], dtype=tf.int32)
        deltas = tf.zeros([P, 6], dtype=tf.float32)
        masks = tf.zeros([P, mask_h, mask_w, mask_d], dtype=tf.float32)

        # rois = tf.compat.v1.Print(rois, [
        #     num_proposals, num_gt
        # ], message="[DET_TGT_EMPTY] num_proposals/num_gt: ", summarize=10)

        return [rois, roi_gt_boxes, roi_gt_class_ids, deltas, masks]

    def _compute_targets():
        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = overlaps_graph(proposals, gt_boxes)

        # ✅ ДИАГНОСТИКА OVERLAPS
        # overlaps = tf.compat.v1.Print(overlaps, [
        #     tf.shape(overlaps),
        #     tf.reduce_min(overlaps),
        #     tf.reduce_max(overlaps),
        #     tf.reduce_mean(overlaps),
        # ], message="[DET_TGT_OVERLAPS] shape/min/max/mean: ", summarize=10)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        # DEBUG: Базовая статистика
        # roi_iou_max = tf.compat.v1.Print(roi_iou_max, [
        #     num_proposals,
        #     num_gt,
        #     tf.reduce_min(roi_iou_max),
        #     tf.reduce_mean(roi_iou_max),
        #     tf.reduce_max(roi_iou_max),
        #     tf.constant(positive_iou_threshold),
        #     tf.constant(negative_iou_threshold)
        # ], message="[DET_TGT] num_props/num_gt/IoU_min/mean/max/pos_thr/neg_thr: ", summarize=10)

        # 1. Positive ROIs
        positive_roi_bool = (roi_iou_max >= positive_iou_threshold)
        positive_indices = tf.where(positive_roi_bool)[:, 0]

        # Считаем распределение по порогам IoU
        num_iou_ge_01 = tf.reduce_sum(tf.cast(roi_iou_max >= 0.1, tf.int32))
        num_iou_ge_02 = tf.reduce_sum(tf.cast(roi_iou_max >= 0.2, tf.int32))
        num_iou_ge_03 = tf.reduce_sum(tf.cast(roi_iou_max >= 0.3, tf.int32))
        num_iou_ge_04 = tf.reduce_sum(tf.cast(roi_iou_max >= 0.4, tf.int32))
        num_iou_ge_05 = tf.reduce_sum(tf.cast(roi_iou_max >= 0.5, tf.int32))

        # Серая зона
        gray_zone_bool = tf.logical_and(
            roi_iou_max >= negative_iou_threshold,
            roi_iou_max < positive_iou_threshold
        )
        num_gray_zone = tf.reduce_sum(tf.cast(gray_zone_bool, tf.int32))

        # DEBUG: Распределение IoU + серая зона
        # positive_indices = tf.compat.v1.Print(positive_indices, [
        #     num_iou_ge_01,
        #     num_iou_ge_02,
        #     num_iou_ge_03,
        #     num_iou_ge_04,
        #     num_iou_ge_05,
        #     num_gray_zone
        # ], message="[DET_TGT] IoU distribution (>=0.1/0.2/0.3/0.4/0.5) | gray_zone: ", summarize=10)

        # 2. Negative ROIs
        negative_indices = tf.where(roi_iou_max < negative_iou_threshold)[:, 0]

        num_positive = tf.shape(positive_indices)[0]
        num_negative = tf.shape(negative_indices)[0]

        # DEBUG: Кандидаты
        # num_positive = tf.compat.v1.Print(num_positive, [
        #     num_positive,
        #     num_negative
        # ], message="[DET_TGT] positive_candidates/negative_candidates: ", summarize=10)

        # Subsample ROIs
        positive_count = tf.cast(
            tf.cast(train_rois_per_image, tf.float32) * roi_positive_ratio,
            tf.int32
        )
        positive_count = tf.minimum(positive_count, num_positive)
        positive_count = tf.maximum(positive_count, 0)

        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]

        # Negative ROIs
        negative_count = train_rois_per_image - tf.shape(positive_indices)[0]
        negative_count = tf.minimum(negative_count, num_negative)
        negative_count = tf.maximum(negative_count, 0)

        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        # DEBUG: Финальное количество
        # positive_indices = tf.compat.v1.Print(positive_indices, [
        #     tf.shape(positive_indices)[0],
        #     negative_count,
        #     train_rois_per_image
        # ], message="[DET_TGT] final_positive/negative/target_total: ", summarize=10)

        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[0], 0),
            lambda: tf.argmax(positive_overlaps, axis=1),
            lambda: tf.constant([], dtype=tf.int64)
        )
        roi_gt_boxes = tf.cond(
            tf.greater(tf.shape(roi_gt_box_assignment)[0], 0),
            lambda: tf.gather(gt_boxes, roi_gt_box_assignment),
            lambda: tf.zeros([0, 6], dtype=tf.float32)
        )
        roi_gt_class_ids = tf.cond(
            tf.greater(tf.shape(roi_gt_box_assignment)[0], 0),
            lambda: tf.gather(gt_class_ids, roi_gt_box_assignment),
            lambda: tf.constant([], dtype=tf.int32)
        )

        # DEBUG: GT assignment
        # roi_gt_class_ids = tf.compat.v1.Print(roi_gt_class_ids, [
        #     tf.shape(roi_gt_class_ids)[0],
        #     tf.cond(tf.shape(roi_gt_class_ids)[0] > 0,
        #             lambda: tf.reduce_min(roi_gt_class_ids),
        #             lambda: tf.constant(0, dtype=tf.int32)),
        #     tf.cond(tf.shape(roi_gt_class_ids)[0] > 0,
        #             lambda: tf.reduce_max(roi_gt_class_ids),
        #             lambda: tf.constant(0, dtype=tf.int32))
        # ], message="[DET_TGT] assigned_gt_count/min_class_id/max_class_id: ", summarize=10)

        # Compute bbox refinement
        deltas = tf.cond(
            tf.greater(tf.shape(positive_rois)[0], 0),
            lambda: utils.box_refinement_graph(positive_rois, roi_gt_boxes) / bbox_std_dev,
            lambda: tf.zeros([0, 6], dtype=tf.float32)
        )

        # DEBUG: Deltas
        # deltas = tf.compat.v1.Print(deltas, [
        #     tf.cond(tf.shape(deltas)[0] > 0,
        #             lambda: tf.reduce_mean(tf.abs(deltas)),
        #             lambda: tf.constant(0.0, dtype=tf.float32)),
        #     tf.cond(tf.shape(deltas)[0] > 0,
        #             lambda: tf.reduce_max(tf.abs(deltas)),
        #             lambda: tf.constant(0.0, dtype=tf.float32))
        # ], message="[DET_TGT] bbox_deltas mean_abs/max_abs: ", summarize=10)

        # Masks
        def _get_masks():
            transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [3, 0, 1, 2]), -1)
            roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

            boxes = positive_rois
            if use_mini_mask:
                y1, x1, z1, y2, x2, z2 = tf.split(positive_rois, 6, axis=1)
                gt_y1, gt_x1, gt_z1, gt_y2, gt_x2, gt_z2 = tf.split(roi_gt_boxes, 6, axis=1)
                gt_h = gt_y2 - gt_y1
                gt_w = gt_x2 - gt_x1
                gt_d = gt_z2 - gt_z1
                y1 = (y1 - gt_y1) / gt_h
                x1 = (x1 - gt_x1) / gt_w
                z1 = (z1 - gt_z1) / gt_d
                y2 = (y2 - gt_y1) / gt_h
                x2 = (x2 - gt_x1) / gt_w
                z2 = (z2 - gt_z1) / gt_d
                boxes = tf.concat([y1, x1, z1, y2, x2, z2], 1)

            box_ids = tf.range(0, tf.shape(roi_masks)[0])
            masks = custom_op.crop_and_resize_3d(
                tf.cast(roi_masks, tf.float32), boxes, box_ids, mask_shape
            )
            masks = tf.squeeze(masks, axis=4)
            masks = tf.round(masks)

            # DEBUG: Маски
            # masks = tf.compat.v1.Print(masks, [
            #     tf.shape(masks)[0],
            #     tf.reduce_mean(masks),
            #     tf.reduce_sum(tf.cast(tf.reduce_sum(masks, axis=[1, 2, 3]) > 0, tf.int32))
            # ], message="[DET_TGT] masks count/mean_val/num_non_empty: ", summarize=10)

            return masks

        masks = tf.cond(
            tf.greater(tf.shape(positive_rois)[0], 0),
            _get_masks,
            lambda: tf.zeros([0, mask_shape[0], mask_shape[1], mask_shape[2]], dtype=tf.float32)
        )

        # Concatenate and pad
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(train_rois_per_image - tf.shape(rois)[0], 0)

        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [(0, N + P), (0, 0), (0, 0), (0, 0)])

        # DEBUG: Финальные размеры
        # rois = tf.compat.v1.Print(rois, [
        #     tf.shape(rois)[0],
        #     tf.shape(roi_gt_class_ids)[0],
        #     tf.reduce_sum(tf.cast(roi_gt_class_ids > 0, tf.int32)),
        #     P
        # ], message="[DET_TGT] final_rois/class_ids/num_positive/padding: ", summarize=10)

        return [rois, roi_gt_boxes, roi_gt_class_ids, deltas, masks]

    result = tf.cond(
        tf.logical_and(tf.greater(num_proposals, 0), tf.greater(num_gt, 0)),
        _compute_targets,
        _empty_targets
    )

    return result[0], result[1], result[2], result[3], result[4]


class DetectionTargetLayer(KE.Layer):
    """
    Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
        proposals: [batch, N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, z1, y2, x2, z2)] in normalized
                  coordinates.
        gt_masks: [batch, height, width, depth, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, z1, y2, x2, z2)] in normalized
              coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, dz, log(dh), log(dw), log(dd)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width, depth]
                     Masks cropped to bbox boundaries and resized to neural
                     network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, train_rois_per_image, roi_positive_ratio, bbox_std_dev,
                 use_mini_mask, mask_shape, images_per_gpu,
                 positive_iou_threshold=0.5, negative_iou_threshold=0.1, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config  # ← ДОБАВИЛИ
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.mask_shape = mask_shape
        self.images_per_gpu = images_per_gpu
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_gt_boxes", "target_class_ids", "target_bbox", "target_mask"]

        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z,
                config=self.config
            ),
            self.images_per_gpu,
            names=names
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 6),  # rois
            (None, self.train_rois_per_image, 6),  # roi_gt_boxes ← УЖЕ ПРАВИЛЬНО
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 6),  # deltas
            (None, self.train_rois_per_image, *self.mask_shape)  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(y, pool_size, num_classes, fc_layers_size, train_bn=True):
    """Classifier head с динамическим reshape для inference."""
    import numpy as np
    import tensorflow as tf
    import keras.layers as KL
    from keras import backend as K

    # ✅ Conv1
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(momentum=0.9), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # ✅ Conv2
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(momentum=0.9), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # ✅ ИСПРАВЛЕНО: Динамический reshape через Lambda
    def _reshape_shared(t):
        b = tf.shape(t)[0]
        n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, fc_layers_size))

    shared = KL.Lambda(_reshape_shared, name="pool_reshape")(x)

    # Bias инициализация
    fg_prior = 0.15
    bias_init = np.array([
        -np.log((1 - fg_prior) / fg_prior),
        np.log(fg_prior / (1 - fg_prior))
    ], dtype=np.float32)

    print(f"[CLASSIFIER] Initializing bias: bg={bias_init[0]:.3f}, fg={bias_init[1]:.3f}")

    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes,
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                 bias_initializer=keras.initializers.Constant(bias_init),
                 kernel_constraint=keras.constraints.MaxNorm(max_value=2.0)),
        name='mrcnn_class_logits'
    )(shared)

    mrcnn_class_logits = KL.Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0),
                                   name="mrcnn_class_logits_clipped")(mrcnn_class_logits)

    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # ✅ BBox head
    x = KL.TimeDistributed(
        KL.Dense(num_classes * 6,
                 activation='linear',
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                 bias_initializer='zeros',
                 kernel_constraint=keras.constraints.MaxNorm(max_value=1.0)),
        name='mrcnn_bbox_fc'
    )(shared)

    # ✅ ИСПРАВЛЕНО: Динамический reshape для bbox
    def _reshape_bbox(t):
        b = tf.shape(t)[0]
        n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, num_classes, 6))

    mrcnn_bbox = KL.Lambda(_reshape_bbox, name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(y, num_classes, conv_channel, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.

    y: [batch, num_rois, mH, mW, mD, C] — уже ROIAlign-выравненные фичи под маски.
    num_classes: глубина выхода по классам.
    conv_channel: ширина каналов для Conv3D-слоёв головы.
    train_bn: обучать ли BatchNorm.
    Возвращает: [batch, num_rois, mH*2, mW*2, mD*2, num_classes]
    """
    # Conv1
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn1")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    # Conv2
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    # Conv3 (+ dilated residual блок)
    res = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    res = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn3")(res, training=train_bn)
    res = KL.Activation("relu")(res)

    x_dil = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same", dilation_rate=(2, 2, 2)),
        name="mrcnn_mask_conv3b"
    )(res)
    x_dil = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn3b")(x_dil, training=train_bn)
    x_dil = KL.Activation("relu")(x_dil)

    x = KL.Add(name="mrcnn_mask_res3")([res, x_dil])

    # Conv4
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn4")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    # Upsample + logits
    x = KL.TimeDistributed(
        KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"),
        name="mrcnn_mask_deconv"
    )(x)
    x = KL.TimeDistributed(
        KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"),
        name="mrcnn_mask"
    )(x)
    return x


def fpn_classifier_graph_with_RoiAlign(rois, feature_maps, image_meta,
                                       pool_size, num_classes, fc_layers_size,
                                       train_bn=True, name_prefix="", config=None):
    """
    ✅ ПРОСТОЕ РЕШЕНИЕ: если батчевый режим, берём топ-N ROI по RPN score
    """
    import keras.layers as KL
    from keras import backend as K
    import numpy as np
    import tensorflow as tf

    N = (lambda n: f"{name_prefix}{n}") if name_prefix else (lambda n: n)

    # ✅ Проверяем нужно ли ограничение ROI
    max_rois = None
    if config is not None:
        use_batching = config.HEAD_CONV_CHANNEL < config.IMAGE_SHAPE[0]
        if use_batching:
            max_rois = getattr(config, 'HEAD_MAX_ROIS', 500)  # По умолчанию 500 ROI

    # ✅ ИСПРАВЛЕНО: Берём первые ROI (они УЖЕ отсортированы по score после ProposalLayer!)
    # ProposalLayer возвращает proposals отсортированные по убыванию score
    if max_rois is not None:
        def limit_rois(roi_tensor):
            """Берём только первые max_rois (уже отсортированы по score)"""
            num_rois = tf.shape(roi_tensor)[1]
            actual_limit = tf.minimum(num_rois, max_rois)
            return roi_tensor[:, :actual_limit, :]

        rois = KL.Lambda(limit_rois, name=N("limit_rois"))(rois)

    # ========== ОРИГИНАЛЬНЫЙ КОД (без изменений) ==========
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name=N("roi_align_classifier"))(
        [rois, image_meta] + feature_maps
    )

    x = KL.TimeDistributed(
        KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
        name=N("mrcnn_class_conv1")
    )(x)
    x = KL.TimeDistributed(BatchNorm(momentum=0.9), name=N('mrcnn_class_bn1'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name=N("mrcnn_class_conv2"))(x)
    x = KL.TimeDistributed(BatchNorm(momentum=0.9), name=N('mrcnn_class_bn2'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    def _reshape_shared(t):
        b = tf.shape(t)[0]
        n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, fc_layers_size))

    shared = KL.Lambda(_reshape_shared, name=N("pool_reshape"))(x)

    fg_prior = 0.15
    bias_init = np.array([
        -np.log((1 - fg_prior) / fg_prior),
        np.log(fg_prior / (1 - fg_prior))
    ], dtype=np.float32)

    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes,
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                 bias_initializer=keras.initializers.Constant(bias_init),
                 kernel_constraint=keras.constraints.MaxNorm(max_value=2.0)),
        name=N('mrcnn_class_logits')
    )(shared)

    mrcnn_class_logits = KL.Lambda(
        lambda x: tf.clip_by_value(x, -10.0, 10.0),
        name=N("mrcnn_class_logits_clipped")
    )(mrcnn_class_logits)

    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name=N("mrcnn_class"))(mrcnn_class_logits)

    x = KL.TimeDistributed(
        KL.Dense(num_classes * 6,
                 activation='linear',
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                 bias_initializer='zeros',
                 kernel_constraint=keras.constraints.MaxNorm(max_value=1.0)),
        name=N('mrcnn_bbox_fc')
    )(shared)

    def _reshape_bbox(t):
        b = tf.shape(t)[0]
        n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, num_classes, 6))

    mrcnn_bbox = KL.Lambda(_reshape_bbox, name=N("mrcnn_bbox"))(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph_with_RoiAlign(rois, feature_maps, image_meta,
                                       pool_size, num_classes, conv_channel,
                                       train_bn=True, name_prefix="", config=None):
    """
    ✅ ПРОСТОЕ РЕШЕНИЕ: если батчевый режим, берём топ-N ROI
    """
    import keras.layers as KL
    import tensorflow as tf

    N = (lambda n: f"{name_prefix}{n}") if name_prefix else (lambda n: n)

    # ✅ Проверяем нужно ли ограничение ROI
    max_rois = None
    if config is not None:
        use_batching = config.HEAD_CONV_CHANNEL < config.IMAGE_SHAPE[0]
        if use_batching:
            max_rois = getattr(config, 'HEAD_MAX_ROIS', 500)

    # ✅ Берём первые ROI (уже отсортированы по score)
    if max_rois is not None:
        def limit_rois(roi_tensor):
            num_rois = tf.shape(roi_tensor)[1]
            actual_limit = tf.minimum(num_rois, max_rois)
            return roi_tensor[:, :actual_limit, :]

        rois = KL.Lambda(limit_rois, name=N("limit_rois_mask"))(rois)

    # ========== ОРИГИНАЛЬНЫЙ КОД ==========
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name=N("roi_align_mask"))(
        [rois, image_meta] + feature_maps
    )

    x = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
        name=N("mrcnn_mask_conv1")
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn1'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
        name=N("mrcnn_mask_conv2")
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn2'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    res = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
        name=N("mrcnn_mask_conv3")
    )(x)
    res = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn3'))(res, training=train_bn)
    res = KL.Activation('relu')(res)

    x_dil = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same", dilation_rate=(2, 2, 2)),
        name=N("mrcnn_mask_conv3b")
    )(res)
    x_dil = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn3b'))(x_dil, training=train_bn)
    x_dil = KL.Activation('relu')(x_dil)

    x = KL.Add(name=N("mrcnn_mask_res3"))([res, x_dil])

    x = KL.TimeDistributed(
        KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
        name=N("mrcnn_mask_conv4")
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn4'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(
        KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"),
        name=N("mrcnn_mask_deconv")
    )(x)
    x = KL.TimeDistributed(
        KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"),
        name=N("mrcnn_mask")
    )(x)

    return x

def refine_detections_graph(rois, probs, deltas, image_meta,
                            detection_min_confidence,
                            detection_nms_threshold,
                            bbox_std_dev=None,
                            detection_max_instances=None):
    """ИСПРАВЛЕННАЯ версия"""

    from core import utils as U

    if bbox_std_dev is None:
        bbox_std_dev = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    bbox_std_dev = tf.cast(bbox_std_dev, tf.float32)

    min_conf = float(detection_min_confidence)
    nms_thr = float(detection_nms_threshold)
    max_inst = int(detection_max_instances) if detection_max_instances is not None else 200

    # Parse image meta
    meta = tf.cond(tf.equal(tf.rank(image_meta), 1),
                   lambda: tf.expand_dims(image_meta, 0),
                   lambda: image_meta)
    parsed = parse_image_meta_graph(meta)
    image_shape = tf.cast(parsed['image_shape'][0], tf.float32)

    # Используем вероятности класса neuron (индекс 1)
    fg_probs = probs[:, 1]
    class_ids_all = tf.ones_like(fg_probs, dtype=tf.int32)
    class_scores_all = fg_probs

    # Фильтр по confidence
    keep_conf_ix = tf.where(class_scores_all >= min_conf)[:, 0]
    rois_conf = tf.gather(rois, keep_conf_ix)
    scores_conf = tf.gather(class_scores_all, keep_conf_ix)
    class_ids_conf = tf.gather(class_ids_all, keep_conf_ix)
    deltas_conf = tf.gather(deltas, keep_conf_ix)

    pos_ix = tf.where(class_ids_conf > 0)[:, 0]
    Kp = tf.shape(pos_ix)[0]

    def _empty():
        return tf.zeros([max_inst, 8], dtype=tf.float32)

    def _non_empty():
        rois_sel = tf.gather(rois_conf, pos_ix)
        scores_sel = tf.gather(scores_conf, pos_ix)
        class_ids_sel = tf.gather(class_ids_conf, pos_ix)
        deltas_sel_all = tf.gather(deltas_conf, pos_ix)

        Kp2 = tf.shape(deltas_sel_all)[0]
        gather_idx = tf.stack([tf.range(Kp2, dtype=tf.int32), class_ids_sel], axis=1)
        deltas_sel = tf.gather_nd(deltas_sel_all, gather_idx)

        # Применяем дельты
        rois_px = U.denorm_boxes_3d_graph(rois_sel, image_shape)
        boxes_px = U.apply_box_deltas_3d_graph(rois_px, deltas_sel, bbox_std_dev)

        # Клип
        H, W, D = image_shape[0], image_shape[1], image_shape[2]
        y1 = tf.clip_by_value(boxes_px[:, 0], 0.0, H)
        x1 = tf.clip_by_value(boxes_px[:, 1], 0.0, W)
        z1 = tf.clip_by_value(boxes_px[:, 2], 0.0, D)
        y2 = tf.clip_by_value(boxes_px[:, 3], 0.0, H)
        x2 = tf.clip_by_value(boxes_px[:, 4], 0.0, W)
        z2 = tf.clip_by_value(boxes_px[:, 5], 0.0, D)
        boxes_px = tf.stack([y1, x1, z1, y2, x2, z2], axis=1)

        # ✅ ИСПРАВЛЕНО: минимальные размеры в ПИКСЕЛЯХ
        min_h = 1.0
        min_w = 1.0
        min_d = 0.5

        hh = boxes_px[:, 3] - boxes_px[:, 0]
        ww = boxes_px[:, 4] - boxes_px[:, 1]
        dd = boxes_px[:, 5] - boxes_px[:, 2]

        ok_ix = tf.where((hh >= min_h) & (ww >= min_w) & (dd >= min_d))[:, 0]
        boxes_px2 = tf.gather(boxes_px, ok_ix)
        scores_sel2 = tf.gather(scores_sel, ok_ix)
        class_ids_2 = tf.gather(class_ids_sel, ok_ix)

        # NMS
        sel = tf.image.non_max_suppression(
            boxes=tf.stack([boxes_px2[:, 1], boxes_px2[:, 0], boxes_px2[:, 4], boxes_px2[:, 3]], axis=1),
            scores=scores_sel2,
            max_output_size=max_inst,
            iou_threshold=nms_thr
        )
        final_b = tf.gather(boxes_px2, sel)
        final_s = tf.gather(scores_sel2, sel)
        final_c = tf.ones_like(final_s, dtype=tf.float32)

        # Top-K
        k = tf.minimum(tf.shape(final_s)[0], tf.constant(max_inst, tf.int32))
        order = tf.nn.top_k(final_s, k=k).indices
        final_b = tf.gather(final_b, order)
        final_s = tf.gather(final_s, order)
        final_c = tf.gather(final_c, order)

        # ✅ Назад в normalized [0,1]
        final_b_nm = U.norm_boxes_3d_graph(final_b, image_shape)

        final_c_id = tf.expand_dims(final_c, 1)
        final_s = tf.expand_dims(final_s, 1)
        det_k = tf.concat([final_b_nm, final_c_id, final_s], axis=1)

        pad = tf.maximum(0, tf.constant(max_inst, tf.int32) - tf.shape(det_k)[0])
        det = tf.pad(det_k, [[0, pad], [0, 0]])
        return det

    return tf.cond(tf.equal(Kp, 0), _empty, _non_empty)


class DetectionLayer(KE.Layer):
    """
    Converts class probabilities and bbox deltas for each ROI to final detections.
    Output: [B, DETECTION_MAX_INSTANCES, (y1,x1,z1,y2,x2,z2, class_id, score)] in normalized coords.
    """

    def __init__(self, bbox_std_dev,
                 detection_min_confidence,
                 detection_max_instances,
                 detection_nms_threshold,
                 images_per_gpu,
                 *args,  # ← поглощаем лишний batch_size
                 **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        # ✅ УБРАЛИ config из параметров - его нет в вызове!
        self.bbox_std_dev = bbox_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = int(detection_max_instances)  # ✅ INT!
        self.detection_nms_threshold = detection_nms_threshold
        self.images_per_gpu = images_per_gpu

    def call(self, inputs):
        rois = inputs[0]  # [B, R, 6] (normalized)
        mrcnn_class = inputs[1]  # [B, R, C]
        mrcnn_bbox = inputs[2]  # [B, R, C, 6]
        image_meta = inputs[3]  # [B, META]

        detections = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, image_meta],
            lambda x_rois, x_probs, x_deltas, x_meta: refine_detections_graph(
                x_rois,
                x_probs,
                x_deltas,
                x_meta,
                self.detection_min_confidence,
                self.detection_nms_threshold,
                self.bbox_std_dev,
                self.detection_max_instances
            ),
            self.images_per_gpu
        )

        # Динамический батч
        B = tf.shape(rois)[0]
        detections = tf.reshape(detections, [B, self.detection_max_instances, 8])
        return detections

    def compute_output_shape(self, input_shape):
        return (None, self.detection_max_instances, 8)



############################################################
#  Loss Functions
############################################################

def smooth_l1(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less = K.cast(K.less(diff, 1.0), "float32")
    return less * 0.5 * diff * diff + (1.0 - less) * (diff - 0.5)


def rpn_class_loss_graph(rpn_match, rpn_class_logits, alpha=0.90, gamma=1.5):
    """RPN loss с мягким Focal + α-взвешиванием позитива."""
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.not_equal(rpn_match, 0))

    def _no_anchors():
        return tf.constant(0.0)

    def _compute():
        anchor_class = tf.gather_nd(rpn_match, indices)                # [-1, +1]
        logits = tf.gather_nd(rpn_class_logits, indices)               # [N,2]
        labels = tf.cast(tf.equal(anchor_class, 1), tf.int32)          # [N]

        # CE по метке (bg=0 / fg=1)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Мягкий focal: (1 - p_t)^gamma
        probs = tf.nn.softmax(logits)
        idx = tf.stack([tf.range(tf.shape(probs)[0], dtype=tf.int64),
                        tf.cast(labels, tf.int64)], axis=1)
        p_t = tf.gather_nd(probs, idx)
        focal_w = tf.pow(1.0 - p_t, gamma)
        ce = focal_w * ce

        # α-весим позитивные примеры
        alpha_t = tf.where(tf.equal(labels, 1), alpha, 1.0 - alpha)
        loss = K.mean(alpha_t * ce)

        # Диагностика (оставляю твои принты)
        n_pos = tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.int32))
        n_neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.int32))
        ce_mean = tf.reduce_mean(ce)
        tf.print("[RPN_CLASS] n_pos/n_neg/ce_mean/loss:",
                 "[", n_pos, "][", n_neg, "][", ce_mean, "][", loss, "]")
        return loss

    return tf.cond(tf.size(indices) > 0, _compute, _no_anchors)



def rpn_bbox_loss_graph(images_per_gpu, target_bbox, rpn_match, rpn_bbox):
    """Стабилизированный bbox loss для малой глубины"""
    rpn_match = K.squeeze(rpn_match, -1)
    pos_indices = tf.where(K.equal(rpn_match, 1))

    def no_pos():
        return tf.constant(0.0, dtype=tf.float32)

    def compute_loss():
        pred_bbox = tf.gather_nd(rpn_bbox, pos_indices)

        # Более жесткое ограничение для стабильности
        pred_bbox = tf.clip_by_value(pred_bbox, -5.0, 5.0)

        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        gt_boxes = batch_pack_graph(target_bbox, batch_counts, images_per_gpu)

        # Ограничиваем разницу для стабильности
        diff = tf.clip_by_value(gt_boxes - pred_bbox, -2.0, 2.0)

        # Huber loss с меньшим порогом для Z координат
        abs_diff = tf.abs(diff)

        # Разные пороги для XY и Z
        xy_mask = tf.constant([1., 1., 0., 1., 1., 0.], dtype=tf.float32)
        z_mask = tf.constant([0., 0., 1., 0., 0., 1.], dtype=tf.float32)

        # Huber с порогом 1.0 для XY, 0.5 для Z
        huber_xy = tf.where(
            abs_diff < 1.0,
            0.5 * tf.square(diff),
            abs_diff - 0.5
        ) * xy_mask

        huber_z = tf.where(
            abs_diff < 0.5,
            0.5 * tf.square(diff),
            0.5 * abs_diff - 0.25  # Меньший вес для Z
        ) * z_mask

        huber_loss = huber_xy + huber_z

        return K.mean(huber_loss)

    return tf.cond(tf.equal(tf.size(pos_indices), 0), no_pos, compute_loss)


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """
    FOCAL LOSS с диагностикой качества ROI.
    - универсальный подсчёт fg_prob (по истинному классу, не предполагает C=2)
    - доп. метрики: top1_acc по позитивам и по фону
    - ✅ PENALTY за уверенные false positives (фон предсказан как объект)
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K

    target_class_ids = tf.cast(target_class_ids, tf.int32)
    B = tf.shape(pred_class_logits)[0]
    T = tf.shape(pred_class_logits)[1]
    C = tf.shape(pred_class_logits)[2]

    # Разрешаем фон всегда (col 0 = 1), остальное как в active_class_ids
    active_class_ids = tf.concat(
        [tf.ones_like(active_class_ids[..., :1]), active_class_ids[..., 1:]],
        axis=-1
    )

    logits = tf.clip_by_value(pred_class_logits, -10.0, 10.0)
    logits_flat = tf.reshape(logits, [B * T, C])
    target_flat = tf.reshape(target_class_ids, [B * T])

    # маска активных классов на уровне ROI
    active_repeated = tf.tile(active_class_ids, [T, 1])
    row_idx = tf.range(B * T, dtype=tf.int32)
    gather_idx = tf.stack([row_idx, tf.cast(target_flat, tf.int32)], axis=1)
    true_active = tf.gather_nd(active_repeated, gather_idx)  # [B*T]

    # Focal
    probs = tf.nn.softmax(logits, axis=-1)
    probs_flat = tf.reshape(probs, [B * T, C])
    target_onehot = tf.one_hot(target_flat, C, dtype=tf.float32)
    pt = tf.reduce_sum(probs_flat * target_onehot, axis=-1)  # p(true class)
    pt = tf.clip_by_value(pt, K.epsilon(), 1.0 - K.epsilon())

    gamma = tf.cast(getattr(__import__('builtins'), 'CLASS_FOCAL_GAMMA', 3.0), tf.float32)
    alpha = tf.cast(getattr(__import__('builtins'), 'CLASS_FOCAL_ALPHA', 0.85), tf.float32)

    focal_weight = tf.pow(1.0 - pt, gamma)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_flat, logits=logits_flat)
    focal_loss = focal_weight * ce

    # alpha-balancing по fg/bg
    is_fg = tf.cast(target_flat > 0, tf.float32)
    is_bg = 1.0 - is_fg
    class_weights = is_fg * alpha + is_bg * (1.0 - alpha)

    # ✅ PENALTY ЗА УВЕРЕННЫЕ FALSE POSITIVES
    # Максимальная вероятность среди НЕ-фоновых классов
    fg_probs = probs_flat[:, 1:]  # [BT, C-1] - все классы кроме фона
    max_fg_prob = tf.reduce_max(fg_probs, axis=-1)  # [BT] - макс вероятность объекта

    # Штраф когда:
    # 1. target = 0 (фон)
    # 2. Модель уверенно предсказывает объект (max_fg_prob > 0.5)
    fp_confidence_threshold = 0.5  # порог уверенности для penalty
    fp_penalty_multiplier = 2.0  # коэффициент усиления loss

    is_confident_fp = tf.cast(
        (target_flat == 0) & (max_fg_prob > fp_confidence_threshold),
        tf.float32
    )

    # Применяем penalty: умножаем loss на multiplier для confident FP
    fp_penalty = 1.0 + is_confident_fp * (fp_penalty_multiplier - 1.0)
    focal_loss = focal_loss * fp_penalty

    # Применяем веса
    focal_loss = focal_loss * class_weights * tf.cast(true_active, focal_loss.dtype)

    # --- Диагностика ---
    pos_ix = tf.where(is_fg > 0.5)[:, 0]
    neg_ix = tf.where(is_bg > 0.5)[:, 0]

    pos_count = tf.size(pos_ix)
    neg_count = tf.size(neg_ix)

    # ✅ Добавляем диагностику confident FP
    confident_fp_count = tf.reduce_sum(is_confident_fp)
    mean_max_fg_prob_on_bg = tf.cond(
        neg_count > 0,
        lambda: tf.reduce_mean(tf.gather(max_fg_prob, neg_ix)),
        lambda: tf.constant(0.0, tf.float32)
    )

    # средняя вероятность истинного класса для позитивов
    def _mean_fg_prob():
        return tf.reduce_mean(tf.gather(pt, pos_ix))

    mean_fg_prob = tf.cond(pos_count > 0, _mean_fg_prob, lambda: tf.constant(0.0, tf.float32))

    # top-1 accuracy по позитивам/фону
    pred_labels = tf.argmax(logits_flat, axis=-1, output_type=tf.int32)

    def _pos_acc():
        return tf.reduce_mean(tf.cast(tf.equal(tf.gather(pred_labels, pos_ix),
                                               tf.gather(target_flat, pos_ix)), tf.float32))

    pos_top1_acc = tf.cond(pos_count > 0, _pos_acc, lambda: tf.constant(0.0, tf.float32))

    def _bg_acc():
        return tf.reduce_mean(tf.cast(tf.equal(tf.gather(pred_labels, neg_ix), 0), tf.float32))

    bg_top1_acc = tf.cond(neg_count > 0, _bg_acc, lambda: tf.constant(0.0, tf.float32))

    loss_mean = tf.reduce_mean(focal_loss)
    loss_fg = tf.cond(pos_count > 0,
                      lambda: tf.reduce_mean(tf.gather(focal_loss, pos_ix)),
                      lambda: tf.constant(0.0, focal_loss.dtype))
    loss_bg = tf.cond(neg_count > 0,
                      lambda: tf.reduce_mean(tf.gather(focal_loss, neg_ix)),
                      lambda: tf.constant(0.0, focal_loss.dtype))

    should_print = tf.random.uniform([]) < 0.02
    focal_with_debug = tf.cond(
        should_print,
        lambda: tf.compat.v1.Print(
            focal_loss,
            [pos_count, neg_count, mean_fg_prob, pos_top1_acc, bg_top1_acc,
             confident_fp_count, mean_max_fg_prob_on_bg, loss_mean, loss_fg, loss_bg],
            message="[CLASS_LOSS] pos/neg/fg_prob/pos_acc/bg_acc/conf_fp/max_fg_on_bg/loss_mean/loss_fg/loss_bg: "
        ),
        lambda: focal_loss
    )

    # Нормировка (по суммарному весу)
    weight_sum = tf.reduce_sum(class_weights * tf.cast(true_active, focal_loss.dtype))
    denom = tf.maximum(weight_sum, K.epsilon())
    return tf.reduce_sum(focal_with_debug) / denom


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox, config=None):
    """
    УЛУЧШЕННАЯ версия: Huber loss + adaptive clipping + детальная диагностика
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K

    target_class_ids = tf.cast(target_class_ids, tf.int32)
    B = tf.shape(target_bbox)[0]
    T = tf.shape(target_bbox)[1]

    y_true = tf.reshape(target_bbox, [B * T, 6])
    cls = tf.reshape(target_class_ids, [B * T])
    y_pred = tf.reshape(pred_bbox, [B * T, -1, 6])

    pos_ix = tf.where(cls > 0)[:, 0]

    def _no_pos():
        return tf.cast(0.0, K.floatx())

    def _with_pos():
        yt = tf.gather(y_true, pos_ix)  # target (normalized)
        pc = tf.gather(cls, pos_ix)  # class ids
        yp = tf.gather(y_pred, pos_ix)  # predictions [P, C, 6]

        N = tf.shape(pos_ix)[0]
        idx = tf.stack([tf.range(N, dtype=tf.int32), tf.cast(pc, tf.int32)], axis=1)
        yp_cls = tf.gather_nd(yp, idx)  # [P, 6]

        # ✅ SOFT CLIPPING: убираем экстремальные выбросы
        # Используем tanh для мягкого ограничения вместо жёсткого clip
        yp_cls = 3.0 * tf.tanh(yp_cls / 3.0)  # ограничение в [-3, 3]

        # ✅ HUBER LOSS: более устойчив к выбросам чем Smooth L1
        delta = 1.0  # Huber threshold (настройте по вкусу: 0.5-2.0)
        abs_diff = K.abs(yt - yp_cls)

        # Huber: квадратичный для малых ошибок, линейный для больших
        huber = tf.where(
            abs_diff <= delta,
            0.5 * K.square(abs_diff),  # L2 для малых
            delta * (abs_diff - 0.5 * delta)  # L1 для больших
        )

        per_coord = huber  # [P, 6]
        per_roi = K.mean(per_coord, axis=-1)  # [P]
        final_loss = K.mean(per_roi)

        # ✅ ДЕТАЛЬНАЯ ДИАГНОСТИКА
        coord_errors = K.abs(yt - yp_cls)
        mean_coord_error = K.mean(coord_errors)
        max_coord_error = K.max(coord_errors)

        # Процент больших ошибок (>2.0 в норм. координатах = >512px)
        large_errors = tf.cast(tf.reduce_sum(tf.cast(coord_errors > 2.0, tf.float32)), tf.float32)
        pct_large = large_errors / tf.cast(tf.size(coord_errors), tf.float32)

        should_print = tf.random.uniform([]) < 0.05
        final_with_debug = tf.cond(
            should_print,
            lambda: tf.compat.v1.Print(final_loss, [
                N, mean_coord_error, max_coord_error, pct_large, final_loss
            ], message="[BBOX_LOSS] N/mean_err/max_err/pct_large/loss: "),
            lambda: final_loss
        )
        return final_with_debug

    return tf.cond(tf.size(pos_ix) > 0, _with_pos, _no_pos)



def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    B = tf.shape(target_masks)[0]; T = tf.shape(target_masks)[1]
    mH = tf.shape(pred_masks)[2]; mW = tf.shape(pred_masks)[3]; mD = tf.shape(pred_masks)[4]
    V  = mH * mW * mD; C = tf.shape(pred_masks)[-1]

    y_true = tf.reshape(target_masks, [B*T, V])             # [BT, V]
    y_pred = tf.reshape(pred_masks, [B*T, V, C])            # [BT, V, C]
    cls1d  = tf.reshape(tf.cast(target_class_ids, tf.int64), [B*T])
    pos_ix = tf.reshape(tf.where(cls1d > 0), [-1])

    def _no_pos(): return K.cast(0.0, K.floatx())

    def _with_pos():
        Npos = tf.shape(pos_ix)[0]
        yt = tf.gather(y_true, pos_ix)                      # [P, V]
        yp = tf.gather(y_pred, pos_ix)                      # [P, V, C]
        pc = tf.gather(cls1d, pos_ix)                       # [P]

        yp_t = tf.transpose(yp, [0, 2, 1])                  # [P, C, V]
        idx  = tf.stack([tf.range(tf.cast(Npos, tf.int64), dtype=tf.int64),
                         tf.cast(pc, tf.int64)], axis=1)
        yp_cls  = tf.gather_nd(yp_t, idx)                   # [P, V]
        yp_prob = K.clip(yp_cls, K.epsilon(), 1.0 - K.epsilon())

        # --- фильтрация пустых таргетов (иногда crop дал нули)
        valid = tf.reduce_sum(yt, axis=-1) > 0              # [P]
        yt = tf.boolean_mask(yt, valid)
        yp_prob = tf.boolean_mask(yp_prob, valid)
        P_valid = tf.shape(yt)[0]
        def _no_valid(): return K.cast(0.0, K.floatx())

        def _loss_valid():
            # BCE (mean!), Dice (mean)
            bce_loss = K.mean(K.binary_crossentropy(yt, yp_prob))
            smooth = 1.0
            inter  = K.sum(yt*yp_prob, axis=-1)
            union  = K.sum(yt, axis=-1) + K.sum(yp_prob, axis=-1)
            dice   = (2.0*inter + smooth)/(union + smooth)
            dice_loss = 1.0 - K.mean(dice)
            final = 0.3*bce_loss + 0.7*dice_loss

            # Диагностика: fg-доли, Dice, COM-ошибка (нормированная по осям)
            fg_pred = K.mean(yp_prob); fg_true = K.mean(yt)

            def _com3d(mask_flat):                           # [Q, V] -> [Q,3] в норм. координатах
                Q = tf.shape(mask_flat)[0]
                mask3d = tf.reshape(mask_flat, [Q, mH, mW, mD])
                z = tf.linspace(0.0, 1.0, tf.cast(mD, tf.int32))
                y = tf.linspace(0.0, 1.0, tf.cast(mH, tf.int32))
                x = tf.linspace(0.0, 1.0, tf.cast(mW, tf.int32))
                zz, yy, xx = tf.meshgrid(z, y, x, indexing='ij')     # [D,H,W]
                zz = tf.cast(zz, K.floatx()); yy = tf.cast(yy, K.floatx()); xx = tf.cast(xx, K.floatx())

                mass = K.sum(mask3d, axis=[1,2,3], keepdims=True) + K.epsilon()   # [Q,1,1,1]
                com_z = K.sum(mask3d*zz[None,...], axis=[1,2,3]) / K.flatten(mass)  # [Q]
                com_y = K.sum(mask3d*yy[None,...], axis=[1,2,3]) / K.flatten(mass)
                com_x = K.sum(mask3d*xx[None,...], axis=[1,2,3]) / K.flatten(mass)
                return tf.stack([com_y, com_x, com_z], axis=-1)       # [Q,3]  (y,x,z)

            com_p = _com3d(yp_prob); com_t = _com3d(yt)
            com_err = K.mean(K.sqrt(K.sum(K.square(com_p - com_t), axis=-1)))     # L2 в [0..√3]

            # печать ~5% раз
            should_print = tf.random.uniform([]) < 0.05
            final = tf.cond(
                should_print,
                lambda: tf.compat.v1.Print(final, [
                    P_valid, fg_pred, fg_true, K.mean(dice), com_err,
                    bce_loss, dice_loss, final
                ], message="[MASK_LOSS] P/fg_pred/fg_true/dice/com_err_norm/bce/dice_l/total: "),
                lambda: final
            )
            return final

        return tf.cond(P_valid > 0, _loss_valid, _no_valid)

    return tf.cond(tf.size(pos_ix) > 0, _with_pos, _no_pos)










############################################################
#  Custom Callbacks
############################################################
class BestAndLatestCheckpoint(keras.callbacks.Callback):
    """Хранит два файла в save_path:
       - latest.h5 — перезаписывается каждую эпоху
       - best.h5   — переписывается при улучшении метрики
       Дополнительно пишет:
       - latest_head.h5 / best_head.h5 — только слои mrcnn_* (голова), + head_meta
    """
    def __init__(self, save_path="", mode="GENERIC"):
        super(BestAndLatestCheckpoint, self).__init__()
        self.save_path = save_path
        self.mode = (mode or "GENERIC").upper()
        self.best = None
        self.monitor_key = None
        self._cmp = None
        self._eps = 1e-6

    def _dump_head_only(self, path):
        try:
            import os, h5py, numpy as np
            prefer = "best" if ("best" in os.path.basename(path)) else "latest"
            candidates = [
                os.path.join(self.save_path, f"{prefer}.h5"),
                os.path.join(self.save_path, "latest.h5"),
                os.path.join(self.save_path, "best.h5"),
            ]
            src = next((p for p in candidates if os.path.exists(p)), None)
            if src is None:
                print(f"[Checkpoint] head-only save skipped: no source full H5 found near {self.save_path}")
                return
            with h5py.File(src, "r") as fin, h5py.File(path, "w") as fout:
                src_root = fin["model_weights"] if "model_weights" in fin.keys() else fin
                dst_root = fout.create_group("model_weights")
                for key in list(src_root.keys()):
                    if not key.startswith("mrcnn_"):
                        continue
                    obj = src_root.get(key, None)
                    if obj is None:
                        continue
                    # рекурсивное копирование групп
                    src_root.file.copy(obj, dst_root, name=key)

                # Минимальные числовые метаданные
                try:
                    import numpy as np
                    meta = fout.create_group("head_meta")
                    k1 = self.model.get_layer("mrcnn_class_conv1").get_weights()[0].shape  # (k,k,k,in,256)
                    k2 = self.model.get_layer("mrcnn_class_conv2").get_weights()[0].shape  # (1,1,1,256,256)
                    meta.attrs["pool_kernel"] = np.asarray(k1[:3], dtype=np.int32)
                    meta.attrs["fc_channels"] = np.asarray([k2[-1]], dtype=np.int32)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Checkpoint] head-only save failed: {e}")

    def _choose_monitor(self, logs):
        # Специальные режимы
        if self.mode == 'RPN':
            self.monitor_key = ("rpn_train_detection_score", "rpn_test_detection_score")
            self._cmp = lambda cur, best: (self.best is None) or (cur > best + self._eps)
            print(f"[Checkpoint] monitoring: sum(RPN detection scores) (maximize)")
            return
        if self.mode == 'HEAD':
            # Если есть head_test_total_loss — минимизируем его
            if logs and ("val_loss" in logs):
                self.monitor_key = "val_loss"
                self._cmp = lambda cur, best: (self.best is None) or (cur < best - self._eps)
                print(f"[Checkpoint] monitoring: val_loss (minimize)")
                return
            # Иначе суммируем доступные head_test_*_mean
            if logs:
                keys = [k for k in logs.keys() if k.startswith("head_test_") and k.endswith("_mean")]
                if keys:
                    self.monitor_key = tuple(keys)
                    self._cmp = lambda cur, best: (self.best is None) or (cur < best - self._eps)
                    print(f"[Checkpoint] monitoring: sum(head_test_*_mean) (minimize)")
                    return
        # Общий случай — loss
        if logs and "loss" in logs:
            self.monitor_key = "loss"
            self._cmp = lambda cur, best: (self.best is None) or (cur < best - self._eps)
            print(f"[Checkpoint] monitoring: loss (minimize)")
            return
        # fallback
        self.monitor_key = None
        self._cmp = None
        print("[Checkpoint] monitoring: disabled (no metric found)")

    def _current_metric(self, logs):
        if self.monitor_key is None:
            return None
        if isinstance(self.monitor_key, tuple):
            try:
                return float(sum(float(logs[k]) for k in self.monitor_key if k in logs))
            except Exception:
                return None
        try:
            return float(logs[self.monitor_key])
        except Exception:
            return None

    def on_epoch_end(self, epoch, logs=None):
        # latest всегда
        latest_path = f"{self.save_path}latest.h5"
        self.model.save_weights(latest_path)
        self._dump_head_only(f"{self.save_path}latest_head.h5")

        logs = logs or {}
        if self.monitor_key is None:
            self._choose_monitor(logs)

        metric = self._current_metric(logs)
        if (metric is None) or (self._cmp is None):
            return  # нечего мониторить

        if (self.best is None) or self._cmp(metric, self.best):
            self.best = float(metric)
            best_path = f"{self.save_path}best.h5"
            self.model.save_weights(best_path)
            self._dump_head_only(f"{self.save_path}best_head.h5")
            print(f"[Checkpoint] best updated @epoch {epoch+1}: {self.monitor_key} = {self.best:.6f}")




class CheckPointCallback(keras.callbacks.Callback):
    def __init__(self, save_path, interval=1000):
        super(CheckPointCallback, self).__init__()
        self.save_path = save_path
        self.interval = interval
        self.counter = 0

    def on_batch_end(self):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.model.save_weights(f"{self.save_path}checkpoint.h5")


class TelemetryEpochLogger(keras.callbacks.Callback):
    """Сохраняет jsonl-телеметрию после каждой эпохи в WEIGHT_DIR/telemetry.jsonl.
       Ничего не считает сам — только вызывает snapshot у core.utils.Telemetry.
    """

    def __init__(self, save_dir, config):
        super().__init__()
        self.save_dir = save_dir
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        # Пишем телеметрию только если включено в конфиге
        if not getattr(self.config, "TELEMETRY", True):
            return
        # гарантируем, что папка есть
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass
        try:
            # соберём ключевые метрики из logs, если есть
            extra = {}
            if logs:
                # стандартные метрики keras
                for k in ("loss", "val_loss", "accuracy", "val_accuracy", "lr"):
                    if k in logs:
                        try:
                            extra[k] = float(logs[k])
                        except Exception:
                            pass
                # специфичные метрики модели
                for k in ("rpn_train_detection_score", "rpn_test_detection_score",
                          "rpn_train_mean_coord_error", "rpn_test_mean_coord_error",
                          "head_test_total_loss"):
                    if k in logs:
                        try:
                            extra[k] = float(logs[k])
                        except Exception:
                            pass
            # прокинем текущие конфигурации якорей внутрь снапшота
            try:
                extra["cfg_scales"] = list(getattr(self.config, "RPN_ANCHOR_SCALES", []))
                extra["cfg_ratios"] = [float(x) for x in getattr(self.config, "RPN_ANCHOR_RATIOS", [])]
            except Exception:
                pass

            from core.utils import Telemetry
            Telemetry.snapshot_and_reset(epoch + 1, self.save_dir, extra=extra)
        except Exception as e:
            print(f"[Telemetry] snapshot failed: {e}")




class RPNEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, model, config, train_dataset, test_dataset, check_boxes=False):
        super(RPNEvaluationCallback, self).__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.check_boxes = check_boxes

    def on_epoch_end(self, epoch, logs=None):
        import io, sys, re
        buf, old_stdout = io.StringIO(), sys.stdout
        try:
            sys.stdout = buf
            rpn_evaluation(
                self.model, self.config,
                ["TRAIN SUBSET", "TEST SUBSET"],
                [self.train_dataset, self.test_dataset],
                self.check_boxes
            )
        finally:
            sys.stdout = old_stdout
            text = buf.getvalue()
            print(text)  # сохраним оригинальный вывод

        logs = logs if logs is not None else {}
        subset = None
        for line in text.splitlines():
            line = line.strip()
            if line in ("TRAIN SUBSET", "TEST SUBSET"):
                subset = line.split()[0].lower()  # 'train' | 'test'
            elif line.startswith("CLASS:") and subset:
                m = re.findall(r"CLASS:\s*([0-9.]+).*\+/-\s*([0-9.]+).*BBOX:\s*([0-9.]+).*\+/-\s*([0-9.]+)", line)
                if m:
                    c_mean, c_std, b_mean, b_std = map(float, m[0])
                    logs[f"rpn_{subset}_class_mean"] = c_mean
                    logs[f"rpn_{subset}_class_std"] = c_std
                    logs[f"rpn_{subset}_bbox_mean"] = b_mean
                    logs[f"rpn_{subset}_bbox_std"] = b_std
            elif line.startswith("Mean Coordinate Error:") and subset:
                m = re.findall(r"Mean Coordinate Error:\s*([0-9.]+)\s*Detection score:\s*([0-9.]+)", line)
                if m:
                    err, det = map(float, m[0])
                    logs[f"rpn_{subset}_mean_coord_error"] = err
                    logs[f"rpn_{subset}_detection_score"] = det


class HeadTrainingMetricsCallback(keras.callbacks.Callback):
    """
    Отслеживает ключевые метрики обучения HEAD.

    РАБОТАЕТ ДЛЯ:
    - training (HEAD-only): использует HeadGenerator
    - training_head_e2e: использует RPNGenerator + full model
    """

    def __init__(self, model, config, val_dataset, check_every=5):
        super().__init__()
        self.val_model = model
        self.config = config
        self.val_dataset = val_dataset
        self.check_every = check_every

        # Определяем режим по типу модели
        self.is_e2e = (config.MODE == "training_head_e2e")

    def on_epoch_end(self, epoch, logs=None):
        # Проверяем только каждые N эпох
        if (epoch + 1) % self.check_every != 0:
            return

        logs = logs or {}

        try:
            if self.is_e2e:
                self._compute_e2e_metrics(logs)
            else:
                self._compute_head_metrics(logs)

        except Exception as e:
            print(f"[Training Metrics] Calculation failed: {e}")
            import traceback
            traceback.print_exc()

    def _compute_e2e_metrics(self, logs):
        """Метрики для E2E режима (raw images → predictions)"""
        from core.data_generators import RPNGenerator

        # ✅ Используем RPNGenerator в targeting режиме (как при обучении)
        original_mode = self.config.MODE
        self.config.MODE = "targeting"

        gen = RPNGenerator(
            dataset=self.val_dataset,
            config=self.config,
            shuffle=False
        )

        self.config.MODE = original_mode

        # Берём 3 батча
        n_batches = min(3, len(gen))

        all_class_losses = []
        all_bbox_losses = []
        all_mask_losses = []

        print(f"\n[E2E Metrics] Evaluating on {n_batches} validation batches...")

        for i in range(n_batches):
            batch_data = gen[i]

            # RPNGenerator возвращает: (inputs, outputs)
            # inputs: [image, image_meta, gt_class_ids, gt_boxes, gt_masks]
            # outputs: dummy (None)

            inputs = batch_data[0]

            # Предсказания модели
            # E2E модель возвращает: [class_logits, probs, bbox, mask, class_loss, bbox_loss, mask_loss]
            outputs = self.val_model.predict(inputs, verbose=0)

            # Последние 3 выхода - это losses
            if len(outputs) >= 3:
                # Пытаемся извлечь loss значения
                try:
                    class_loss = float(K.get_value(outputs[-3]))
                    bbox_loss = float(K.get_value(outputs[-2]))
                    mask_loss = float(K.get_value(outputs[-1]))

                    all_class_losses.append(class_loss)
                    all_bbox_losses.append(bbox_loss)
                    all_mask_losses.append(mask_loss)
                except:
                    # Если не скаляры, берём mean
                    class_loss = float(np.mean(outputs[-3]))
                    bbox_loss = float(np.mean(outputs[-2]))
                    mask_loss = float(np.mean(outputs[-1]))

                    all_class_losses.append(class_loss)
                    all_bbox_losses.append(bbox_loss)
                    all_mask_losses.append(mask_loss)

        # Логируем метрики
        if all_class_losses:
            mean_class = float(np.mean(all_class_losses))
            mean_bbox = float(np.mean(all_bbox_losses))
            mean_mask = float(np.mean(all_mask_losses))
            total_loss = mean_class + mean_bbox + mean_mask

            logs['val_class_loss'] = mean_class
            logs['val_bbox_loss'] = mean_bbox
            logs['val_mask_loss'] = mean_mask
            logs['val_total_loss'] = total_loss

            print(f"[E2E Metrics] Validation losses:")
            print(f"  Class: {mean_class:.4f}")
            print(f"  BBox:  {mean_bbox:.4f}")
            print(f"  Mask:  {mean_mask:.4f}")
            print(f"  Total: {total_loss:.4f}")

            # Проверка на коллапс
            if mean_class < 0.01:
                print(f"  ⚠️  WARNING: Very low class loss - may indicate collapse!")
            if mean_bbox < 0.001:
                print(f"  ⚠️  WARNING: Very low bbox loss - may indicate collapse!")

    def _compute_head_metrics(self, logs):
        """Метрики для HEAD-only режима (оригинальная логика)"""
        from core.data_generators import HeadGenerator

        gen = HeadGenerator(self.val_dataset, self.config, shuffle=False, training=False)

        # Берём 3 батча для быстрой проверки
        n_batches = min(3, len(gen))

        all_accuracies = []
        all_ious = []
        all_fg_probs = []

        for i in range(n_batches):
            inputs, _ = gen[i]

            # inputs: [rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask]
            target_class_ids = inputs[3][0]  # [T]
            target_bbox = inputs[4][0]  # [T, 6]

            # Предсказания
            outputs = self.val_model.predict(inputs, verbose=0)

            # Ищем выходы по именам
            output_names = [t.name.split("/")[0].split(":")[0] for t in self.val_model.outputs]

            # mrcnn_class (softmax probabilities)
            cls_idx = None
            for idx, name in enumerate(output_names):
                if "mrcnn_class" in name and "logits" not in name:
                    cls_idx = idx
                    break

            if cls_idx is not None:
                probs = outputs[cls_idx][0]  # [T, C]
                pred_classes = np.argmax(probs, axis=1)

                # Считаем только на валидных ROI (не padding)
                valid_mask = target_class_ids >= 0
                if np.any(valid_mask):
                    acc = np.mean(pred_classes[valid_mask] == target_class_ids[valid_mask])
                    all_accuracies.append(float(acc))

                    # Средняя вероятность fg класса
                    fg_probs = probs[valid_mask, 1] if probs.shape[1] > 1 else np.zeros(np.sum(valid_mask))
                    all_fg_probs.extend(fg_probs.tolist())

            # mrcnn_bbox (deltas)
            bbox_idx = None
            for idx, name in enumerate(output_names):
                if "mrcnn_bbox" in name:
                    bbox_idx = idx
                    break

            if bbox_idx is not None:
                pred_bbox = outputs[bbox_idx][0]  # [T, C, 6]

                # Берём дельты для правильного класса
                valid_pos = (target_class_ids > 0)
                if np.any(valid_pos):
                    # Для простоты считаем L1 расстояние между pred и target bbox
                    pos_target_cls = target_class_ids[valid_pos]
                    pos_target_bbox = target_bbox[valid_pos]

                    # Извлекаем дельты для правильных классов
                    pos_pred_bbox = pred_bbox[valid_pos]
                    pos_pred_bbox_cls = pos_pred_bbox[np.arange(len(pos_target_cls)), pos_target_cls]

                    # L1 ошибка (чем меньше, тем лучше bbox regression)
                    l1_error = np.mean(np.abs(pos_pred_bbox_cls - pos_target_bbox))
                    all_ious.append(float(l1_error))

        # Логируем метрики
        if all_accuracies:
            mean_acc = float(np.mean(all_accuracies))
            logs['val_class_accuracy'] = mean_acc
            print(f"\n[Training Metrics] Classification Accuracy: {mean_acc:.3f}")

        if all_ious:
            mean_l1 = float(np.mean(all_ious))
            logs['val_bbox_l1_error'] = mean_l1
            print(f"[Training Metrics] BBox L1 Error: {mean_l1:.4f} (lower is better)")

        if all_fg_probs:
            fg_arr = np.array(all_fg_probs)
            mean_fg = float(np.mean(fg_arr))
            std_fg = float(np.std(fg_arr))
            logs['val_mean_fg_prob'] = mean_fg
            print(f"[Training Metrics] FG Probability: mean={mean_fg:.3f}, std={std_fg:.3f}")

            # Проверка на коллапс
            if std_fg < 0.05:
                print(f"  ⚠️ WARNING: Low variance in predictions - model may be collapsing!")


class AutoTuneRPNCallback(keras.callbacks.Callback):
    """
    КОНСЕРВАТИВНЫЙ автотюнер для RPN якорей.
    Использует проверенные функции из utils для вычисления IoU.
    """

    def __init__(self, train_generator, config):
        super().__init__()
        self.gen = train_generator
        self.config = config
        self.analysis_done = False

    def on_epoch_end(self, epoch, logs=None):
        if not getattr(self.config, "AUTO_TUNE_RPN", False):
            return

        if self.analysis_done or epoch != 0:
            return

        print("\n" + "=" * 80)
        print("🔍 CONSERVATIVE DATASET ANALYSIS FOR RPN ANCHORS")
        print("=" * 80)
        print("⏱️  This will take 5-10 minutes to analyze real anchor-GT deltas...")
        print("=" * 80)

        stats = self._analyze_full_dataset()

        if stats is None:
            print("❌ Failed to analyze dataset")
            self.analysis_done = True
            return

        self._print_statistics(stats)
        recommendations = self._generate_recommendations(stats)
        self._print_recommendations(recommendations, stats)

        self.analysis_done = True
        print("=" * 80 + "\n")

    def _analyze_full_dataset(self):
        """Полный анализ датасета включая реальные дельты"""
        import numpy as np

        dataset = None
        if hasattr(self.gen, 'dataset'):
            dataset = self.gen.dataset
        elif hasattr(self, 'train_dataset'):
            dataset = self.train_dataset

        if dataset is None or not hasattr(dataset, 'image_ids'):
            return None

        print(f"\n📊 Phase 1/2: Collecting GT statistics from {len(dataset.image_ids)} images...")

        xy_sizes = []
        z_sizes = []
        z_ratios = []
        heights = []
        widths = []
        all_gt_boxes = []

        processed = 0
        for image_id in dataset.image_ids:
            try:
                boxes, class_ids, _ = dataset.load_data(image_id)

                if boxes is None or len(boxes) == 0:
                    continue

                fg_mask = class_ids > 0
                boxes = boxes[fg_mask]

                if len(boxes) == 0:
                    continue

                for box in boxes:
                    y1, x1, z1, y2, x2, z2 = box

                    h = y2 - y1
                    w = x2 - x1
                    d = z2 - z1

                    if h <= 0 or w <= 0 or d <= 0:
                        continue

                    xy_size = np.sqrt(h * w)
                    xy_sizes.append(xy_size)
                    z_sizes.append(d)
                    z_ratios.append(d / xy_size if xy_size > 0 else 0.1)
                    heights.append(h)
                    widths.append(w)
                    all_gt_boxes.append([y1, x1, z1, y2, x2, z2])

                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{len(dataset.image_ids)} images...", end='\r')

            except Exception:
                continue

        print(f"  Processed {processed}/{len(dataset.image_ids)} images - Done!")

        if len(xy_sizes) == 0:
            return None

        # Phase 2: Вычисляем реальные BBOX_STD_DEV
        print(f"\n📊 Phase 2/2: Computing real anchor→GT deltas (this takes time)...")
        bbox_std_dev = self._compute_real_bbox_std_dev(np.array(all_gt_boxes))

        return {
            'xy_sizes': np.array(xy_sizes),
            'z_sizes': np.array(z_sizes),
            'z_ratios': np.array(z_ratios),
            'heights': np.array(heights),
            'widths': np.array(widths),
            'total_objects': len(xy_sizes),
            'bbox_std_dev': bbox_std_dev,
            'all_gt_boxes': np.array(all_gt_boxes)
        }

    def _compute_real_bbox_std_dev(self, gt_boxes):
        """
        Вычисляет РЕАЛЬНЫЕ BBOX_STD_DEV из данных.
        Использует проверенную функцию compute_overlaps_3d из utils.
        """
        import numpy as np
        from core import utils

        # Получаем текущие якоря
        anchors = getattr(self.gen, 'anchors', None)
        if anchors is None:
            try:
                self.gen.rebuild_anchors()
                anchors = self.gen.anchors
            except:
                print("    ⚠️  Could not get anchors, using conservative defaults")
                return [0.15, 0.15, 0.15, 0.25, 0.25, 0.35]

        if anchors is None or len(anchors) == 0:
            print("    ⚠️  No anchors available, using conservative defaults")
            return [0.15, 0.15, 0.15, 0.25, 0.25, 0.35]

        print(f"    Computing deltas for {len(gt_boxes)} GT boxes vs {len(anchors)} anchors...")

        # DEBUG: Проверяем формат данных
        print(f"    [DEBUG] GT boxes shape: {gt_boxes.shape}")
        print(f"    [DEBUG] GT sample: {gt_boxes[0]}")
        print(f"    [DEBUG] Anchors shape: {anchors.shape}")
        print(f"    [DEBUG] Anchor sample (normalized): {anchors[0]}")

        # ========== КРИТИЧНО: ДЕНОРМАЛИЗУЕМ ЯКОРЯ! ==========
        # Якоря в [0, 1], GT в пикселях - нужно привести к одной системе!
        image_shape = self.config.IMAGE_SHAPE  # [H, W, D]

        anchors_denorm = anchors.copy()
        anchors_denorm[:, [0, 3]] *= image_shape[0]  # y1, y2
        anchors_denorm[:, [1, 4]] *= image_shape[1]  # x1, x2
        anchors_denorm[:, [2, 5]] *= image_shape[2]  # z1, z2

        print(f"    [DEBUG] Image shape: {image_shape}")
        print(f"    [DEBUG] Anchor sample (denormalized): {anchors_denorm[0]}")
        print(
            f"    [DEBUG] Anchor ranges (denorm): y=[{anchors_denorm[:, 0].min():.1f}, {anchors_denorm[:, 3].max():.1f}], "
            f"x=[{anchors_denorm[:, 1].min():.1f}, {anchors_denorm[:, 4].max():.1f}], "
            f"z=[{anchors_denorm[:, 2].min():.1f}, {anchors_denorm[:, 5].max():.1f}]")
        print(f"    [DEBUG] GT ranges: y=[{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 3].max():.1f}], "
              f"x=[{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 4].max():.1f}], "
              f"z=[{gt_boxes[:, 2].min():.1f}, {gt_boxes[:, 5].max():.1f}]")

        # Берем сэмпл для скорости
        sample_size = min(800, len(gt_boxes))
        if len(gt_boxes) > sample_size:
            indices = np.random.choice(len(gt_boxes), sample_size, replace=False)
            gt_sample = gt_boxes[indices]
        else:
            gt_sample = gt_boxes

        print(f"    Using {len(gt_sample)} GT boxes for delta computation...")

        all_deltas = []

        # Обрабатываем батчами по 50 GT
        batch_size = 50
        total_valid = 0

        for i in range(0, len(gt_sample), batch_size):
            batch = gt_sample[i:i + batch_size]

            try:
                # Используем ДЕНОРМАЛИЗОВАННЫЕ якоря!
                overlaps = utils.compute_overlaps_3d(batch, anchors_denorm)

                if i == 0:
                    print(f"    [DEBUG] Batch 0: overlaps shape={overlaps.shape}, "
                          f"max IoU={overlaps.max():.4f}, mean IoU={overlaps.mean():.6f}, "
                          f"IoU>0.1: {(overlaps > 0.1).sum()}, IoU>0.3: {(overlaps > 0.3).sum()}")

                # Для каждого GT находим лучший якорь
                best_indices = np.argmax(overlaps, axis=1)
                best_ious = np.max(overlaps, axis=1)

                # Используем разумный порог
                valid_mask = best_ious > 0.1
                n_valid = np.sum(valid_mask)

                if n_valid == 0:
                    continue

                total_valid += n_valid

                batch_gt = batch[valid_mask]
                # Используем ДЕНОРМАЛИЗОВАННЫЕ якоря для вычисления дельт
                batch_anchors = anchors_denorm[best_indices[valid_mask]]

                # Вычисляем дельты
                deltas = self._compute_deltas(batch_anchors, batch_gt)
                all_deltas.append(deltas)

            except Exception as e:
                print(f"    ⚠️  Error in batch {i // batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if (i // batch_size) % 5 == 0:
                print(
                    f"    Processed {min(i + batch_size, len(gt_sample))}/{len(gt_sample)} GT boxes... (valid: {total_valid})",
                    end='\r')

        print(f"    Processed {len(gt_sample)}/{len(gt_sample)} GT boxes - Done! Valid pairs: {total_valid}        ")

        if len(all_deltas) == 0 or total_valid < 10:
            print("    ⚠️  Too few valid deltas, using conservative defaults")
            return [0.15, 0.15, 0.15, 0.25, 0.25, 0.35]

        # Объединяем все дельты
        all_deltas = np.concatenate(all_deltas, axis=0)

        print(f"    ✓ Got {len(all_deltas)} valid anchor-GT pairs")
        print(f"    Computing robust statistics...")

        # Робастная оценка: 68-й перцентиль |delta|
        std_p68 = np.percentile(np.abs(all_deltas), 68, axis=0)

        # Дополнительная оценка через MAD
        median_deltas = np.median(all_deltas, axis=0)
        mad = np.median(np.abs(all_deltas - median_deltas), axis=0)
        std_mad = mad * 1.4826

        # Берем среднее двух оценок
        std_final = (std_p68 + std_mad) / 2.0

        # Накладываем разумные ограничения
        std_final[0:3] = np.clip(std_final[0:3], 0.10, 0.35)  # координаты
        std_final[3:6] = np.clip(std_final[3:6], 0.15, 0.45)  # размеры (log)

        print(f"    ✓ BBOX_STD_DEV computed from {len(all_deltas)} real deltas:")
        labels = ['dy', 'dx', 'dz', 'dh', 'dw', 'dd']
        for label, val in zip(labels, std_final):
            print(f"      {label}: {val:.3f}")

        # Показываем статистику дельт
        print(f"    Delta statistics (for verification):")
        for j, label in enumerate(labels):
            p50 = np.median(all_deltas[:, j])
            p16 = np.percentile(all_deltas[:, j], 16)
            p84 = np.percentile(all_deltas[:, j], 84)
            print(f"      {label}: median={p50:+.3f}, p16={p16:+.3f}, p84={p84:+.3f}")

        return [float(x) for x in std_final]

    def _compute_deltas(self, anchors, gt_boxes):
        """Вычисляет дельты для bbox регрессии"""
        import numpy as np

        # Размеры и центры якорей
        a_h = anchors[:, 3] - anchors[:, 0]
        a_w = anchors[:, 4] - anchors[:, 1]
        a_d = anchors[:, 5] - anchors[:, 2]
        a_cy = anchors[:, 0] + 0.5 * a_h
        a_cx = anchors[:, 1] + 0.5 * a_w
        a_cz = anchors[:, 2] + 0.5 * a_d

        # Размеры и центры GT
        g_h = gt_boxes[:, 3] - gt_boxes[:, 0]
        g_w = gt_boxes[:, 4] - gt_boxes[:, 1]
        g_d = gt_boxes[:, 5] - gt_boxes[:, 2]
        g_cy = gt_boxes[:, 0] + 0.5 * g_h
        g_cx = gt_boxes[:, 1] + 0.5 * g_w
        g_cz = gt_boxes[:, 2] + 0.5 * g_d

        # Дельты
        dy = (g_cy - a_cy) / np.maximum(a_h, 1e-3)
        dx = (g_cx - a_cx) / np.maximum(a_w, 1e-3)
        dz = (g_cz - a_cz) / np.maximum(a_d, 1e-3)

        dh = np.log(np.maximum(g_h, 1e-3) / np.maximum(a_h, 1e-3))
        dw = np.log(np.maximum(g_w, 1e-3) / np.maximum(a_w, 1e-3))
        dd = np.log(np.maximum(g_d, 1e-3) / np.maximum(a_d, 1e-3))

        return np.stack([dy, dx, dz, dh, dw, dd], axis=1)

    def _print_statistics(self, stats):
        """Выводит статистику датасета"""
        import numpy as np

        xy = stats['xy_sizes']
        z = stats['z_sizes']
        z_ratios = stats['z_ratios']

        print(f"\n📊 Dataset Statistics ({stats['total_objects']} objects):")
        print("-" * 80)

        print("\nXY SIZE (sqrt(height × width)):")
        print(f"  Min:    {np.min(xy):6.1f}px  (outlier)")
        print(f"  P10:    {np.percentile(xy, 10):6.1f}px  ← Focus start")
        print(f"  P25:    {np.percentile(xy, 25):6.1f}px")
        print(f"  Median: {np.median(xy):6.1f}px")
        print(f"  P75:    {np.percentile(xy, 75):6.1f}px")
        print(f"  P90:    {np.percentile(xy, 90):6.1f}px  ← Focus end")
        print(f"  Max:    {np.max(xy):6.1f}px  (outlier)")
        print(f"  Mean:   {np.mean(xy):6.1f}px ± {np.std(xy):5.1f}px")

        print("\nZ SIZE:")
        print(f"  Min:    {np.min(z):6.1f}px")
        print(f"  P10:    {np.percentile(z, 10):6.1f}px")
        print(f"  Median: {np.median(z):6.1f}px")
        print(f"  P90:    {np.percentile(z, 90):6.1f}px")
        print(f"  Max:    {np.max(z):6.1f}px")

        print("\nZ RATIO (z / sqrt(xy)):")
        print(f"  Median: {np.median(z_ratios):.3f}")
        print(f"  P90:    {np.percentile(z_ratios, 90):.3f}")

        print("\nXY SIZE DISTRIBUTION:")
        bins = [0, 15, 25, 40, 60, 90, 130, 200, np.inf]
        labels = ['<15', '15-25', '25-40', '40-60', '60-90', '90-130', '130-200', '>200']
        hist, _ = np.histogram(xy, bins=bins)

        for label, count in zip(labels, hist):
            pct = 100 * count / len(xy)
            bar = '█' * int(pct / 2)
            if pct > 15:
                marker = " ← Main group"
            else:
                marker = ""
            print(f"  {label:>10}px: {bar:<40} {count:4d} ({pct:5.1f}%){marker}")

    def _generate_recommendations(self, stats):
        """Генерирует КОНСЕРВАТИВНЫЕ рекомендации"""
        import numpy as np

        xy = stats['xy_sizes']
        z = stats['z_sizes']
        z_ratios = stats['z_ratios']

        # ========== SCALES: 5 штук для P15, P35, P55, P75, P90 ==========
        p15 = np.percentile(xy, 15)
        p35 = np.percentile(xy, 35)
        p55 = np.percentile(xy, 55)
        p75 = np.percentile(xy, 75)
        p90 = np.percentile(xy, 90)

        scales = [
            int(round(p15)),
            int(round(p35)),
            int(round(p55)),
            int(round(p75)),
            int(round(p90))
        ]

        scales = sorted(list(set(scales)))

        while len(scales) < 5:
            gaps = [(scales[i + 1] - scales[i], i) for i in range(len(scales) - 1)]
            max_gap, idx = max(gaps)
            if max_gap > 10:
                new_scale = int((scales[idx] + scales[idx + 1]) / 2)
                scales.insert(idx + 1, new_scale)
            else:
                break

        scales = scales[:5]

        # ========== RATIOS: 3-4 штуки ==========
        r15 = np.percentile(z_ratios, 15)
        r50 = np.percentile(z_ratios, 50)
        r85 = np.percentile(z_ratios, 85)

        nice_ratios = [0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

        def round_to_nice(x):
            return min(nice_ratios, key=lambda v: abs(v - x))

        ratios = sorted(list(set([
            round_to_nice(r15),
            round_to_nice(r50),
            round_to_nice(r85)
        ])))

        if len(ratios) < 3:
            mid = round_to_nice((ratios[0] + ratios[-1]) / 2)
            if mid not in ratios:
                ratios.append(mid)
                ratios = sorted(ratios)

        ratios = ratios[:4]

        # ========== IoU ==========
        median_z = np.median(z)
        median_xy = np.median(xy)
        aspect_3d = median_z / median_xy

        if aspect_3d < 0.08:
            iou_rec = 0.25
            iou_reason = "extremely thin objects (conservative)"
        elif aspect_3d < 0.15:
            iou_rec = 0.25
            iou_reason = "very thin objects"
        elif aspect_3d < 0.25:
            iou_rec = 0.28
            iou_reason = "thin objects"
        else:
            iou_rec = 0.30
            iou_reason = "moderate objects"

        # ========== BBOX_STD_DEV ==========
        bbox_std_dev = stats.get('bbox_std_dev')
        if bbox_std_dev is None:
            bbox_std_dev = [0.15, 0.15, 0.15, 0.25, 0.25, 0.35]

        train_anchors = 1536

        current_scales = list(self.config.RPN_ANCHOR_SCALES)
        current_ratios = list(self.config.RPN_ANCHOR_RATIOS)
        current_iou = float(getattr(self.config, "RPN_POSITIVE_IOU", 0.3))
        current_bbox_std = list(getattr(self.config, 'RPN_BBOX_STD_DEV', [0.1, 0.1, 0.2, 0.2, 0.2, 0.4]))

        return {
            'scales': scales,
            'ratios': ratios,
            'iou': iou_rec,
            'iou_reason': iou_reason,
            'aspect_3d': aspect_3d,
            'bbox_std_dev': bbox_std_dev,
            'train_anchors': train_anchors,
            'current_scales': current_scales,
            'current_ratios': current_ratios,
            'current_iou': current_iou,
            'current_bbox_std_dev': current_bbox_std,
            'coverage_p10_p90': (np.percentile(xy, 10), np.percentile(xy, 90))
        }

    def _print_recommendations(self, rec, stats):
        """Выводит консервативные рекомендации"""
        import numpy as np

        print("\n" + "=" * 80)
        print("💡 CONSERVATIVE RECOMMENDATIONS (focus on P10-P90, ignore outliers)")
        print("=" * 80)

        print(f"\n🎯 Philosophy:")
        print(f"  • Focus on main 80% of data (P10-P90)")
        print(f"  • Ignore outliers (<P10, >P90)")
        print(f"  • Minimize anchor count (faster, more stable)")
        print(f"  • Use real BBOX_STD_DEV from data")
        print(f"  • Conservative IoU thresholds")

        print("\n🎯 RPN_ANCHOR_SCALES:")
        print(f"\n  Current:     {rec['current_scales']}")
        print(f"  Recommended: {rec['scales']}")
        print(f"\n  Coverage: P10={rec['coverage_p10_p90'][0]:.0f}px to P90={rec['coverage_p10_p90'][1]:.0f}px")

        xy = stats['xy_sizes']
        for s in rec['scales']:
            coverage = np.sum((xy >= s * 0.75) & (xy <= s * 1.25))
            pct = 100 * coverage / len(xy)
            print(f"    Scale {s:3d}: {coverage:4d} objects ({pct:4.1f}%)")

        print("\n🎯 RPN_ANCHOR_RATIOS:")
        print(f"\n  Current:     {rec['current_ratios']}")
        print(f"  Recommended: {rec['ratios']}")
        print(f"  Count: {len(rec['ratios'])} (minimal set)")

        print("\n🎯 RPN_POSITIVE_IOU:")
        print(f"\n  Current:     {rec['current_iou']}")
        print(f"  Recommended: {rec['iou']}")
        print(f"  Reason: {rec['iou_reason']} (aspect={rec['aspect_3d']:.3f})")

        print("\n🎯 RPN_BBOX_STD_DEV:")
        print(f"\n  Current:     {[round(x, 3) for x in rec['current_bbox_std_dev']]}")
        print(f"  Recommended: {[round(x, 3) for x in rec['bbox_std_dev']]}")
        print(
            f"  Source: {'✓ Real data (68th percentile)' if stats.get('bbox_std_dev') else '⚠️  Conservative defaults'}")

        print("\n" + "=" * 80)
        print("📋 COPY-PASTE CONFIG:")
        print("=" * 80)

        config_dict = {
            "RPN_ANCHOR_SCALES": rec['scales'],
            "RPN_ANCHOR_RATIOS": rec['ratios'],
            "RPN_POSITIVE_IOU": rec['iou'],
            "RPN_BBOX_STD_DEV": [round(x, 3) for x in rec['bbox_std_dev']],
            "RPN_TRAIN_ANCHORS_PER_IMAGE": rec['train_anchors']
        }

        import json
        print(json.dumps(config_dict, indent=4))

        print("\n" + "=" * 80)
        print("🚀 EXPECTED RESULTS:")
        print("=" * 80)

        est_anchors = len(rec['scales']) * len(rec['ratios']) * 5000
        print(f"  Total anchors: ~{est_anchors} ({len(rec['scales'])} scales × {len(rec['ratios'])} ratios)")
        print(f"  Coverage: 80% of data (P10-P90)")
        print(f"  Detection@0.50 on epoch 15-20: 55-65%")
        print(f"  Training speed: Fast (minimal anchors)")
        print(f"  Stability: High (conservative parameters)")


class TF1EarlyStopping(keras.callbacks.Callback):
    """TF1-совместимый EarlyStopping"""

    def __init__(self, monitor='val_loss', patience=10, verbose=1,
                 restore_best_weights=True, mode='auto'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        if mode == 'auto':
            if 'acc' in monitor or 'accuracy' in monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'
        else:
            self.mode = mode

        if self.mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda current, best: current < best
        else:
            self.best = float('-inf')
            self.monitor_op = lambda current, best: current > best

        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        # ДИАГНОСТИКА: проверяем наличие метрики
        if current is None:
            if self.verbose:
                print(f"\n⚠️ Metric '{self.monitor}' was not finded in logs!")
                print(f"Available metrics: {list(logs.keys())}")
            return

        # Проверяем улучшение
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved: {current:.5f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor}={current:.5f}, "
                      f"best val_loss={self.best:.5f}, wait {self.wait}/{self.patience}")

        # Проверка на остановку
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"\n🛑 Early stopping! Stop after {self.patience} Epochs without improvement.")
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    print(f"Restore best weights {epoch - self.patience + 1}")
                self.model.set_weights(self.best_weights)
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f'\nTraining stopped on Epoch {self.stopped_epoch + 1}')


class TF1ReduceLROnPlateau(keras.callbacks.Callback):
    """TF1-совместимый ReduceLROnPlateau"""

    def __init__(self, monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.wait = 0
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None or current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            try:
                import keras.backend as K
                lr_var = self.model.optimizer.lr

                # Безопасное получение значения для TF1
                try:
                    old_lr = float(K.get_value(lr_var))
                except:
                    old_lr = float(_get_value_safe(lr_var))

                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    K.set_value(lr_var, new_lr)
                    if self.verbose:
                        print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing LR to {new_lr:.2e}")
                    self.wait = 0
            except Exception:
                pass  # Тихо игнорируем ошибки

class TF1LearningRateScheduler(keras.callbacks.Callback):
    """TF1-safe версия LearningRateScheduler: не читает lr через K.get_value (чтобы не дергать .numpy()),
    а просто ставит новое значение через K.set_value.
    schedule(epoch) -> float (новый lr)"""
    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = int(verbose)

    def on_epoch_begin(self, epoch, logs=None):
        try:
            new_lr = float(self.schedule(epoch))
        except Exception as e:
            if self.verbose:
                print(f"\n[TF1LRS] schedule failed at epoch {epoch+1}: {e}")
            return
        try:
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose:
                print(f"\nEpoch {epoch+1}: TF1LearningRateScheduler sets lr -> {new_lr:.6e}")
        except Exception as e:
            if self.verbose:
                print(f"\n[TF1LRS] can't set lr: {e}")
############################################################
#  Models
############################################################


class RPN():
    """
    Encapsulates the RPN Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        config: RPN configuration
        weight_dir: training weight directory
        train_dataset: data.Dataset train object
        test_dataset: data.Dataset test object
        """

        assert config.MODE in ['training', 'targeting']

        self.config = config

        self.keras_model = self.build()
        
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()

    def prepare_datasets(self):
        """
        RPN/TARGETING: грузим RAW CSV (names, images, segs, cabs, masks) через ToyDataset.
        Никаких head-колонок здесь не требуется.
        """
        from core.data_generators import ToyDataset

        # train
        train_dataset = ToyDataset()
        train_dataset.config = self.config
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
        train_dataset.prepare()
        train_dataset.filter_positive()

        # val
        test_dataset = ToyDataset()
        test_dataset.config = self.config
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, is_train=False)
        test_dataset.prepare()
        test_dataset.filter_positive()

        print("[RPN.prepare_datasets] loaded train/test (raw CSV with names/images/segs/cabs/masks)", flush=True)
        return train_dataset, test_dataset


    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    def build(self):
        """
        Build RPN Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        # h, w, d = self.config.IMAGE_SHAPE[:3]
        # if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6) or d / 2 ** 6 != int(d / 2 ** 6):
        #     raise Exception("Image size must be dividable by 2 at least 6 times "
        #                     "to avoid fractions when downscaling and upscaling."
        #                     "For example, use 256, 320, 384, 448, 512, ... etc. ")
        h, w, d = self.config.IMAGE_SHAPE[:3]
        # Важна только кратность 64 по XY (для FPN).  Глубину больше не проверяем.
        if (h % 64) or (w % 64):
            raise ValueError("IMAGE_SHAPE height & width must be multiples of 64")
        # Inputs
        input_image = KL.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE],name="input_image_meta")
        
        # RPN targets
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 6], name="input_rpn_bbox", dtype=tf.float32)
        if np.random.rand() < 0.02:  # печатаем ~раз на 50 батчей
            print(f"[RPN targets] +:{np.sum(input_rpn_match == 1)}  -:{np.sum(input_rpn_match == -1)}  0:{np.sum(input_rpn_match == 0)}")
        # CNN
        _, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE, stage5=True, train_bn=self.config.TRAIN_BN)
        
        # Top-down Layers
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)

        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])

        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])
        
        # FPN last step
        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=(2,2,1), name="fpn_p6")(P5)

        # Features maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]

        # Anchors
        # anchors = self.get_anchors(self.config.IMAGE_SHAPE)
        # # Duplicate across the batch dimension because Keras requires it
        # anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        # # A hack to get around Keras's bad support for constants
        # anchors = KL.Lambda(
        #     lambda x, a=tf.constant(anchors, dtype=tf.float32): a,
        #     name="anchors"
        # )(input_image)
        anchors_np = self.get_anchors(self.config.IMAGE_SHAPE).astype(np.float32)

        def _anchors_layer(img, a=tf.constant(anchors_np, dtype=tf.float32)):
            # img: [B,H,W,D,C] → берём динамический B
            b = tf.shape(img)[0]
            a_exp = tf.expand_dims(a, 0)  # [1,A,6]
            a_tiled = tf.tile(a_exp, [b, 1, 1])  # [B,A,6]
            # диагностика (оставь на время дебага)
            # a_tiled = tf.compat.v1.Print(a_tiled, [
            #     tf.shape(a_tiled),
            #     tf.reduce_mean(a_tiled),
            #     tf.reduce_min(a_tiled),
            #     tf.reduce_max(a_tiled),
            #     tf.reduce_mean(a_tiled[0, :, 5] - a_tiled[0, :, 2]),
            # ], message="[ANCHORS_TILED] shape/mean/min/max/depth: ", summarize=10)
            return a_tiled

        anchors = KL.Lambda(_anchors_layer, name="anchors")(input_image)
        # RPN Model
        rpn = build_rpn_model(
            self.config.RPN_ANCHOR_STRIDE, 
            len(self.config.RPN_ANCHOR_RATIOS), 
            self.config.TOP_DOWN_PYRAMID_SIZE
        )

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        rpn_rois = ProposalLayer(
            proposal_count=self.config.POST_NMS_ROIS_TRAINING,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            image_depth=self.config.IMAGE_DEPTH,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        if self.config.MODE == "targeting":

            input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = input_gt_boxes

            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None], name="input_gt_masks", dtype=bool)

            rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config=self.config,
                train_rois_per_image=self.config.TRAIN_ROIS_PER_IMAGE,  # ← ИМЯ ПАРАМЕТРА!
                roi_positive_ratio=self.config.ROI_POSITIVE_RATIO,
                bbox_std_dev=self.config.BBOX_STD_DEV,
                use_mini_mask=self.config.USE_MINI_MASK,
                mask_shape=self.config.MASK_SHAPE,
                images_per_gpu=self.config.IMAGES_PER_GPU,
                positive_iou_threshold=self.config.RPN_POSITIVE_IOU,
                negative_iou_threshold=self.config.RPN_NEGATIVE_IOU,
                name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            rois_aligned = PyramidROIAlign(
                [self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE],
                name="roi_align_classifier"
            )([rois, input_image_meta] + [P2, P3, P4, P5])
            
            mask_aligned = PyramidROIAlign(
                [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
                name="roi_align_mask"
            )([rois, input_image_meta] + [P2, P3, P4, P5])
            
            # Model
            inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            outputs = [rois, rois_aligned, mask_aligned, target_gt_boxes, target_class_ids, target_bbox, target_mask]

            model = KM.Model(inputs, outputs, name='rpn_targeting')

        else:

            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])

            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            
            # Model
            inputs = [input_image, input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, rpn_class_loss, rpn_bbox_loss]

            model = KM.Model(inputs, outputs, name='rpn_training')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:

            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        # Создаем оптимизатор с улучшенными настройками
        opt_name = str(self.config.OPTIMIZER.get("name", "SGD")).upper()
        oparams = _keras_opt_params(self.config.OPTIMIZER.get("parameters", {}))

        if opt_name == "SGD":
            optimizer = keras.optimizers.SGD(**oparams)
        elif opt_name == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**oparams)
        else:
            optimizer = keras.optimizers.Adam(**oparams)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

        # Улучшенная балансировка весов
        custom_weights = {
            "rpn_class_loss": 1.0,
            "rpn_bbox_loss": 1.5  # Увеличиваем вес bbox loss для лучшей локализации
        }

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * custom_weights.get(name, self.config.LOSS_WEIGHTS.get(name, 1.))
            )
            self.keras_model.add_loss(loss)

        # Уменьшаем L2 регуляризацию для лучшей сходимости
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY * 0.5)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

    def train(self):
        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = RPNGenerator(dataset=self.train_dataset, config=self.config)

        # Callbacks
        evaluation = RPNEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.test_dataset)
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='RPN')
        telemetry_cb = TelemetryEpochLogger(save_dir=self.config.WEIGHT_DIR, config=self.config)
        autotune_cb = AutoTuneRPNCallback(
            train_generator=train_generator,
            config=self.config)
        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Конфигурация модели
        # Устанавливаем правильные пороги IoU в конфиге
        self.config.RPN_POSITIVE_IOU = float(getattr(self.config, "RPN_POSITIVE_IOU", 0.60))  # Было 0.7, затем 0.6
        self.config.RPN_NEGATIVE_IOU = float(getattr(self.config, "RPN_NEGATIVE_IOU", 0.20))  # Было 0.3
        self.config.RPN_TRAIN_ANCHORS_PER_IMAGE = int(getattr(self.config, "RPN_TRAIN_ANCHORS_PER_IMAGE", 512))
        self.config.RPN_POSITIVE_RATIO = float(getattr(self.config, "RPN_POSITIVE_RATIO", 0.5))
        try:
            bs = list(getattr(self.config, "BACKBONE_STRIDES"))
            fixed = []
            for s in bs:
                if isinstance(s, (tuple, list)):
                    fixed.append((int(s[0]), int(s[1]), 1))
                else:
                    ss = int(s)
                    fixed.append((ss, ss, 1))
            self.config.BACKBONE_STRIDES = fixed
        except Exception:
            pass
        # Компиляция с обновленными параметрами
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights if self.config.RPN_WEIGHTS is not None
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Новое расписание обучения
        def cosine_decay_with_warmup(epoch, total_epochs=None):
            """Косинусное затухание с разогревом и перезапуском"""
            if total_epochs is None:
                total_epochs = self.config.EPOCHS

            initial_lr = float(self.config.OPTIMIZER["parameters"]["learning_rate"])
            min_lr = initial_lr * 0.05
            warmup_epochs = 5

            # Длинный теплый старт
            if epoch < warmup_epochs:
                # Плавный разогрев с низкого значения
                return float(initial_lr * 0.1 + (initial_lr * 0.9) * (epoch / warmup_epochs))
            else:
                # Затем косинусное затухание
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return float(min_lr + (initial_lr - min_lr) * cosine_decay)

        #lr_scheduler = TF1LearningRateScheduler(cosine_decay_with_warmup, verbose=1)

        # Early Stopping для предотвращения переобучения
        early_stop = TF1EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = TF1ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)

        callbacks = [evaluation, save_weights, telemetry_cb, reduce_lr, early_stop]
        if getattr(self.config, "AUTO_TUNE_RPN", False):
            callbacks.insert(0, autotune_cb)
        # Training loop с оптимизированными параметрами
        from core.utils import Telemetry
        Telemetry.log_config_params(self.config)
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=callbacks,
            validation_data=None,
            max_queue_size=20,
            workers=1,
            use_multiprocessing=False,

        )

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        key = tuple(image_shape)
        if key not in self._anchor_cache:
            # 1) якоря в ПИКСЕЛЯХ - уже с правильными Z из ratios
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE,
                config=self.config
            )

            H, W, D = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])

            # НЕ ТРОГАЕМ Z-КООРДИНАТЫ! Они уже правильные из generate_anchors
            # Просто клипируем в границы изображения
            a[:, 0] = np.clip(a[:, 0], 0, H - 1)
            a[:, 1] = np.clip(a[:, 1], 0, W - 1)
            a[:, 2] = np.clip(a[:, 2], 0, D - 1)
            a[:, 3] = np.clip(a[:, 3], 1, H)
            a[:, 4] = np.clip(a[:, 4], 1, W)
            a[:, 5] = np.clip(a[:, 5], 1, D)

            # Гарантируем минимальные размеры
            a[:, 3] = np.maximum(a[:, 3], a[:, 0] + 1)
            a[:, 4] = np.maximum(a[:, 4], a[:, 1] + 1)
            a[:, 5] = np.maximum(a[:, 5], a[:, 2] + 0.5)

            # 3) Простая нормализация
            scale = np.array([H, W, D, H, W, D], dtype=np.float32)
            a_norm = a / scale
            a_norm = np.clip(a_norm, 0.0, 1.0).astype(np.float32)

            self._anchor_cache[key] = a_norm

            # Диагностика
            print(f"\n[RPN.get_anchors] Generated {a_norm.shape[0]} anchors")
            sizes_y = a[:, 3] - a[:, 0]
            sizes_x = a[:, 4] - a[:, 1]
            sizes_z = a[:, 5] - a[:, 2]
            xy_sizes = np.sqrt(sizes_y * sizes_x)
            z_ratios = sizes_z / (xy_sizes + 1e-8)

            print(f"  XY sizes (px): {np.percentile(xy_sizes, [10, 50, 90]).round(1)}")
            print(f"  Z sizes (px): {np.percentile(sizes_z, [10, 50, 90]).round(2)}")
            print(f"  Z/XY ratios: {np.percentile(z_ratios, [10, 50, 90]).round(3)}")
            print(f"  Z range: [{sizes_z.min():.1f}, {sizes_z.max():.1f}] (need [1, 12])")

        return self._anchor_cache[key]

    def head_target_generation(self):
        """
        Генерация таргетов для головы (TRAIN/TEST):
          - MODE == "targeting": модель отдаёт [rois, rois_aligned, mask_aligned, target_gt_boxes?, target_class_ids, target_bbox, target_mask]
          - Сохранение на диск:
              rois (fp32/.npz), rois_aligned (fp16/.npz),
              mask_aligned (битовая упаковка + shape в .npz),
              target_class_ids (int32/.npz),
              target_bbox (fp32/.npz),
              target_mask (битовая упаковка + shape в .npz)
          - CSV: OUTPUT_DIR/datasets/{train|test}.csv
          - Anchors-переменные (если есть) инициализируются точечно, без правок build().
        """
        assert self.config.MODE == "targeting", "Create model in targeting mode."

        import os
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        def _ensure_anchors_initialized():
            """Инициализирует переменные TF1, содержащие 'anchors' в имени (если они есть)."""
            try:
                import tensorflow as tf
                K = tf.compat.v1.keras.backend
                try:
                    sess = K.get_session()
                except Exception:
                    sess = tf.compat.v1.get_default_session()
                if sess is None:
                    return
                vars_anc = []
                all_vars = set(tf.compat.v1.global_variables() +
                               tf.compat.v1.trainable_variables() +
                               tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
                for v in all_vars:
                    try:
                        if "anchors" in v.name:
                            vars_anc.append(v)
                    except Exception:
                        pass
                if not vars_anc:
                    return
                to_init = []
                for v in vars_anc:
                    try:
                        if not sess.run(tf.compat.v1.is_variable_initialized(v)):
                            to_init.append(v)
                    except Exception:
                        pass
                if to_init:
                    sess.run(tf.compat.v1.variables_initializer(to_init))
            except Exception as e:
                print(f"[HEAD-TARGETS] anchors init skipped: {type(e).__name__}: {e}")

        def _bitpack(arr):
            """(packed_uint8, shape_int32). Бинаризуем >0.5, packbits по flat-оси."""
            if arr is None:
                return None, None
            a = arr
            if a.dtype != np.uint8:
                a = (a > 0.5).astype(np.uint8, copy=False)
            flat = a.reshape(-1)
            packed = np.packbits(flat)
            shape = np.array(a.shape, dtype=np.int32)
            return packed, shape

        def _save_npz(path, **arrays):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            base, _ = os.path.splitext(path)
            outp = base + ".npz"
            np.savez_compressed(outp, **arrays)
            return outp

        def _save_arrays(r, ra, ma, tci, tb, tm,
                         r_path, ra_path, ma_path, tci_path, tb_path, tm_path,
                         use_npz=True):
            """Сохраняет все артефакты; возвращает фактические пути (.npz если use_npz)."""
            if use_npz:
                if r is not None:
                    r_path = _save_npz(r_path, rois=r.astype(np.float32, copy=False))
                if ra is not None:
                    ra_path = _save_npz(ra_path, rois_aligned=ra.astype(np.float16, copy=False))
                if ma is not None:
                    mbits, mshape = _bitpack(ma)
                    ma_path = _save_npz(ma_path, mask_bits=mbits, mask_shape=mshape)
                if tci is not None:
                    tci_path = _save_npz(tci_path, tci=tci.astype(np.int32, copy=False))
                if tb is not None:
                    tb_path = _save_npz(tb_path, bbox=tb.astype(np.float32, copy=False))
                if tm is not None:
                    tm_bits, tm_shape = _bitpack(tm)
                    tm_path = _save_npz(tm_path, tm_bits=tm_bits, tm_shape=tm_shape)
            else:
                if r is not None:   np.save(r_path, r, allow_pickle=False)
                if ra is not None:  np.save(ra_path, ra, allow_pickle=False)
                if ma is not None:  np.save(ma_path, ma, allow_pickle=False)
                if tci is not None: np.save(tci_path, tci, allow_pickle=False)
                if tb is not None:  np.save(tb_path, tb, allow_pickle=False)
                if tm is not None:  np.save(tm_path, tm, allow_pickle=False)

            def _npz_pat(p):  # вернуть .npz-пути, если сохраняли в npz
                base, ext = os.path.splitext(p)
                return base + ".npz" if use_npz else p

            return (_npz_pat(r_path), _npz_pat(ra_path), _npz_pat(ma_path),
                    _npz_pat(tci_path), _npz_pat(tb_path), _npz_pat(tm_path))

        def _run_split(set_type, generator, dataset):
            ratio = float(getattr(self.config, "TARGET_RATIO", 1.0))
            n_total = len(generator.image_ids)
            n = max(1, int(round(ratio * n_total)))

            # ✅ ДОБАВИЛИ: порог для фильтрации
            min_positive = int(getattr(self.config, "MIN_POSITIVE_TARGETS", 25))

            base_path = f"{self.config.OUTPUT_DIR}"
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(os.path.join(base_path, "datasets"), exist_ok=True)

            dirs = {
                "rois": os.path.join(base_path, "rois"),
                "rois_aligned": os.path.join(base_path, "rois_aligned"),
                "mask_aligned": os.path.join(base_path, "mask_aligned"),
                "target_class_ids": os.path.join(base_path, "target_class_ids"),
                "target_bbox": os.path.join(base_path, "target_bbox"),
                "target_mask": os.path.join(base_path, "target_mask"),
            }
            for p in dirs.values():
                os.makedirs(p, exist_ok=True)

            df = pd.DataFrame(columns=[
                "rois", "rois_aligned", "mask_aligned",
                "target_class_ids", "target_bbox", "target_mask"
            ])

            print(f"[HEAD-TARGETS] split={set_type} n={n}/{n_total}")
            print(f"[HEAD-TARGETS] min_positive_threshold={min_positive}")

            _ensure_anchors_initialized()

            # ✅ ДОБАВИЛИ: статистика фильтрации
            skipped_images = []
            processed_count = 0
            saved_count = 0

            for ex_id in tqdm(range(n), desc=f"targets:{set_type}"):
                try:
                    image_id = int(generator.image_ids[ex_id])
                    info = dataset.image_info[image_id]
                    name = os.path.splitext(os.path.basename(info["path"]))[0]

                    # --- корректная распаковка из генератора ---
                    item = generator.__getitem__(ex_id)
                    if isinstance(item, tuple):
                        if len(item) == 2 and isinstance(item[0], str):
                            # (name, inputs) — targeting у RPNGenerator
                            _, inputs = item
                        elif len(item) == 2 and isinstance(item[0], (list, tuple, np.ndarray)):
                            # (inputs, outputs) — привычная форма
                            inputs, _ = item
                        else:
                            raise TypeError(
                                f"Unexpected generator item format at ex={ex_id}: types=({type(item[0])}, ...)")
                    else:
                        inputs = item  # на всякий случай

                    # --- прогон модели ---
                    outs = self.keras_model.predict(inputs, verbose=0, batch_size=1)

                    # [0]=rois, [1]=rois_aligned, [2]=mask_aligned,
                    # [3]=target_gt_boxes(опц.), [4]=target_class_ids, [5]=target_bbox, [6]=target_mask
                    if not isinstance(outs, (list, tuple)):
                        outs = [outs]

                    o = []
                    for x in outs:
                        if x is None:
                            o.append(None)
                        elif hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == 1:
                            o.append(x[0])  # убираем batch dimension
                        else:
                            o.append(x)

                    rois = o[0] if len(o) > 0 else None
                    rois_aligned = o[1] if len(o) > 1 else None
                    mask_aligned = o[2] if len(o) > 2 else None
                    target_class_ids = o[4] if len(o) > 4 else None
                    target_bbox = o[5] if len(o) > 5 else None
                    target_mask = o[6] if len(o) > 6 else None

                    processed_count += 1

                    # ✅ ФИЛЬТРАЦИЯ: проверяем количество positive примеров
                    if target_class_ids is not None:
                        pos_count = int(np.sum(target_class_ids > 0))

                        if pos_count < min_positive:
                            skipped_images.append((name, pos_count))
                            print(f"[SKIP] {name}: only {pos_count} positive (min={min_positive})")
                            continue  # ← ПРОПУСКАЕМ это изображение!
                    else:
                        print(f"[SKIP] {name}: target_class_ids is None")
                        skipped_images.append((name, 0))
                        continue

                    # Пути сохранения
                    r_path = os.path.join(dirs["rois"], f"{name}.npy")
                    ra_path = os.path.join(dirs["rois_aligned"], f"{name}.npy")
                    ma_path = os.path.join(dirs["mask_aligned"], f"{name}.npy")
                    tci_path = os.path.join(dirs["target_class_ids"], f"{name}.npy")
                    tb_path = os.path.join(dirs["target_bbox"], f"{name}.npy")
                    tm_path = os.path.join(dirs["target_mask"], f"{name}.npy")

                    # Сохранение (бит-упаковка масок)
                    r_path, ra_path, ma_path, tci_path, tb_path, tm_path = _save_arrays(
                        rois, rois_aligned, mask_aligned, target_class_ids, target_bbox, target_mask,
                        r_path, ra_path, ma_path, tci_path, tb_path, tm_path, use_npz=True
                    )

                    # CSV строка
                    df.loc[len(df)] = [r_path, ra_path, ma_path, tci_path, tb_path, tm_path]
                    saved_count += 1

                except Exception as e:
                    print(f"[HEAD-TARGETS][{set_type}][ex={ex_id}] error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

            # ✅ ИТОГОВАЯ СТАТИСТИКА
            print(f"\n{'=' * 80}")
            print(f"[HEAD-TARGETS][{set_type}] FILTERING SUMMARY:")
            print(f"  Processed:  {processed_count}")
            print(f"  Saved:      {saved_count}")
            print(f"  Skipped:    {len(skipped_images)}")
            print(f"  Keep ratio: {100 * saved_count / processed_count if processed_count else 0:.1f}%")

            if skipped_images:
                print(f"\n  Skipped images (first 20):")
                for name, cnt in skipped_images[:20]:
                    print(f"    - {name}: {cnt} positive")
                if len(skipped_images) > 20:
                    print(f"    ... and {len(skipped_images) - 20} more")

            print(f"{'=' * 80}\n")

            csv_path = os.path.join(base_path, "datasets", f"{set_type}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"[HEAD-TARGETS] {set_type} CSV saved -> {csv_path}  (rows={len(df)})")

        # --- опциональная подгрузка RPN-весов (если указаны) ---
        head_rpn_w = getattr(self.config, "RPN_WEIGHTS", None)
        if head_rpn_w:
            try:
                self.keras_model.load_weights(head_rpn_w, by_name=True)
                print("[RPN] RPN_WEIGHTS loaded by_name")
            except Exception as e:
                print(f"[RPN] load RPN_WEIGHTS failed ({type(e).__name__}): {e}")

        # --- генераторы targeting-входов ---
        from core.data_generators import RPNGenerator
        train_generator = RPNGenerator(dataset=self.train_dataset, config=self.config)
        test_generator = RPNGenerator(dataset=self.test_dataset, config=self.config)

        _run_split("train", train_generator, self.train_dataset)
        _run_split("test", test_generator, self.test_dataset)


    def evaluate(self):
        
        # Load RPN_WEIGHTS
        self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        evaluation = RPNEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.test_dataset, check_boxes=True)

        evaluation.on_epoch_end(self.epoch)

class HeadObjScoreMonitor(keras.callbacks.Callback):
    """
    Безопасный монитор "objectness" на HEAD-тренировке.
    Если в модели нет выхода 'mrcnn_obj' (как у текущей head-модели),
    монитор просто пропускается, чтобы не ронять обучение.
    """
    def __init__(self, val_gen_or_seq, steps=50):
        super().__init__()
        self.val_src = val_gen_or_seq
        self.steps = steps

    def _get_batch(self, i):
        # Если это keras.utils.Sequence — берём по индексу, циклически
        if hasattr(self.val_src, "__len__") and hasattr(self.val_src, "__getitem__"):
            L = len(self.val_src)
            if L == 0:
                raise StopIteration
            return self.val_src[i % L]
        # Иначе считаем, что это итератор
        return next(self.val_src)

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np

        names = getattr(self.model, "output_names", [])
        if "mrcnn_obj" not in names:
            # В твоей head-модели этого выхода нет — корректно пропускаем монитор
            print(f"[HEAD][epoch {epoch+1}] obj monitor skipped: no 'mrcnn_obj' output")
            return

        pos_scores, neg_scores = [], []
        oi = names.index("mrcnn_obj")

        for i in range(self.steps):
            try:
                x, _ = self._get_batch(i)
            except StopIteration:
                break

            outs = self.model.predict_on_batch(x)
            mrcnn_obj = outs[oi]  # ожидание: [B, T] или [B, T, 1]
            tci = x[3]            # target_class_ids: [B, T]

            # Приводим формы
            if mrcnn_obj.ndim == 3 and mrcnn_obj.shape[-1] == 1:
                mrcnn_obj = mrcnn_obj[..., 0]
            mrcnn_obj = mrcnn_obj.reshape(-1)
            tci = tci.reshape(-1)

            # Разделяем на pos/neg по целевым меткам
            pos = mrcnn_obj[tci > 0]
            neg = mrcnn_obj[tci <= 0]
            if pos.size:
                pos_scores.extend(pos.tolist())
            if neg.size:
                neg_scores.extend(neg.tolist())

        if pos_scores and neg_scores:
            pos_scores = np.asarray(pos_scores, dtype=np.float64)
            neg_scores = np.asarray(neg_scores, dtype=np.float64)
            # Быстрый ранговый AUC (как у тебя было)
            ranks = np.argsort(np.argsort(np.r_[pos_scores, neg_scores])).astype(np.float64)
            auc = (ranks[:len(pos_scores)].sum() - len(pos_scores) * (len(pos_scores) + 1) / 2.0) \
                  / (len(pos_scores) * len(neg_scores) + 1e-9)
            print(f"[HEAD][epoch {epoch+1}] obj_pos_mean={pos_scores.mean():.3f}  "
                  f"obj_neg_mean={neg_scores.mean():.3f}  AUC≈{auc:.3f}")
        else:
            print(f"[HEAD][epoch {epoch+1}] obj monitor: n/a (no pos/neg)")


class HEAD():
    """
    Encapsulates the Head Mask RCNN model functionality.

    MODES:
    - training: Standard HEAD training with pre-generated targets (uses ToyHeadDataset)
    - training_head_e2e: End-to-end training with frozen RPN (uses ToyDataset - raw images)
    - targeting: Generate targets for standard training
    """

    def __init__(self, config, show_summary):
        self.config = config

        # Validate required parameters
        if not hasattr(config, "MASK_SHAPE"):
            config.MASK_SHAPE = (28, 28, 28)
            print("[HEAD] Setting default MASK_SHAPE = (28, 28, 28)")

        if not hasattr(config, "MASK_POOL_SIZE"):
            config.MASK_POOL_SIZE = 14
            print("[HEAD] Setting default MASK_POOL_SIZE = 14")

        if not hasattr(config, "TOP_DOWN_PYRAMID_SIZE"):
            config.TOP_DOWN_PYRAMID_SIZE = 256
            print("[HEAD] Setting default TOP_DOWN_PYRAMID_SIZE = 256")

        self.keras_model = self.build()
        print("[HEAD.__init__] model built", flush=True)
        self.epoch = self.config.FROM_EPOCH

        # Load appropriate datasets based on mode
        self.train_dataset, self.test_dataset = self.prepare_datasets()
        print("[HEAD.__init__] datasets prepared", flush=True)

        if show_summary:
            print("[HEAD.__init__] printing summary ...", flush=True)
            self.print_summary()

    def prepare_datasets(self):
        """
        Load datasets based on training mode:
        - training_head_e2e: ToyDataset (raw images, like RPN)
        - training: ToyHeadDataset (pre-generated targets)
        - targeting: ToyDataset (for target generation)
        """
        import os
        from core.data_generators import ToyDataset, ToyHeadDataset

        if self.config.MODE == "training_head_e2e":
            # ✅ E2E MODE: Use RAW images (ToyDataset), same as RPN
            print("[HEAD.prepare_datasets] E2E mode: loading ToyDataset (raw images)")

            train_dataset = ToyDataset()
            train_dataset.config = self.config
            train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
            train_dataset.prepare()
            train_dataset.filter_positive()

            test_dataset = ToyDataset()
            test_dataset.config = self.config
            test_dataset.load_dataset(data_dir=self.config.DATA_DIR, is_train=False)
            test_dataset.prepare()
            test_dataset.filter_positive()

            print(f"[HEAD.prepare_datasets] E2E: train={len(train_dataset.image_ids)}, "
                  f"test={len(test_dataset.image_ids)}")

            return train_dataset, test_dataset

        elif self.config.MODE == "training":
            # ✅ STANDARD MODE: Use pre-generated targets (ToyHeadDataset)
            print("[HEAD.prepare_datasets] Standard mode: loading ToyHeadDataset (pre-generated targets)")

            import pandas as pd

            def _has_head_cols(csv_path):
                try:
                    td = pd.read_csv(csv_path, sep=None, engine="python")
                    cols = set(c.lower() for c in td.columns)
                    need_ra = {"rois_aligned", "ra_path", "aligned_rois"}
                    need_tci = {"target_class_ids", "tci", "tci_path"}
                    return any(k in cols for k in need_ra) and any(k in cols for k in need_tci)
                except Exception:
                    return False

            # Search for head-target CSV
            roots = []
            od = getattr(self.config, "OUTPUT_DIR", None)
            dd = getattr(self.config, "DATA_DIR", None)
            if isinstance(od, str) and od: roots.append(od)
            if isinstance(dd, str) and dd and dd != od: roots.append(dd)

            chosen = None
            for root in roots:
                csv_path = os.path.join(root, "datasets", "train.csv")
                if os.path.exists(csv_path) and _has_head_cols(csv_path):
                    chosen = root
                    break

            if chosen is None:
                raise RuntimeError(
                    "[HEAD.prepare_datasets] Head-target CSV not found with required columns "
                    "(rois_aligned/target_class_ids). Run targeting mode first or use training_head_e2e mode."
                )

            train_dataset = ToyHeadDataset()
            train_dataset.config = self.config
            train_dataset.load_dataset(data_dir=chosen)
            train_dataset.prepare()
            train_dataset.filter_positive()

            test_dataset = ToyHeadDataset()
            test_dataset.config = self.config
            test_dataset.load_dataset(data_dir=chosen, is_train=False)
            test_dataset.prepare()

            print(f"[HEAD.prepare_datasets] Standard: train={len(train_dataset.image_ids)}, "
                  f"test={len(test_dataset.image_ids)} from {chosen}")

            return train_dataset, test_dataset

        elif self.config.MODE == "targeting":
            # ✅ TARGETING MODE: Use raw images to generate targets
            print("[HEAD.prepare_datasets] Targeting mode: loading ToyDataset (for target generation)")

            train_dataset = ToyDataset()
            train_dataset.config = self.config
            train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
            train_dataset.prepare()
            train_dataset.filter_positive()

            test_dataset = ToyDataset()
            test_dataset.config = self.config
            test_dataset.load_dataset(data_dir=self.config.DATA_DIR, is_train=False)
            test_dataset.prepare()
            test_dataset.filter_positive()

            return train_dataset, test_dataset

        else:
            raise ValueError(f"[HEAD.prepare_datasets] Unknown MODE: {self.config.MODE}")

    def print_summary(self):
        self.keras_model.summary(line_length=140)
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")
        self.config.display()

    def build(self):
        """
        Build Head Mask R-CNN architecture.

        TWO ARCHITECTURES:
        1. training_head_e2e: Full end-to-end model (RPN + HEAD)
        2. training/targeting: HEAD-only model (expects pre-generated features)
        """
        import keras as KM
        import keras.layers as KL

        if self.config.MODE == "training_head_e2e":
            return self._build_e2e_model()
        elif self.config.MODE == "targeting":
            return self._build_targeting_model()
        else:  # training
            return self._build_head_only_model()

    def _build_head_only_model(self):
        """HEAD-only model for training with pre-generated targets"""
        import keras as KM
        import keras.layers as KL

        # Inputs: pre-aligned features + targets
        input_rois_aligned = KL.Input(
            shape=[
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE,
                self.config.TOP_DOWN_PYRAMID_SIZE
            ],
            name="input_rois_aligned"
        )
        input_mask_aligned = KL.Input(
            shape=[
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE,
                self.config.TOP_DOWN_PYRAMID_SIZE
            ],
            name="input_mask_aligned"
        )
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_target_class_ids = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE], name="input_target_class_ids",
                                          dtype=tf.int32)
        input_target_bbox = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, 6], name="input_target_bbox")
        input_target_mask = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, *self.config.MASK_SHAPE, 1],
                                     name="input_target_mask")

        active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

        # HEAD networks
        mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = fpn_classifier_graph(
            y=input_rois_aligned,
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
            train_bn=True
        )
        mrcnn_mask = build_fpn_mask_graph(
            y=input_mask_aligned,
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,
            train_bn=True
        )

        # Losses
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [input_target_class_ids, mrcnn_class_logits, active_class_ids]
        )
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [input_target_bbox, input_target_class_ids, mrcnn_bbox]
        )
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [input_target_mask, input_target_class_ids, mrcnn_mask]
        )

        inputs = [
            input_rois_aligned, input_mask_aligned, input_image_meta,
            input_target_class_ids, input_target_bbox, input_target_mask
        ]
        outputs = [
            mrcnn_class_logits, mrcnn_prob, mrcnn_bbox, mrcnn_mask,
            class_loss, bbox_loss, mask_loss
        ]
        model = KM.Model(inputs, outputs, name='head_training')

        if self.config.GPU_COUNT > 1:
            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def _build_targeting_model(self):
        """Model for generating HEAD targets (same as RPN targeting mode)"""
        import keras as KM
        import keras.layers as KL
        import tensorflow as tf

        # Inputs
        input_image = KL.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)

        if self.config.USE_MINI_MASK:
            input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None],
                                      name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None],
                                      name="input_gt_masks", dtype=bool)

        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

        # Backbone + FPN
        _, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE,
                                         stage5=True, train_bn=self.config.TRAIN_BN)

        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])

        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 1), name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]

        # Anchors
        anchors_np = self._get_anchors(self.config.IMAGE_SHAPE).astype(np.float32)

        def _anchors_layer(img, a=tf.constant(anchors_np, dtype=tf.float32)):
            b = tf.shape(img)[0]
            a_exp = tf.expand_dims(a, 0)
            return tf.tile(a_exp, [b, 1, 1])

        anchors = KL.Lambda(_anchors_layer, name="anchors")(input_image)

        # RPN
        rpn = build_rpn_model(
            self.config.RPN_ANCHOR_STRIDE,
            len(self.config.RPN_ANCHOR_RATIOS),
            self.config.TOP_DOWN_PYRAMID_SIZE
        )

        layer_outputs = [rpn([p]) for p in rpn_feature_maps]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Proposals
        rpn_rois = ProposalLayer(
            proposal_count=self.config.POST_NMS_ROIS_TRAINING,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            image_depth=self.config.IMAGE_DEPTH,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        # Detection targets
        rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
            config=self.config,
            train_rois_per_image=self.config.TRAIN_ROIS_PER_IMAGE,
            roi_positive_ratio=self.config.ROI_POSITIVE_RATIO,
            bbox_std_dev=self.config.BBOX_STD_DEV,
            use_mini_mask=self.config.USE_MINI_MASK,
            mask_shape=self.config.MASK_SHAPE,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            positive_iou_threshold=self.config.RPN_POSITIVE_IOU,
            negative_iou_threshold=self.config.RPN_NEGATIVE_IOU,
            name="proposal_targets"
        )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # ROI Align
        rois_aligned = PyramidROIAlign(
            [self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE],
            name="roi_align_classifier"
        )([rois, input_image_meta] + [P2, P3, P4, P5])

        mask_aligned = PyramidROIAlign(
            [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
            name="roi_align_mask"
        )([rois, input_image_meta] + [P2, P3, P4, P5])

        inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        outputs = [rois, rois_aligned, mask_aligned, target_gt_boxes, target_class_ids, target_bbox, target_mask]

        model = KM.Model(inputs, outputs, name='rpn_targeting')

        if self.config.GPU_COUNT > 1:
            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def _build_e2e_model(self):
        """
        End-to-end model: Full RPN (frozen) + trainable HEAD.

        Pipeline:
        1. Raw image → Backbone (frozen) → FPN (frozen)
        2. RPN (frozen) generates proposals
        3. DetectionTargetLayer samples ROIs dynamically
        4. PyramidROIAlign extracts features
        5. HEAD networks (trainable) predict class/bbox/mask

        NO pre-generation needed - matches inference distribution!
        """
        import keras as KM
        import keras.layers as KL
        import tensorflow as tf

        # ========== INPUTS (same as RPN training) ==========
        input_image = KL.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)

        if self.config.USE_MINI_MASK:
            input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None],
                                      name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None],
                                      name="input_gt_masks", dtype=bool)

        # Normalize GT boxes
        gt_boxes = input_gt_boxes

        # ========== BACKBONE + FPN (will be frozen) ==========
        _, C2, C3, C4, C5 = resnet_graph(input_image, self.config.BACKBONE,
                                         stage5=True, train_bn=False)

        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)

        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])

        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])

        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])

        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 1), name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]

        # ========== ANCHORS ==========
        anchors_np = self._get_anchors(self.config.IMAGE_SHAPE).astype(np.float32)

        def _anchors_layer(img, a=tf.constant(anchors_np, dtype=tf.float32)):
            b = tf.shape(img)[0]
            a_exp = tf.expand_dims(a, 0)
            a_tiled = tf.tile(a_exp, [b, 1, 1])
            return a_tiled

        anchors = KL.Lambda(_anchors_layer, name="anchors")(input_image)

        # ========== RPN (will be frozen) ==========
        rpn = build_rpn_model(
            self.config.RPN_ANCHOR_STRIDE,
            len(self.config.RPN_ANCHOR_RATIOS),
            self.config.TOP_DOWN_PYRAMID_SIZE
        )

        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # ========== PROPOSAL LAYER ==========
        rpn_rois = ProposalLayer(
            proposal_count=self.config.POST_NMS_ROIS_TRAINING,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            image_depth=self.config.IMAGE_DEPTH,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        # ========== DETECTION TARGET LAYER (dynamic sampling!) ==========
        rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
            config=self.config,
            train_rois_per_image=self.config.TRAIN_ROIS_PER_IMAGE,
            roi_positive_ratio=self.config.ROI_POSITIVE_RATIO,
            bbox_std_dev=self.config.BBOX_STD_DEV,
            use_mini_mask=self.config.USE_MINI_MASK,
            mask_shape=self.config.MASK_SHAPE,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            positive_iou_threshold=self.config.RPN_POSITIVE_IOU,
            negative_iou_threshold=self.config.RPN_NEGATIVE_IOU,
            name="proposal_targets"
        )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # ========== ROI ALIGN ==========
        rois_aligned = PyramidROIAlign(
            [self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE],
            name="roi_align_classifier"
        )([rois, input_image_meta] + [P2, P3, P4, P5])

        mask_aligned = PyramidROIAlign(
            [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
            name="roi_align_mask"
        )([rois, input_image_meta] + [P2, P3, P4, P5])

        # ========== HEAD NETWORKS (trainable!) ==========
        active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

        mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = fpn_classifier_graph(
            y=rois_aligned,
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
            train_bn=True  # ← Trainable!
        )

        mrcnn_mask = build_fpn_mask_graph(
            y=mask_aligned,
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,
            train_bn=True  # ← Trainable!
        )

        # ========== LOSSES ==========
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [target_class_ids, mrcnn_class_logits, active_class_ids]
        )
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [target_bbox, target_class_ids, mrcnn_bbox]
        )
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [target_mask, target_class_ids, mrcnn_mask]
        )

        # ========== MODEL ==========
        inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        outputs = [
            mrcnn_class_logits, mrcnn_prob, mrcnn_bbox, mrcnn_mask,
            class_loss, bbox_loss, mask_loss
        ]

        model = KM.Model(inputs, outputs, name='head_e2e_training')

        if self.config.GPU_COUNT > 1:
            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def _get_anchors(self, image_shape):
        """Generate anchors (same logic as RPN)"""

        import core.utils as utils

        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        a = utils.generate_pyramid_anchors(
            self.config.RPN_ANCHOR_SCALES,
            self.config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            self.config.BACKBONE_STRIDES,
            self.config.RPN_ANCHOR_STRIDE,
            config=self.config
        )

        H, W, D = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])

        a[:, 0] = np.clip(a[:, 0], 0, H - 1)
        a[:, 1] = np.clip(a[:, 1], 0, W - 1)
        a[:, 2] = np.clip(a[:, 2], 0, D - 1)
        a[:, 3] = np.clip(a[:, 3], 1, H)
        a[:, 4] = np.clip(a[:, 4], 1, W)
        a[:, 5] = np.clip(a[:, 5], 1, D)

        a[:, 3] = np.maximum(a[:, 3], a[:, 0] + 1)
        a[:, 4] = np.maximum(a[:, 4], a[:, 1] + 1)
        a[:, 5] = np.maximum(a[:, 5], a[:, 2] + 0.5)

        scale = np.array([H, W, D, H, W, D], dtype=np.float32)
        a_norm = a / scale
        a_norm = np.clip(a_norm, 0.0, 1.0).astype(np.float32)

        return a_norm

    def compile(self):
        """Compile model with losses"""
        import keras
        import keras.backend as K

        m = self.keras_model
        for layer in m.layers:
            if not layer.trainable and isinstance(layer, KL.BatchNormalization):
                # Отключаем updates для frozen BN
                if hasattr(layer, '_per_input_updates'):
                    layer._per_input_updates = {}
        try:
            m._losses.clear()
        except Exception:
            m._losses = []
        m._per_input_losses = {}

        # Add losses
        loss_layer_names = [
            "mrcnn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss"
        ]

        for lname in loss_layer_names:
            try:
                layer = m.get_layer(lname)
            except Exception:
                continue
            weight = float(getattr(self.config, "LOSS_WEIGHTS", {}).get(lname, 1.0))
            m.add_loss(K.mean(layer.output) * weight)

        # L2 regularization
        wd = float(getattr(self.config, "WEIGHT_DECAY", 0.0))
        if wd > 0.0:
            l2_terms = []
            for w in m.trainable_weights:
                wn = w.name
                if "gamma" in wn or "beta" in wn:
                    continue
                l2_terms.append(keras.regularizers.l2(wd)(w) / K.cast(K.prod(K.shape(w)), K.floatx()))
            if l2_terms:
                m.add_loss(tf.add_n(l2_terms))

        # Optimizer
        opt_cfg = getattr(self.config, "OPTIMIZER",
                          {"name": "SGD", "parameters": {"learning_rate": 1e-3, "momentum": 0.9}})
        oname = str(opt_cfg.get("name", "SGD")).upper()
        oparams = _keras_opt_params(opt_cfg.get("parameters", {}))

        if oname == "SGD":
            optimizer = keras.optimizers.SGD(**oparams)
        elif oname == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**oparams)
        else:
            optimizer = keras.optimizers.Adam(**oparams)

        m.compile(optimizer=optimizer, loss=[None] * len(m.outputs))

    def train(self):
        """
        Train HEAD model.
        Routes to appropriate method based on MODE.
        """
        if self.config.MODE == "training_head_e2e":
            self._train_e2e()
        else:
            self._train_standard()

    def _train_e2e(self):
        """
        ✅ END-TO-END TRAINING:
        - Uses ToyDataset (raw images)
        - Uses RPNGenerator in TARGETING mode (not training mode!)
        - RPN layers frozen
        - HEAD layers trainable
        """
        assert self.config.MODE == "training_head_e2e"

        print("\n" + "=" * 80)
        print("🚀 END-TO-END HEAD TRAINING")
        print("=" * 80)
        print("✅ RPN: FROZEN (loaded from weights)")
        print("✅ HEAD: TRAINABLE (learns from live RPN proposals)")
        print("✅ Dataset: ToyDataset (raw images, same as RPN)")
        print("✅ Generator: RPNGenerator in TARGETING mode")
        print("✅ Distribution: matches inference (end-to-end pipeline)")
        print("=" * 80 + "\n")

        # ========== GENERATOR (use targeting-style inputs!) ==========
        from core.data_generators import RPNGenerator

        # ✅ КРИТИЧНО: временно переключаем MODE на "targeting" для генератора
        original_mode = self.config.MODE
        self.config.MODE = "targeting"  # ← Генератор будет возвращать GT данные

        train_generator = RPNGenerator(
            dataset=self.train_dataset,  # ToyDataset!
            config=self.config,
            shuffle=True
        )

        val_generator = RPNGenerator(
            dataset=self.test_dataset,  # ToyDataset!
            config=self.config,
            shuffle=False
        )

        # Возвращаем режим обратно
        self.config.MODE = original_mode

        print(f"[HEAD_E2E] Using RPNGenerator (targeting mode):")
        print(f"  Train steps: {len(train_generator)}")
        print(f"  Val steps: {len(val_generator)}")
        print(f"  Inputs: [image, image_meta, gt_class_ids, gt_boxes, gt_masks]")
        print()

        # ========== CALLBACKS ==========
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='HEAD_E2E')
        early_stop = TF1EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
        reduce_lr = TF1ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, min_lr=1e-7)

        training_metrics = HeadTrainingMetricsCallback(
            self.keras_model,
            self.config,
            self.test_dataset,
            check_every=1
        )

        callbacks = [save_weights, early_stop, reduce_lr, training_metrics]

        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # 1) ✅ ОБЯЗАТЕЛЬНО загружаем RPN
        if not self.config.RPN_WEIGHTS:
            raise ValueError("[HEAD_E2E] RPN_WEIGHTS is required for end-to-end training!")

        print(f"\n[HEAD_E2E] Loading RPN weights: {self.config.RPN_WEIGHTS}")
        self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True, skip_mismatch=True)

        # 2) ✅ Загружаем HEAD: либо явно указанный, либо latest.h5 при продолжении
        if self.config.FROM_EPOCH > 0:

            # Продолжаем обучение → загружаем latest.h5
            latest_path = os.path.join(self.config.WEIGHT_DIR, "best.h5")
            if os.path.exists(latest_path):
                print(f"[HEAD_E2E] Continuing from epoch {self.config.FROM_EPOCH}")
                print(f"[HEAD_E2E] Loading HEAD weights: {latest_path}")
                self.keras_model.load_weights(latest_path, by_name=True, skip_mismatch=True)
            else:
                print(f"[HEAD_E2E] ⚠️  FROM_EPOCH={self.config.FROM_EPOCH} but {latest_path} not found!")
                print(f"[HEAD_E2E] Starting from scratch (random HEAD weights)")
        elif self.config.HEAD_WEIGHTS:
            # Начинаем с нуля, но есть pre-trained HEAD веса
            print(f"[HEAD_E2E] Loading pre-trained HEAD weights: {self.config.HEAD_WEIGHTS}")
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True, skip_mismatch=True)
        else:
            # Начинаем с нуля, случайные HEAD веса
            print(f"[HEAD_E2E] Starting from scratch (random HEAD weights)")

        # ✅ FREEZE RPN
        self._freeze_rpn_layers()
        # ========== COMPILATION ==========
        self.compile()
        # ========== TRAINING ==========
        print(f"\n[HEAD_E2E] Starting end-to-end training...")
        print(f"  Epochs: {self.config.FROM_EPOCH} → {self.config.FROM_EPOCH + self.config.EPOCHS}")
        print(f"  RPN: FROZEN ❄️")
        print(f"  HEAD: TRAINABLE 🔥")
        print()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(train_generator),
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            max_queue_size=20,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
            verbose=1
        )

        print(f"\n[HEAD_E2E] Training completed!")

    def _freeze_rpn_layers(self):
        """Freeze all RPN-related layers (backbone, FPN, RPN)"""
        print("\n[HEAD_E2E] Freezing RPN layers...")

        frozen_prefixes = [
            'res',  # ResNet backbone
            'bn',  # Batch norm in backbone
            'fpn_',  # FPN layers
            'rpn_',  # RPN conv/class/bbox
            'anchors',  # Anchors layer
        ]

        frozen_exact = ['roi']

        trainable_prefixes = [
            'mrcnn_class',
            'mrcnn_bbox',
            'mrcnn_mask',
            'roi_align',
        ]

        frozen_count = 0
        trainable_count = 0

        for layer in self.keras_model.layers:
            layer_name = layer.name.lower()

            is_head_layer = any(prefix in layer_name for prefix in trainable_prefixes)
            is_frozen_exact = layer_name in frozen_exact
            is_rpn_layer = any(layer_name.startswith(prefix) for prefix in frozen_prefixes)

            if is_head_layer:
                layer.trainable = True
                trainable_count += 1
                print(f"  🔥 TRAINABLE: {layer.name}")
            elif is_frozen_exact or is_rpn_layer:
                layer.trainable = False
                frozen_count += 1

                # ✅ КРИТИЧНО: если это BatchNorm → отключаем обновление статистики
                if isinstance(layer, KL.BatchNormalization):
                    # Явно отключаем обновление moving mean/variance
                    layer._per_input_updates = {}

            else:
                layer.trainable = False
                frozen_count += 1

        print(f"\n  ❄️  Frozen layers: {frozen_count}")
        print(f"  🔥 Trainable layers: {trainable_count}")

        # ✅ ПРОВЕРКА ПАРАМЕТРОВ
        import keras.backend as K

        trainable_params = sum([K.count_params(w) for w in self.keras_model.trainable_weights])
        non_trainable_params = sum([K.count_params(w) for w in self.keras_model.non_trainable_weights])

        print(f"\n[PARAM CHECK] After freezing:")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Non-trainable params: {non_trainable_params:,}")

        # ✅ Диагностика: показываем какие веса trainable
        if non_trainable_params < 10_000_000:
            print(f"\n❌ ERROR: Too few frozen params!")
            print(f"\nTrainable weights (should only be HEAD):")

            for w in self.keras_model.trainable_weights[:20]:  # Первые 20
                print(f"  - {w.name}: shape={w.shape}")

            raise RuntimeError(
                f"RPN not properly frozen!\n"
                f"  Expected non-trainable: >10M\n"
                f"  Actual non-trainable: {non_trainable_params:,}"
            )

        print(f"  ✅ RPN appears to be frozen\n")

    def _train_standard(self):
        """Standard HEAD training with pre-generated targets (uses HeadGenerator)"""
        assert self.config.MODE == "training"

        print(f"\n[HEAD] Standard training mode")
        print(f"  Train dataset: {len(self.train_dataset.image_ids)} images")
        print(f"  Test dataset: {len(self.test_dataset.image_ids)} images")
        print(f"  Using: ToyHeadDataset (pre-generated targets)")
        print()

        # ========== GENERATOR ==========
        from core.data_generators import HeadGenerator

        train_generator = HeadGenerator(
            dataset=self.train_dataset,  # ToyHeadDataset!
            config=self.config,
            shuffle=True
        )
        train_generator.training = True

        val_generator = HeadGenerator(
            dataset=self.test_dataset,  # ToyHeadDataset!
            config=self.config,
            shuffle=False
        )
        val_generator.training = False

        # ========== ДИАГНОСТИКА КАЧЕСТВА ТАРГЕТОВ ==========
        print("\n" + "=" * 80)
        print("🔍 TARGET QUALITY CHECK (Train Generator)")
        print("=" * 80)

        pos_counts = []
        neg_counts = []
        coverage_stats = []

        check_batches = min(10, len(train_generator))

        for i in range(check_batches):
            try:
                x, y = train_generator[i]

                # x[3] = target_class_ids: [1, T]
                target_class_ids = x[3][0]  # [T]
                target_mask = x[5][0]  # [T, mH, mW, mD, 1]

                # Считаем позитивы/негативы
                pos_mask = target_class_ids > 0
                pos = int(np.sum(pos_mask))
                neg = int(np.sum(~pos_mask))
                total = len(target_class_ids)

                pos_counts.append(pos)
                neg_counts.append(neg)

                # Проверяем coverage
                if pos > 0:
                    pos_masks = target_mask[pos_mask]
                    if pos_masks.ndim == 5:
                        pos_masks = pos_masks[..., 0]

                    coverages = np.array([float(m.mean()) for m in pos_masks])
                    coverage_stats.append({
                        'min': coverages.min(),
                        'mean': coverages.mean(),
                        'max': coverages.max()
                    })

                    print(f"  Batch {i:2d}: pos={pos:3d}/{total} ({100.0 * pos / total:5.1f}%), "
                          f"neg={neg:3d}, coverage: min={coverages.min():.3f}, mean={coverages.mean():.3f}")
                else:
                    print(f"  Batch {i:2d}: pos={pos:3d}/{total} ({100.0 * pos / total:5.1f}%), neg={neg:3d}")

            except Exception as e:
                print(f"  Batch {i:2d}: ERROR - {str(e)}")
                import traceback
                traceback.print_exc()

        # Итоговая статистика
        if pos_counts:
            avg_pos = np.mean(pos_counts)
            avg_neg = np.mean(neg_counts)
            avg_pos_ratio = avg_pos / (avg_pos + avg_neg) if (avg_pos + avg_neg) > 0 else 0.0

            balance_enabled = bool(getattr(self.config, "HEAD_BALANCE_POS", False))
            expected_ratio = 0.5 if balance_enabled else float(getattr(self.config, "HEAD_POS_FRAC", 0.33))

            print(f"\n  Average: pos={avg_pos:.1f}, neg={avg_neg:.1f}, pos_ratio={avg_pos_ratio:.3f}")
            print(f"  HEAD_BALANCE_POS: {balance_enabled}")
            print(f"  Expected pos_ratio: {expected_ratio:.3f}")

            # Coverage
            if coverage_stats:
                all_means = [s['mean'] for s in coverage_stats]
                print(f"  Positive mask coverage (mean): {np.mean(all_means):.3f}")

            # Проверки
            warnings = []
            if avg_pos < 10:
                warnings.append(f"⚠️  Very few positive examples ({avg_pos:.1f} < 10)!")
            if balance_enabled and abs(avg_pos_ratio - 0.5) > 0.15:
                warnings.append(f"⚠️  Balance not working! Got {avg_pos_ratio:.3f}, expected 0.5!")
            if coverage_stats and np.mean(all_means) < 0.1:
                warnings.append(f"⚠️  Very low mask coverage ({np.mean(all_means):.3f} < 0.1)!")

            if warnings:
                print(f"\n  {'=' * 76}")
                for w in warnings:
                    print(f"  {w}")
                print(f"  {'=' * 76}")

                if avg_pos < 5 or (balance_enabled and abs(avg_pos_ratio - 0.5) > 0.3):
                    raise RuntimeError("Invalid training targets! Check HEAD_BALANCE_POS and __getitem__")
            else:
                print(f"\n  ✅ OK: Target quality looks good!")
        else:
            raise RuntimeError("Failed to load training batches!")

        print("=" * 80 + "\n")

        # ========== CALLBACKS ==========
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='HEAD')
        early_stop = TF1EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = TF1ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)
        training_metrics = HeadTrainingMetricsCallback(
            self.keras_model,
            self.config,
            self.test_dataset,  # ✅ Валидация на test_dataset
            check_every=1
        )
        callbacks = [save_weights, early_stop, reduce_lr, training_metrics]

        # ========== КОМПИЛЯЦИЯ ==========
        self.compile()

        # ========== ЗАГРУЗКА ВЕСОВ ==========
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        if self.config.HEAD_WEIGHTS:
            print(f"[HEAD] Loading HEAD weights: {self.config.HEAD_WEIGHTS}")
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)
        elif self.config.RPN_WEIGHTS:
            print(f"[HEAD] Loading RPN weights: {self.config.RPN_WEIGHTS}")
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True, skip_mismatch=True)
        elif self.config.MASK_WEIGHTS:
            print(f"[HEAD] Loading MASK weights: {self.config.MASK_WEIGHTS}")
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True, skip_mismatch=True)

        # ========== ОБУЧЕНИЕ ==========
        print(f"\n[HEAD] Starting training...")
        print(f"  Epochs: {self.config.FROM_EPOCH} → {self.config.FROM_EPOCH + self.config.EPOCHS}")
        print(f"  Train steps: {len(train_generator)}")
        print(f"  Val steps: {len(val_generator)}")

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(train_generator),
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            max_queue_size=1,
            workers=0,
            use_multiprocessing=False,
            shuffle=False,  # Shuffle в генераторе
            verbose=1
        )

        print(f"[HEAD] Training completed!")




from keras import backend as K

def _ensure_tf1_graph():
    try:
        tf.compat.v1.disable_v2_behavior()
    except Exception:
        pass
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    try:
        if hasattr(K, "set_session"):
            K.set_session(_TF_SESSION)
        if hasattr(tf.compat.v1.keras.backend, "set_session"):
            tf.compat.v1.keras.backend.set_session(_TF_SESSION)
    except Exception as e:
        print("[tf-compat] _ensure_tf1_graph set_session warn:", e)



class MaskRCNN():
    """Encapsulates the whole Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert config.MODE in ['training', 'inference']

        self.config = config

        self.keras_model = self.build()

        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.test_dataset = self.prepare_datasets()

        if show_summary:
            self.print_summary()


    def _force_load_head_by_suffix(self, h5path):
        """
        Ремап весов головы в новый H5 под точные имена/формы ожидаемых тензоров,
        затем стандартная загрузка by_name. Никаких присваиваний в графе.
        """
        import os, h5py, numpy as np, tempfile, shutil

        def _open_root(f):
            return f["model_weights"] if "model_weights" in f.keys() else f

        def _list_all_datasets_with_shapes(root):
            """Плоский индекс: basename -> [(full_path, np.shape)]"""
            idx = {}

            def _visit(name, obj):
                import h5py
                if isinstance(obj, h5py.Dataset):
                    base = name.split("/")[-1]  # например 'kernel:0'
                    shp = tuple(obj.shape)
                    idx.setdefault(base, []).append((name, shp))

            root.visititems(_visit)
            return idx

        def _expected_names_and_shapes(layer):
            """Ожидаемые имена датасетов и их формы из layer.weights (ТОЛЬКО имена/формы, без чтения значений)."""
            names, shapes = [], []
            for w in layer.weights:
                wname = w.name.split("/")[-1]  # 'kernel:0' / 'bias:0' / 'gamma:0' ...
                shp = tuple(w.shape.as_list())
                names.append(wname)
                shapes.append(shp)
            return names, shapes

        def _match_from_src(src_index, lname, wname, shape):
            """
            Ищем подходящий датасет в исходном H5.
            Стратегия:
              1) точное совпадение basename + shape,
              2) совпадение по суффиксу '<layer>/<wbase>:0' + shape,
              3) совпадение по суффиксу '<layer>_<wbase>:0' + shape,
              4) любое basename=wname (без shape-фильтра) — как последний шанс.
            """
            wbase = wname.split(":")[0]  # kernel/bias/gamma/...
            # 1) точное basename + shape
            for full, shp in src_index.get(wname, []):
                if shp == shape:
                    return full
            # 2) .../<layer>/<wbase>:0
            cand = f"{lname}/{wbase}:0"
            for base, lst in src_index.items():
                for full, shp in lst:
                    if full.endswith("/" + cand) and shp == shape:
                        return full
            # 3) .../<layer>_<wbase>:0
            cand = f"{lname}_{wbase}:0"
            for base, lst in src_index.items():
                for full, shp in lst:
                    if full.endswith("/" + cand) and shp == shape:
                        return full
            # 4) basename без проверки формы (крайний случай)
            lst = src_index.get(wname, [])
            return lst[0][0] if lst else None

        if not h5path or not os.path.exists(h5path):
            print(f"[HEAD][force] skip: H5 not found -> {h5path}")
            return

        # Временный H5 с ремапом
        tmp_dir = tempfile.mkdtemp(prefix="mrcnn_head_remap_")
        tmp_h5 = os.path.join(tmp_dir, "head_remap.h5")
        try:
            with h5py.File(h5path, "r") as fin, h5py.File(tmp_h5, "w") as fout:
                src_root = _open_root(fin)
                dst_root = fout.create_group("model_weights")
                # Индекс всех датасетов с формами для быстрого поиска
                src_index = _list_all_datasets_with_shapes(src_root)

                loaded, skipped = 0, 0

                for layer in self.keras_model.layers:
                    lname = getattr(layer, "name", "")
                    if not lname.startswith("mrcnn_"):
                        continue

                    # Ожидаемые имена/формы
                    try:
                        exp_names, exp_shapes = _expected_names_and_shapes(layer)
                    except Exception as e:
                        print(f"[HEAD][force] skip {lname}: can't read expected shapes ({e})")
                        skipped += 1
                        continue
                    if not exp_names:
                        # слой без собственных весов — создадим пустую группу для совместимости
                        lgrp = dst_root.create_group(lname)
                        lsub = lgrp.create_group(lname)
                        print(f"[HEAD][force] {lname:22s} h5_norms=[]")
                        continue

                    # Создаём группы назначения
                    lgrp = dst_root.create_group(lname)
                    lsub = lgrp.create_group(lname)

                    ok_this = True
                    norms = []
                    for wn, shp in zip(exp_names, exp_shapes):
                        src_path = _match_from_src(src_index, lname, wn, shp)
                        if src_path is None:
                            print(f"[HEAD][force] {lname}: miss '{wn}' shape{shp} in src")
                            ok_this = False
                            break
                        ds = fin[src_path]
                        arr = np.array(ds)
                        if tuple(arr.shape) != tuple(shp):
                            print(f"[HEAD][force] {lname}: shape mismatch for '{wn}': src{arr.shape} vs dst{shp}")
                            ok_this = False
                            break
                        lsub.create_dataset(wn, data=arr)
                        norms.append(float(np.linalg.norm(arr)))

                    if ok_this:
                        loaded += 1
                        print(f"[HEAD][force] {lname:22s} h5_norms={norms}")
                    else:
                        # убираем неполную группу
                        del dst_root[lname]
                        skipped += 1

            # Теперь обычная безопасная загрузка «по имени»
            self.keras_model.load_weights(tmp_h5, by_name=True)
            print(f"[HEAD][force] remap+load by_name done: {loaded} layers, skipped={skipped}")

        finally:
            # Временные файлы подчищаем
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


    def _load_head_with_class_slice(self, h5path, class_ix_map):
        """
        Догружает веса mrcnn_* из H5, делая срез по классам.
        class_ix_map: список индексов классов из чекпоинта, которые надо оставить
                      (включая фон 0). Пример: [0, 1] для binary (bg+1).
        Ничего не меняет в архитектуре; только set_weights нужных слоёв.
        """
        import os, h5py, numpy as np
        if not h5path or not os.path.exists(h5path) or class_ix_map is None:
            return
        class_ix_map = [int(i) for i in class_ix_map]

        with h5py.File(h5path, "r") as f:
            root = f["model_weights"] if "model_weights" in f.keys() else f

            def _find_group(h5root, name):
                if name in h5root:
                    return h5root[name]
                hit = None
                def _visit(n, obj):
                    nonlocal hit
                    if hit is None and isinstance(obj, h5py.Group):
                        if n.endswith("/" + name) or n.endswith(name):
                            hit = obj
                h5root.visititems(_visit)
                return hit

            def _read_kb(layer_name):
                g = _find_group(root, layer_name)
                if g is None:
                    return None, None
                # чаще всего kernel/bias лежат в подгруппе с тем же именем
                sub = g.get(layer_name.split("/")[-1], None)
                cand = sub if isinstance(sub, h5py.Group) else g
                k = cand.get("kernel:0", None)
                b = cand.get("bias:0",   None)
                k = np.array(k[()]) if k is not None else None
                b = np.array(b[()]) if b is not None else None
                return k, b

            def _safe_set(layer_name, kernel, bias, slicer=None):
                try:
                    layer = self._get_layer_by_suffix(layer_name)
                    if layer is None or kernel is None or bias is None:
                        return False
                    if slicer is not None:
                        kernel = kernel[..., slicer]
                        bias   = bias[slicer]
                    layer.set_weights([kernel, bias])
                    print(f"[HEAD][slice] set {layer_name} <- {kernel.shape} / {bias.shape}")
                    return True
                except Exception as e:
                    print(f"[HEAD][slice] failed {layer_name}: {e}")
                    return False

            # ----- class logits -----
            k, b = _read_kb("mrcnn_class_logits")
            if k is not None and b is not None:
                _safe_set("mrcnn_class_logits", k, b, slicer=np.array(class_ix_map, dtype=np.int64))

            # ----- bbox fc (по 6 параметров на класс) -----
            k, b = _read_kb("mrcnn_bbox_fc")
            if k is not None and b is not None and b.size % 6 == 0:
                oldC = b.size // 6
                idx = []
                for c in class_ix_map:
                    if 0 <= c < oldC:
                        idx.extend(range(6 * c, 6 * (c + 1)))
                idx = np.array(idx, dtype=np.int64)
                # срез по последней оси: (..., 6*C_old) -> (..., 6*C_new)
                k_s = k[..., idx]
                b_s = b[idx]
                _safe_set("mrcnn_bbox_fc", k_s, b_s, slicer=None)

            # ----- mask conv (последняя ось = C) -----
            k, b = _read_kb("mrcnn_mask")
            if k is not None and b is not None:
                _safe_set("mrcnn_mask", k, b, slicer=np.array(class_ix_map, dtype=np.int64))


    def _infer_head_params_from_h5(self, weights_path):

        import h5py
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"no head weights: {weights_path}")

        with h5py.File(weights_path, "r") as f:
            root = f["model_weights"] if "model_weights" in f.keys() else f

            def _find_group(h5root, name):
                if name in h5root:
                    return h5root[name]
                hit = None

                def _visit(n, obj):
                    nonlocal hit
                    if hit is None and isinstance(obj, h5py.Group):
                        if n.endswith("/" + name) or n.endswith(name):
                            hit = obj

                h5root.visititems(_visit)
                if hit is None:
                    raise KeyError(f"layer '{name}' not found in H5")
                return hit

            def _kernel_shape(layer_name):
                g = _find_group(root, layer_name)
                # Частый Keras-формат: <layer>/<layer>/{kernel:0, bias:0, ...}
                # Сначала ищем kernel в этом слое, затем в одноимённой подгруппе.
                for candidate in (g, g.get(layer_name.split("/")[-1], None)):
                    if isinstance(candidate, h5py.Group) and "kernel:0" in candidate:
                        return tuple(candidate["kernel:0"].shape)
                # Фолбэк: рекурсивно ищем первый датасет с ndim==5 (Conv3D kernel)
                target = {"shape": None}

                def _visit(n, obj):
                    if target["shape"] is None and isinstance(obj, h5py.Dataset):
                        shp = tuple(obj.shape)
                        if len(shp) == 5:
                            target["shape"] = shp

                g.visititems(_visit)
                if target["shape"] is None:
                    raise KeyError(f"no tensors in layer '{layer_name}'")
                return target["shape"]

            # Conv3D classifier conv1: (pool, pool, pool, Cin, fc)
            s_cls1 = _kernel_shape("mrcnn_class_conv1")
            # Conv3D mask conv1: (3, 3, 3, Cin, mask_ch) — берём последний канал как HEAD_CONV_CHANNEL
            s_m1 = _kernel_shape("mrcnn_mask_conv1")

            pool = int(s_cls1[0])  # пространственный размер ядра = POOL_SIZE
            fc = int(s_cls1[-1])  # выходных каналов conv1 = размер FC-слоя классификатора
            mask = int(s_m1[-1])  # выходных каналов mask_conv1 = HEAD_CONV_CHANNEL

            # Санити-чек на POOL (адекватные значения обычно 3..32)
            if not (1 <= pool <= 64):
                pool = int(getattr(self.config, "POOL_SIZE", 7))

            return {"pool": pool, "fc": fc, "mask": mask}

    def _get_layer_by_suffix(self, suffix):
        for layer in self.keras_model.layers:
            if layer.name.endswith(suffix):
                return layer
        return None

    def _build_head_skeleton(self, fc_ch):
        """Мини-сетка головы (ROIAlign + cls/bbox + mask) для проверки соответствия весам."""
        # Входы
        rois = KL.Input(shape=[None, 6], name="input_rois")
        meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_meta")
        # FPN-фичи (формально — просто входы нужной формы; имена важны)
        P2 = KL.Input(shape=[None, None, None, self.config.TOP_DOWN_PYRAMID_SIZE], name="P2")
        P3 = KL.Input(shape=[None, None, None, self.config.TOP_DOWN_PYRAMID_SIZE], name="P3")
        P4 = KL.Input(shape=[None, None, None, self.config.TOP_DOWN_PYRAMID_SIZE], name="P4")
        P5 = KL.Input(shape=[None, None, None, self.config.TOP_DOWN_PYRAMID_SIZE], name="P5")

        # Классификатор/регрессор
        cls_log, cls_prob, bbox = fpn_classifier_graph_with_RoiAlign(
        rois=rois, feature_maps=[P2,P3,P4,P5], image_meta=meta,
        pool_size=self.config.POOL_SIZE, num_classes=self.config.NUM_CLASSES,
        fc_layers_size=int(fc_ch), train_bn=False, name_prefix="probe_"
    )
        masks = build_fpn_mask_graph_with_RoiAlign(
        rois=rois, feature_maps=[P2,P3,P4,P5], image_meta=meta,
        pool_size=self.config.MASK_POOL_SIZE, num_classes=self.config.NUM_CLASSES,
        conv_channel=int(fc_ch), train_bn=False, name_prefix="probe_"
    )

        return KM.Model([rois, meta, P2, P3, P4, P5],
                        [cls_log, cls_prob, bbox, masks],
                        name=f"head_skeleton_{int(fc_ch)}")

    def _count_h5_matches(self, model, weights_path):
        """Подсчитать, сколько слоёв модели имеют совместимые веса в .h5.
           Поддерживает layout Keras: корень и 'model_weights'. Сравнение по exact-name и по suffix.
        """
        try:
            import h5py
            f = h5py.File(weights_path, 'r')
        except Exception:
            return 0

        # где лежат веса
        root = f['model_weights'] if 'model_weights' in f.keys() else f

        # собрать в H5: имя_слоя -> кол-во тензоров и их формы
        import h5py as _h5
        h5_layers = {}

        def _walk(group, prefix=""):
            for k, v in group.items():
                if isinstance(v, _h5.Group):
                    # если внутри есть датасеты — считаем как слой
                    dsets = [x for x in v.values() if isinstance(x, _h5.Dataset)]
                    if dsets:
                        h5_layers[prefix + k] = [tuple(x.shape) for x in dsets]
                    # обход глубже
                    _walk(v, prefix + k + "/")

        _walk(root, "")

        # слои модели: имя -> формы тензоров
        model_layers = {}
        for l in model.layers:
            try:
                shapes = []
                for w in l.weights:
                    s = w.shape
                    if hasattr(s, "as_list"):
                        s = tuple(s.as_list())
                    else:
                        s = tuple(s)
                    shapes.append(s)
                model_layers[l.name] = shapes
            except Exception:
                pass

        # метчим: exact-name, иначе по суффиксу
        matched = 0
        for lname, lshapes in model_layers.items():
            hit = None
            if lname in h5_layers:
                hit = lname
            else:
                # suffix match (на случай префиксов)
                cands = [k for k in h5_layers.keys() if k.endswith("/" + lname) or k.endswith(lname)]
                if cands:
                    hit = cands[0]
            if hit:
                # грубая проверка совместимости: совпадает число тензоров и последняя размерность ядра
                h5sh = h5_layers[hit]
                if len(h5sh) >= 1 and len(lshapes) >= 1:
                    matched += 1

        try:
            f.close()
        except Exception:
            pass
        return matched

    def _pick_head_channels(self):
        """Выбрать ширину головы так, чтобы веса HEAD_WEIGHTS совпали максимально.
           Кандидаты: HEAD_CONV_CHANNEL и FPN_CLASSIF_FC_LAYERS_SIZE.
        """
        import os
        w = getattr(self.config, "HEAD_WEIGHTS", None)
        # если весов нет — оставляем config.HEAD_CONV_CHANNEL
        if not w or not os.path.exists(w):
            return int(getattr(self.config, "HEAD_CONV_CHANNEL", 128))

        cand = []
        hch = int(getattr(self.config, "HEAD_CONV_CHANNEL", 128))
        if hch not in cand:
            cand.append(hch)
        fc512 = int(getattr(self.config, "FPN_CLASSIF_FC_LAYERS_SIZE", hch))
        if fc512 not in cand:
            cand.append(fc512)

        best = cand[0]
        best_hits = -1
        for ch in cand:
            sk = self._build_head_skeleton(ch)
            hits = self._count_h5_matches(sk, w)
            # простая эвристика: больше совпало — лучше
            if hits > best_hits:
                best_hits = hits
                best = ch

        if best != hch:
            print(f"[HEAD][auto] override channels: config.HEAD_CONV_CHANNEL={hch} -> using {best} (by weights match)")
        else:
            print(f"[HEAD][auto] using HEAD_CONV_CHANNEL={best}")
        return int(best)

    def _head_weight_healthcheck(self):
        import numpy as np
        print("[HEAD] Weights healthcheck (L2-norms):")

        # --- безопасно читаем нормы прямо из H5, с рекурсивным обходом ---
        try:
            import os, h5py
            wpath = getattr(self.config, "HEAD_WEIGHTS", None)
            if not wpath or not os.path.exists(wpath):
                print("[HEAD] H5 norms skipped: HEAD_WEIGHTS not set/found")
                return
            f = h5py.File(wpath, 'r')
            root = f['model_weights'] if 'model_weights' in f.keys() else f

            wanted = [
                "mrcnn_class_conv1", "mrcnn_class_conv2", "mrcnn_class_logits", "mrcnn_bbox_fc",
                # ВНИМАНИЕ: 'mrcnn_bbox' — это reshape-слой, у него нет собственных весов.
                "mrcnn_bbox",
                "mrcnn_mask_conv1", "mrcnn_mask_conv2", "mrcnn_mask_conv3", "mrcnn_mask_conv4", "mrcnn_mask_deconv",
                "mrcnn_mask",
            ]

            def _collect_groups(h5root):
                import h5py
                hits = {}

                def _visit(name, obj):
                    if isinstance(obj, h5py.Group):
                        # считаем группой-слоем, если внутри есть датасеты (kernel/bias/...)
                        if any(isinstance(v, h5py.Dataset) for v in obj.values()):
                            hits[name] = [k for k, v in obj.items() if isinstance(v, h5py.Dataset)]

                h5root.visititems(_visit)
                return hits

            groups = _collect_groups(root)

            print("[HEAD] H5 norms:")
            for suf in wanted:
                cand = None
                for g in groups.keys():
                    if g == suf or g.endswith("/" + suf) or g.endswith(suf):
                        cand = g
                        break
                if not cand:
                    # отдельная пометка для mrcnn_bbox
                    if suf == "mrcnn_bbox":
                        print(f"  {suf:22s} : - (reshape, no own weights)")
                    else:
                        print(f"  {suf:22s} : -")
                    continue
                try:
                    ds = [root[cand][k][()] for k in groups[cand]]
                    norms = [float(np.linalg.norm(np.array(d))) for d in ds if hasattr(d, "shape")]
                    if norms:
                        print(f"  {suf:22s} : " + ", ".join(f"{n:.3f}" for n in norms))
                    else:
                        if suf == "mrcnn_bbox":
                            print(f"  {suf:22s} : - (reshape, no own weights)")
                        else:
                            print(f"  {suf:22s} : -")
                except Exception:
                    if suf == "mrcnn_bbox":
                        print(f"  {suf:22s} : - (reshape, no own weights)")
                    else:
                        print(f"  {suf:22s} : -")
            try:
                f.close()
            except Exception:
                pass
        except Exception as e:
            print(f"[HEAD] H5 norms skipped: {e}")


    def _debug_list_h5_head_layers(self, h5path, prefix="mrcnn_"):
            """Печатает первые слои из H5, чьи имена содержат prefix (по-умолчанию 'mrcnn_')."""
            try:
                import os, h5py
                if not h5path or not os.path.exists(h5path):
                    print("[HEAD][H5] not found:", h5path)
                    return
                f = h5py.File(h5path, "r")
                root = f["model_weights"] if "model_weights" in f.keys() else f
                names = []

                def _visit(name, obj):
                    if isinstance(obj, h5py.Group):
                        if any(isinstance(v, h5py.Dataset) for v in obj.values()) and prefix in name:
                            names.append(name)

                root.visititems(_visit)
                print(f"[HEAD][H5] groups with '{prefix}':", len(names))
                for n in names[:30]:
                    print("   -", n)
                try:
                    f.close()
                except Exception:
                    pass
            except Exception as e:
                print(f"[HEAD][H5] list failed ({type(e).__name__}):", e)

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = ToyDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
        train_dataset.prepare()


        test_dataset = ToyDataset()
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, is_train=False)
        test_dataset.prepare()
        test_dataset.filter_positive()

        print("[HEAD.prepare_datasets] loaded train/test", flush=True)
        return train_dataset, test_dataset


    def print_summary(self):

        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()




    def build(self):
        """
        Полная сборка Mask R-CNN 3D:
          TRAINING:
            - TRAIN_PHASE='rpn'   : только RPN + лоссы RPN
            - TRAIN_PHASE='heads' : DetectionTarget -> Heads -> лоссы головы
          INFERENCE:
            - RPN -> Proposals -> Heads -> DetectionLayer -> MaskHead
          Плюс eval-модели с безопасными префиксами имён.
        """

        import keras.layers as KL
        import keras.models as KM
        from keras import backend as K

        assert self.config.MODE in ("training", "inference")
        if self.config.MODE == "training":
            phase = str(getattr(self.config, "TRAIN_PHASE", "rpn")).lower()
            assert phase in ("rpn", "heads")

        # ---------- (0) чистый граф (TF1-совместимый) ----------
        _ensure_tf1_graph()

        # ---------- (1) подобрать ширину головы по сохранённым весам ----------
        fc_ch = int(getattr(self.config, "HEAD_CONV_CHANNEL", 128))
        try:
            fc_ch = int(self._pick_head_channels())
            print("[build] using HEAD_CONV_CHANNEL:", fc_ch)
        except Exception as e:
            print("[build] _pick_head_channels() failed; fallback", fc_ch, ":", e)

        # СНОВА чистый граф после пробы, чтобы не было коллизий имён
        _ensure_tf1_graph()

        # ---------- (2) Inputs ----------
        input_image = KL.Input(shape=[*self.config.IMAGE_SHAPE], name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_anchors = KL.Input(shape=[None, 6], name="anchors")

        # ---------- (3) Backbone ----------
        # resnet_graph -> (C1, C2, C3, C4, C5)
        _, C2, C3, C4, C5 = resnet_graph(input_image,
                                         self.config.BACKBONE,
                                         stage5=True,
                                         train_bn=self.config.TRAIN_BN)

        # ---------- (4) FPN (апсемплим по XY) ----------
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c5p5')(C5)

        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p3upsampled")(P3),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c2p2')(C2)
        ])

        P2 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="same", name="fpn_p2")(P2)
        P3 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="same", name="fpn_p3")(P3)
        P4 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="same", name="fpn_p4")(P4)
        P5 = KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="same", name="fpn_p5")(P5)
        P6 = KL.MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 1), name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # ---------- (5) RPN ----------
        rpn = build_rpn_model(self.config.RPN_ANCHOR_STRIDE,
                              len(self.config.RPN_ANCHOR_RATIOS),
                              self.config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = [rpn([p]) for p in rpn_feature_maps]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        rpn_class_logits, rpn_class, rpn_bbox = [
            KL.Concatenate(axis=1, name=n)([o[i] for o in layer_outputs])
            for i, n in enumerate(output_names)
        ]

        # ---------- (6) Proposals ----------
        proposal_count = (self.config.POST_NMS_ROIS_TRAINING
                          if self.config.MODE == "training"
                          else self.config.POST_NMS_ROIS_INFERENCE)
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            image_depth=self.config.IMAGE_DEPTH,
            name="ROI"
        )([rpn_class, rpn_bbox, input_anchors])

        # =========================
        # ======== TRAINING =======
        # =========================
        if self.config.MODE == "training":
            # --- GT inputs ---
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 6], name="input_rpn_bbox", dtype=tf.float32)

            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None],
                                          name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None],
                                          name="input_gt_masks", dtype=bool)

            # нормализованные GT для DetectionTargetLayer
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]),
                                 name="gt_boxes_norm")(input_gt_boxes)

            if phase == "rpn":
                # --- RPN losses ---
                rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                           name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
                rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x),
                                          name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])

                inputs = [input_image, input_image_meta, input_anchors, input_rpn_match, input_rpn_bbox]
                outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, rpn_class_loss, rpn_bbox_loss]
                self.keras_model = KM.Model(inputs, outputs, name="mask_rcnn_train_rpn")
                return self.keras_model

            # --- Detection targets (для тренировки головы) ---
            rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config=self.config,
                train_rois_per_image=self.config.TRAIN_ROIS_PER_IMAGE,  # ← ИМЯ ПАРАМЕТРА!
                roi_positive_ratio=self.config.ROI_POSITIVE_RATIO,
                bbox_std_dev=self.config.BBOX_STD_DEV,
                use_mini_mask=self.config.USE_MINI_MASK,
                mask_shape=self.config.MASK_SHAPE,
                images_per_gpu=self.config.IMAGES_PER_GPU,
                positive_iou_threshold=self.config.RPN_POSITIVE_IOU,
                negative_iou_threshold=self.config.RPN_NEGATIVE_IOU,
                name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # --- Heads (classifier+bbox) ---
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(
                rois=rois,
                feature_maps=mrcnn_feature_maps,
                image_meta=input_image_meta,
                pool_size=self.config.POOL_SIZE,
                num_classes=self.config.NUM_CLASSES,
                fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,  # <-- фикс: ширина classifier как в инференсе
                train_bn=True,
                name_prefix="", config=self.config  # каноничные имена mrcnn_*
            )

            # --- Mask head ---
            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
                rois=rois,
                feature_maps=mrcnn_feature_maps,
                image_meta=input_image_meta,
                pool_size=self.config.MASK_POOL_SIZE,
                num_classes=self.config.NUM_CLASSES,
                conv_channel=self.config.HEAD_CONV_CHANNEL,  # <-- фикс: ширина mask-ветки = HEAD_CONV_CHANNEL
                train_bn=True,
                name_prefix=""  # каноничные имена mrcnn_*
            )

            # --- Лоссы головы ---
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"],
                name="active_class_ids"
            )(input_image_meta)

            # ВАЖНО: передаём аргументы лоссов строго в каноническом порядке и через lambda x: fn(*x)
            mrcnn_class_loss = KL.Lambda(
                lambda x: mrcnn_class_loss_graph(*x),
                name="mrcnn_class_loss"
            )([target_class_ids, mrcnn_class_logits, active_class_ids])

            mrcnn_bbox_loss = KL.Lambda(
                lambda x: mrcnn_bbox_loss_graph(*x),
                name="mrcnn_bbox_loss"
            )([target_bbox, target_class_ids, mrcnn_bbox])

            mrcnn_mask_loss = KL.Lambda(
                lambda x: mrcnn_mask_loss_graph(*x),
                name="mrcnn_mask_loss"
            )([target_mask, target_class_ids, mrcnn_mask])

            # --- финальные inputs/outputs для TRAINING(heads) ---
            inputs = [input_image, input_image_meta, input_anchors,
                      input_gt_class_ids, input_gt_boxes, input_gt_masks]
            outputs = [mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
            self.keras_model = KM.Model(inputs, outputs, name="mask_rcnn_train_heads")
            return self.keras_model

        # =========================
        # ======== INFERENCE ======
        # =========================

        head_h5 = getattr(self.config, "HEAD_WEIGHTS", None)
        try:
            hp = self._infer_head_params_from_h5(head_h5)
            pool_from_h5 = int(hp["pool"])
            fc_class_ch = int(hp["fc"])
            mask_ch = int(hp["mask"])

            self.config.POOL_SIZE = pool_from_h5
            self.config.FPN_CLASSIF_FC_LAYERS_SIZE = fc_class_ch
            self.config.HEAD_CONV_CHANNEL = mask_ch

            print(f"[HEAD][infer] using POOL_SIZE={pool_from_h5}, "
                  f"classifier_ch={fc_class_ch}, mask_ch={mask_ch} from H5")
        except Exception as e:
            print(f"[HEAD][infer] H5 introspection failed: {e} ; fallback to config")
            fc_class_ch = int(getattr(self.config, "FPN_CLASSIF_FC_LAYERS_SIZE",
                                      getattr(self.config, "HEAD_CONV_CHANNEL", 128)))
            mask_ch = int(getattr(self.config, "HEAD_CONV_CHANNEL", fc_class_ch))
            self.config.FPN_CLASSIF_FC_LAYERS_SIZE = fc_class_ch
            self.config.HEAD_CONV_CHANNEL = mask_ch

        # ========== ROI ALIGN - ТЕ ЖЕ ИМЕНА ЧТО И В HEAD TRAINING! ==========
        # ✅ Для classifier - ОТДЕЛЬНЫЙ ROI Align
        rois_aligned_classifier = PyramidROIAlign(
            [self.config.POOL_SIZE, self.config.POOL_SIZE, self.config.POOL_SIZE],
            name="roi_align_classifier"
        )([rpn_rois, input_image_meta] + mrcnn_feature_maps)

        # ✅ Classifier + BBox с ИСПРАВЛЕННЫМ динамическим reshape
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
            y=rois_aligned_classifier,
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
            train_bn=False
        )

        # ========== DETECTION LAYER ==========
        detections = DetectionLayer(
            self.config.BBOX_STD_DEV,
            float(self.config.DETECTION_MIN_CONFIDENCE),
            int(self.config.DETECTION_MAX_INSTANCES),
            float(self.config.DETECTION_NMS_THRESHOLD),
            int(self.config.IMAGES_PER_GPU),
            int(getattr(self.config, "BATCH_SIZE", self.config.IMAGES_PER_GPU)),
            name="mrcnn_detection"
        )([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        detection_boxes = KL.Lambda(lambda x: x[..., :6], name="detection_boxes")(detections)

        # ========== MASK HEAD ==========
        mask_aligned = PyramidROIAlign(
            [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
            name="roi_align_mask"
        )([detection_boxes, input_image_meta] + mrcnn_feature_maps)

        mrcnn_mask = build_fpn_mask_graph(
            y=mask_aligned,
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,
            train_bn=False
        )

        # --- Основная инференс-модель ---
        infer_inputs = [input_image, input_image_meta, input_anchors]
        infer_outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois]
        self.keras_model = KM.Model(infer_inputs, infer_outputs, name="mask_rcnn_inference")

        # --- eval models остаются как есть ---
        try:
            self.keras_infer_core = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [detections, mrcnn_feature_maps[0], mrcnn_feature_maps[1],
                 mrcnn_feature_maps[2], mrcnn_feature_maps[3]],
                name="mrcnn_infer_core"
            )
        except Exception as e:
            print("[infer_core] build failed:", e)
            self.keras_infer_core = None

        return self.keras_model

    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        self.keras_model.metrics_tensors = []

        # Use Adam by default for better convergence
        opt_name = str(self.config.OPTIMIZER.get("name", "SGD")).upper()
        oparams = _keras_opt_params(self.config.OPTIMIZER.get("parameters", {}))

        if opt_name == "SGD":
            optimizer = keras.optimizers.SGD(**oparams)
        elif opt_name == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**oparams)
        else:
            optimizer = keras.optimizers.Adam(**oparams)

        # Add Losses
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        if self.config.LEARNING_LAYERS == "rpn":
            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]
        elif self.config.LEARNING_LAYERS == "heads":
            loss_names = ["mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        elif self.config.LEARNING_LAYERS == "all":
            loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.)
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Add metrics for monitoring
        def mean_loss(y_true, y_pred):
            return K.mean(y_pred)

        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs),
            metrics={'mrcnn_class_loss': mean_loss, 'mrcnn_bbox_loss': mean_loss,
                     'mrcnn_mask_loss': mean_loss} if "heads" in self.config.LEARNING_LAYERS else None
        )

    def train(self):
        assert self.config.MODE == "training", "Create model in training mode."

        # Split dataset into train/val (80/20)
        train_ids = self.train_dataset.image_ids
        np.random.shuffle(train_ids)
        split = int(0.8 * len(train_ids))
        train_ids, val_ids = train_ids[split:], train_ids[:split]

        # Generator with z-score normalization
        def normalize_volume(vol):
            mu, sigma = np.mean(vol), np.std(vol)
            return (vol - mu) / sigma if sigma > 0 else vol

        class NormalizedMrcnnGenerator(MrcnnGenerator):
            def load_image_gt(self, dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = super().load_image_gt(dataset, config, image_id,
                                                                                            augment, augmentation,
                                                                                            use_mini_mask)
                image = normalize_volume(image)
                return image, image_meta, gt_class_ids, gt_boxes, gt_masks

        train_generator = NormalizedMrcnnGenerator(dataset=self.train_dataset.subset(train_ids), config=self.config)
        val_generator = NormalizedMrcnnGenerator(dataset=self.train_dataset.subset(val_ids), config=self.config)

        # Callbacks
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='MRCNN')
        early_stop = TF1EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = TF1ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)
        training_metrics = HeadTrainingMetricsCallback(
            self.keras_model,
            self.config,
            self.train_dataset.subset(val_ids),
            check_every=1
        )

        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Workers=0 for stability
        workers = 0

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(train_ids),
            callbacks=[save_weights, early_stop, reduce_lr, training_metrics],
            validation_data=val_generator,
            validation_steps=len(val_ids),
            max_queue_size=1,
            workers=workers,
            use_multiprocessing=False,
            verbose=1
        )

    def _refine_detections_numpy(self, rpn_rois_batch, mrcnn_class, mrcnn_bbox, image_meta,
                                 min_conf=None, nms_thr=None, max_inst=None):
        """
        ✅ УЛУЧШЕНО: добавлена диагностика того, насколько bbox-refinement меняет форму
        """
        import numpy as np
        from core import utils

        cfg = self.config
        if min_conf is None:
            min_conf = float(getattr(cfg, "DETECTION_MIN_CONFIDENCE", 0.1) or 0.1)
        if nms_thr is None:
            nms_thr = float(getattr(cfg, "DETECTION_NMS_THRESHOLD", 0.3) or 0.3)
        if max_inst is None:
            max_inst = int(getattr(cfg, "DETECTION_MAX_INSTANCES", 100) or 100)

        rois_nm = rpn_rois_batch[0] if rpn_rois_batch.ndim == 3 else rpn_rois_batch
        assert rois_nm.ndim == 2 and rois_nm.shape[1] == 6, "rois must be [N,6] normalized"

        cls = mrcnn_class[0] if mrcnn_class.ndim == 3 else mrcnn_class
        assert cls.ndim == 2, "mrcnn_class must be [N,C]"
        N, C = cls.shape
        if C <= 1:
            return np.zeros((0, 8), dtype=np.float32)

        bb = mrcnn_bbox
        if bb.ndim == 4:
            bb = bb[0]
        if bb.ndim == 2 and bb.shape[1] == C * 6:
            bb = bb.reshape((N, C, 6))
        assert bb.ndim == 3 and bb.shape[1] == C and bb.shape[2] == 6, "mrcnn_bbox must be [N,C,6]"

        # Softmax
        logits = cls.astype(np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)

        # Лучший FG класс
        probs_fg = probs[:, 1:]
        best_rel = np.argmax(probs_fg, axis=1)
        class_ids = best_rel + 1
        scores = probs[np.arange(N), class_ids]

        # ✅ ДИАГНОСТИКА: распределение scores ДО фильтрации
        print(f"\n[REFINE] BEFORE filter: {N} ROIs, "
              f"scores: mean={np.mean(scores):.3f}, "
              f"max={np.max(scores):.3f}, "
              f">0.5: {np.sum(scores > 0.5)}, "
              f">{min_conf}: {np.sum(scores >= min_conf)}")

        # Фильтр по confidence
        keep = scores >= float(min_conf)
        if not np.any(keep):
            print(f"[REFINE] SKIP: No ROIs above threshold {min_conf:.3f}")
            return np.zeros((0, 8), dtype=np.float32)

        rois_nm = rois_nm[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]
        deltas = bb[keep, :, :][np.arange(np.sum(keep)), class_ids, :]

        # ✅ ДИАГНОСТИКА: насколько сильные дельты?
        delta_magnitude = np.abs(deltas).mean(axis=1)
        print(f"[REFINE] Delta magnitude: mean={np.mean(delta_magnitude):.3f}, "
              f"max={np.max(delta_magnitude):.3f}, "
              f">0.5: {np.sum(delta_magnitude > 0.5)}")

        # Применяем дельты
        img_shape = cfg.IMAGE_SHAPE[:3]
        rois_px = utils.denorm_boxes(rois_nm, img_shape).astype(np.float32)
        std = np.array(cfg.BBOX_STD_DEV, dtype=np.float32).reshape((1, 6))
        refined_px = utils.apply_box_deltas_3d(rois_px, deltas.astype(np.float32), std)

        # ✅ ДИАГНОСТИКА: насколько изменились боксы?
        size_before = (rois_px[:, 3] - rois_px[:, 0]) * (rois_px[:, 4] - rois_px[:, 1]) * (
                    rois_px[:, 5] - rois_px[:, 2])
        size_after = (refined_px[:, 3] - refined_px[:, 0]) * (refined_px[:, 4] - refined_px[:, 1]) * (
                    refined_px[:, 5] - refined_px[:, 2])
        size_change_ratio = size_after / np.maximum(size_before, 1.0)

        print(f"[REFINE] Box size change: mean_ratio={np.mean(size_change_ratio):.3f}, "
              f"shrink(<0.9): {np.sum(size_change_ratio < 0.9)}, "
              f"grow(>1.1): {np.sum(size_change_ratio > 1.1)}")

        # Клип в окно
        H, W, D = img_shape
        refined_px[:, 0] = np.clip(refined_px[:, 0], 0, H)
        refined_px[:, 1] = np.clip(refined_px[:, 1], 0, W)
        refined_px[:, 2] = np.clip(refined_px[:, 2], 0, D)
        refined_px[:, 3] = np.clip(refined_px[:, 3], 0, H)
        refined_px[:, 4] = np.clip(refined_px[:, 4], 0, W)
        refined_px[:, 5] = np.clip(refined_px[:, 5], 0, D)

        # Min size
        min_size = 2.0
        refined_px[:, 3] = np.maximum(refined_px[:, 0] + min_size, refined_px[:, 3])
        refined_px[:, 4] = np.maximum(refined_px[:, 1] + min_size, refined_px[:, 4])
        refined_px[:, 5] = np.maximum(refined_px[:, 2] + min_size, refined_px[:, 5])

        # NMS
        try:
            keep_idx = utils.nms_3d(refined_px.astype(np.float32), scores.astype(np.float32), nms_thr)
        except Exception:
            keep_idx = np.argsort(-scores)

        keep_idx = keep_idx[:max_inst]

        print(f"[REFINE] AFTER NMS: {len(keep_idx)} detections (from {len(scores)})")

        refined_px = refined_px[keep_idx]
        class_ids = class_ids[keep_idx]
        scores = scores[keep_idx]

        # Назад в нормализованные
        refined_nm = utils.norm_boxes(refined_px, img_shape).astype(np.float32)

        detections = np.zeros((refined_nm.shape[0], 8), dtype=np.float32)
        detections[:, :6] = refined_nm
        detections[:, 6] = class_ids.astype(np.float32)
        detections[:, 7] = scores.astype(np.float32)

        return detections

    def process_image(self, image_id, generator, result_dir, roi_probe=None, cls_probe=None, det_idx=0, mask_idx=3):
        """
        Process a single image for evaluation (без сохранения .npy; TIFF как (D,H,W) -> 12 слайдов).
        """
        import os, numpy as np
        from skimage.io import imsave

        # Входы
        name, inputs = generator.get_input_prediction(image_id)
        input_image = inputs[0][0]  # (H,W,D,1)

        # Пробы (только лог; без .npy)
        roiN = None
        maxClsProb = None
        if roi_probe:
            try:
                rois = roi_probe.predict(inputs, batch_size=1, verbose=0)[0]  # (R,6) norm
                roiN = int(rois.shape[0])
                print(f"[DEBUG] {name}: {roiN} proposals")
                # sanity: IoU с GT
                try:
                    gt_boxes, _, _ = self.test_dataset.load_data(image_id)
                    if gt_boxes is not None and gt_boxes.size > 0:
                        from keras import backend as K
                        image_shape = self.config.IMAGE_SHAPE[:3]
                        gt_boxes_norm = utils.norm_boxes(gt_boxes.astype(np.float32), image_shape)
                        overlaps = overlaps_graph(rois, gt_boxes_norm)  # tf.Tensor в TF1
                        overlaps = K.get_value(overlaps)
                        if isinstance(overlaps, np.ndarray) and overlaps.size:
                            mean_iou = float(np.mean(np.max(overlaps, axis=1)))
                        else:
                            mean_iou = 0.0
                        print(f"[DEBUG] {name} proposals mean IoU with GT: {mean_iou:.3f}")
                except Exception as e:
                    print(f"[DEBUG] IoU computation failed: {e}")
            except Exception as e:
                print(f"[WARN] ROI probe failed: {e}")

        if cls_probe:
            try:
                cls = cls_probe.predict(inputs, batch_size=1, verbose=0)[0]
                print(f"[DEBUG] {name}: cls shape: {tuple(cls.shape)}")# (R,C)
                # для бинарного кейса берём столбец класса 1
                maxClsProb = float(np.max(cls[:, 1])) if cls.ndim == 2 and cls.shape[1] > 1 and cls.size else 0.0
                print(f"[DEBUG] {name}: max class prob: {maxClsProb:.3f}")
            except Exception as e:
                print(f"[WARN] CLS probe failed: {e}")

        # Предсказание моделью
        outputs = self.keras_model.predict(inputs, verbose=0)
        detections = outputs[det_idx]  # (1,N,8) норм. [y1,x1,z1,y2,x2,z2, cls, score]
        mrcnn_mask = outputs[mask_idx]  # (1,N,maskH,maskW,maskD)

        # Отладка «сырых» скорингов
        if detections.size:
            # detections[0,:,7] — score
            max_raw_score = float(np.max(detections[0, :, 7])) if detections.shape[-1] > 7 else 0.0
        else:
            max_raw_score = 0.0
        print(f"[DEBUG] {name}: Max raw detection score: {max_raw_score:.3f}")

        # Разбор детекций
        pd_boxes, pd_scores, pd_class_ids, pd_masks, pd_segs = self.unmold_detections(detections[0], mrcnn_mask[0])
        pred_inst = int(pd_boxes.shape[0])
        print(f"[DEBUG] {name}: {pred_inst} detections after unmold")

        # GT для метрик
        try:
            gt_boxes, gt_class_ids, gt_masks = self.test_dataset.load_data(image_id, masks_needed=True)
            print(f"[DEBUG] {name}: GT boxes: {gt_boxes.shape[0] if gt_boxes is not None else 0}")
        except TypeError:
            gt_boxes, gt_class_ids, gt_masks = self.test_dataset.load_data(image_id)
            print(f"[DEBUG] {name}: GT boxes (fallback): {gt_boxes.shape[0] if gt_boxes is not None else 0}")
        if gt_boxes is not None and getattr(gt_boxes, "size", 0) > 0 and \
                gt_masks is not None and getattr(gt_masks, "ndim", 0) == 4:
            Ng = int(gt_boxes.shape[0])
            if gt_masks.shape[-1] != Ng:
                Ng2 = min(Ng, gt_masks.shape[-1])
                gt_boxes = gt_boxes[:Ng2]
                gt_class_ids = gt_class_ids[:Ng2]
                gt_masks = gt_masks[..., :Ng2]

            # Pred: длины boxes/scores/class_ids/masks должны совпадать
        Kp = int(pd_boxes.shape[0])
        if pd_scores.shape[0] != Kp or pd_class_ids.shape[0] != Kp or \
                (pd_masks is not None and getattr(pd_masks, "ndim", 0) == 4 and pd_masks.shape[-1] != Kp):
            Kp2 = min(
                Kp,
                pd_scores.shape[0],
                pd_class_ids.shape[0],
                (pd_masks.shape[-1] if (pd_masks is not None and getattr(pd_masks, "ndim", 0) == 4 and pd_masks.shape[
                    -1] > 0) else Kp)
            )
            pd_boxes = pd_boxes[:Kp2]
            pd_scores = pd_scores[:Kp2]
            pd_class_ids = pd_class_ids[:Kp2]
            if pd_masks is not None and getattr(pd_masks, "ndim", 0) == 4:
                pd_masks = pd_masks[..., :Kp2]
            pred_inst = int(Kp2)
        # Метрики (надёжная обработка типов + пустых кейсов)
        if (gt_masks is None or getattr(gt_masks, "size", 0) == 0 or gt_masks.shape[-1] == 0) and pred_inst == 0:
            ap50 = prec50 = rec50 = miou = pxP = pxR = pxF1 = 0.0
            dice_mean = dice_std = 0.0
            dice_n = 0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ap50, prec50, rec50, ious = compute_ap(
                    gt_boxes, gt_class_ids, gt_masks,
                    pd_boxes, pd_class_ids, pd_scores, pd_masks,
                    iou_threshold=0.5
                )
                # compute_ap может вернуть ious как list → приводим к ndarray
                try:
                    ious = np.asarray(ious, dtype=np.float32)
                except Exception:
                    ious = np.array([], dtype=np.float32)
                miou = float(np.nanmean(ious)) if ious.size else 0.0

            # пиксельные метрики (если есть сегментации)
            if (gt_masks is not None and getattr(gt_masks, "size", 0) > 0 and
                    pd_segs is not None and getattr(pd_segs, "size", 0) > 0):
                # бинализация: предсказание — «любая инстанс-метка > 0», GT — «хоть один канал > 0»
                pred_bin = (pd_segs > 0)
                gt_bin = ((gt_masks > 0).any(axis=-1))
                pxP, pxR, pxF1 = self._pixelwise_metrics(pred_bin, gt_bin)
                pxP, pxR, pxF1 = float(pxP), float(pxR), float(pxF1)
            else:
                pxP = pxR = pxF1 = 0.0

            # instance-level Dice c IoU-матчингом
            dice_mean, dice_std, dice_n = self._instance_dice(pd_masks, gt_masks, iou_thr=0.5)

        # Визуализация/сохранение (TIFF как (D,H,W) → 12 слоёв; .npy НЕ сохраняем)
        try:
            # Гарантируем сохранение TIFF даже при отсутствии детекций
            if pd_segs is None or getattr(pd_segs, "size", 0) == 0:
                H, W, D = input_image.shape[:3]
                empty_segs = np.zeros((H, W, D), dtype=np.uint8)
                seg_stack = np.moveaxis(empty_segs, -1, 0)  # (D,H,W)
            else:
                seg_stack = np.moveaxis(pd_segs.astype(np.uint8), -1, 0)  # (D,H,W)

            imsave(os.path.join(result_dir, f"{name}.tiff"), seg_stack, check_contrast=False)

            # CSV с боксами/классами сохраняем всегда (даже пустой)
            self.save_classes_and_boxes(pd_class_ids, pd_boxes, name)
            print(f"[DEBUG] Outputs saved for {name}")
        except Exception as e:
            print(f"[WARN] Save outputs failed for {name}: {e}")

        return [name, pred_inst, ap50, prec50, rec50, miou, pxP, pxR, pxF1,
                float(dice_mean), float(dice_std), int(dice_n),
                (roiN if roiN is not None else -1), (maxClsProb if maxClsProb is not None else -1)]

    def _pixelwise_metrics(self, pred_bin, gt_bin):
        """Calculate pixelwise precision, recall, F1, and IoU."""
        tp = np.logical_and(pred_bin, gt_bin).sum(dtype=np.int64)
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum(dtype=np.int64)
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum(dtype=np.int64)

        prec = (tp / (tp + fp + 1e-9)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn + 1e-9)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec + 1e-9)) if (prec + rec) > 0 else 0.0
        iou = (tp / (tp + fp + fn + 1e-9)) if (tp + fp + fn) > 0 else 0.0

        return float(prec), float(rec), float(f1), float(iou)

    def _instance_dice(self, pred_masks, gt_masks, iou_thr=0.5):
        """Calculate instance-level DICE score with IoU-based matching + DETAILED LOGGING."""
        K = int(pred_masks.shape[-1]) if pred_masks is not None and pred_masks.ndim == 4 else 0
        G = int(gt_masks.shape[-1]) if gt_masks is not None and gt_masks.ndim == 4 else 0

        if K == 0 or G == 0:
            print(f"[INSTANCE] No masks to match (K={K}, G={G})")
            return 0.0, 0.0, 0

        pred_bin = (pred_masks > 0.5) if pred_masks.dtype != np.bool_ else pred_masks
        gt_bin = (gt_masks > 0.5) if gt_masks.dtype != np.bool_ else gt_masks

        P = pred_bin.reshape((-1, K)).astype(np.bool_)
        T = gt_bin.reshape((-1, G)).astype(np.bool_)
        print(f"[DEBUG RESHAPE]")
        print(f"  pred_bin shape before reshape: {pred_bin.shape}")
        print(f"  gt_bin shape before reshape: {gt_bin.shape}")
        print(f"  P shape after reshape: {P.shape}")
        print(f"  T shape after reshape: {T.shape}")
        print(f"  P[:, 0] sum: {P[:, 0].sum()}")
        print(f"  T[:, 0] sum: {T[:, 0].sum()}")
        inter_test = np.logical_and(P[:, 0], T[:, 0]).sum()
        print(f"  P[:, 0] AND T[:, 0]: {inter_test} pixels")
        print(f"\n[AXIS DEBUG]")
        print(f"  pred_masks shape: {pred_masks.shape}")
        print(f"  gt_masks shape: {gt_masks.shape}")
        print(f"  P reshaped: {P.shape}")
        print(f"  T reshaped: {T.shape}")

        # Проверяем есть ли вообще пересечение первых двух масок
        if K > 0 and G > 0:
            pred_0 = pred_masks[..., 0] > 0.5
            gt_0 = gt_masks[..., 0] > 0.5
            intersection = np.logical_and(pred_0, gt_0).sum()
            print(f"  Direct test (pred#0 ∩ gt#0): {intersection} pixels")

            if intersection == 0:
                print(f"  ⚠️  NO INTERSECTION! Checking axis order...")
                # Пробуем разные перестановки
                for perm_name, perm in [("ZYX", (2, 0, 1)), ("XYZ", (1, 0, 2)), ("XZY", (1, 2, 0))]:
                    try:
                        gt_perm = np.transpose(gt_masks[..., 0], perm) > 0.5
                        if gt_perm.shape != pred_0.shape:
                            continue
                        test_inter = np.logical_and(pred_0, gt_perm).sum()
                        if test_inter > 0:
                            print(f"    ✅ {test_inter} pixels with GT order: {perm_name} {perm}")
                    except:
                        pass
        P_sum = P.sum(axis=0).astype(np.int64)
        T_sum = T.sum(axis=0).astype(np.int64)
        inter = (P.T @ T).astype(np.int64)
        union = (P_sum[:, None] + T_sum[None, :]) - inter

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = inter / np.maximum(union, 1)

        # ✅ НОВОЕ: детальное логирование IoU матрицы
        print(f"\n{'=' * 70}")
        print(f"[INSTANCE MATCHING] Pred={K} masks, GT={G} masks, threshold={iou_thr}")
        print(f"{'=' * 70}")

        # Показываем лучшие IoU для каждой предсказанной маски
        for k in range(min(K, 10)):
            best_g = np.argmax(iou[k, :])
            best_iou = iou[k, best_g]
            above_thr = np.sum(iou[k, :] >= iou_thr)
            status = "✅" if best_iou >= iou_thr else "❌"
            print(f"  {status} Pred#{k}: best_IoU={best_iou:.3f} (GT#{best_g}), "
                  f"size={P_sum[k]:>6}, matches_above_thr={above_thr}")

        if K > 10:
            print(f"  ... +{K - 10} more pred masks")

        # Matching
        used_p = np.zeros((K,), dtype=np.bool_)
        used_g = np.zeros((G,), dtype=np.bool_)
        dices = []

        print(f"\n[MATCHING PROCESS]")
        iteration = 0
        while True:
            iou_mask = iou.copy()
            iou_mask[used_p, :] = -1.0
            iou_mask[:, used_g] = -1.0
            k, g = np.unravel_index(np.argmax(iou_mask), iou_mask.shape)

            if iou_mask[k, g] < iou_thr:
                print(f"  Iter {iteration}: STOP (best remaining IoU={iou_mask[k, g]:.3f})")
                break

            denom = P_sum[k] + T_sum[g]
            d = (2.0 * inter[k, g] / denom) if denom > 0 else 0.0
            dices.append(float(d))

            print(f"  Iter {iteration}: pred#{k} ↔ GT#{g}, IoU={iou_mask[k, g]:.3f}, DICE={d:.3f}")

            used_p[k] = True
            used_g[g] = True
            iteration += 1

        # Итоги
        unmatched_p = K - used_p.sum()
        unmatched_g = G - used_g.sum()
        print(f"\n[RESULT] Matched={len(dices)}, Unmatched_pred={unmatched_p}, Missed_GT={unmatched_g}")

        if len(dices) == 0:
            print(f"⚠️  WARNING: NO MATCHES! Попробуйте снизить iou_thr (сейчас {iou_thr})")
            print(f"{'=' * 70}\n")
            return 0.0, 0.0, 0

        mean_dice = float(np.mean(dices))
        std_dice = float(np.std(dices))
        print(f"✅ Instance DICE: {mean_dice:.3f} ± {std_dice:.3f}")
        print(f"{'=' * 70}\n")

        return mean_dice, std_dice, int(len(dices))

    def _print_metrics_summary(self, name, metrics):
        """Печатает 4 ключевые метрики для каждого TIFF файла."""
        print(f"\n{'=' * 70}")
        print(f"📊 МЕТРИКИ: {name}.tiff")
        print(f"{'=' * 70}")

        # 1. Pixelwise
        if 'pixelwise' in metrics:
            prec, rec, f1, iou = metrics['pixelwise']
            print(f"1️⃣  PIXELWISE")
            print(f"   IoU:  {iou:.4f}")
            print(f"   F1:   {f1:.4f}  (prec={prec:.3f}, rec={rec:.3f})")

        # 2. Instance DICE
        if 'instance_dice' in metrics:
            mean_dice, std_dice, n_matched = metrics['instance_dice']
            print(f"\n2️⃣  INSTANCE DICE")
            print(f"   Mean: {mean_dice:.4f} ± {std_dice:.4f}  (n={n_matched} matches)")

        # 3. Detection
        if 'detection_performance' in metrics:
            det = metrics['detection_performance']
            print(f"\n3️⃣  DETECTION")
            print(f"   F1:   {det.get('f1', 0):.4f}")
            print(f"   Prec: {det.get('precision', 0):.4f}")
            print(f"   Rec:  {det.get('recall', 0):.4f}")

        print(f"{'=' * 70}\n")

    def save_classes_and_boxes(self, pd_class_ids, pd_boxes, name):
        import os
        cab_df = pd.DataFrame(
            {
                "class": [],
                "y1": [],
                "x1": [],
                "z1": [],
                "y2": [],
                "x2": [],
                "z2": [],
            }
        )

        for i in range(pd_boxes.shape[0]):
            y1, x1, z1, y2, x2, z2 = pd_boxes[i]
            class_id = pd_class_ids[i]
            cab_df.loc[len(cab_df)] = [class_id, y1, x1, z1, y2, x2, z2]

        # Гарантируем существование каталога и корректный join пути
        out_dir = getattr(self.config, "OUTPUT_DIR", "./")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{name}.csv")
        cab_df.to_csv(out_csv, index=False)

    def evaluate(self):
        import os, numpy as np, keras, pandas as pd
        from tqdm import tqdm
        from skimage.io import imsave
        from core.data_generators import MrcnnGenerator

        # === Вспомогательные функции ===
        all_confidence_scores = []
        confidence_histogram = {
            '0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0,
            '0.4-0.5': 0, '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0,
            '0.8-0.9': 0, '0.9-1.0': 0
        }
        def _draw_masks_overlay(name, image, pd_masks, gt_masks, pd_boxes, gt_boxes,
                                out_dir, pd_scores=None, metrics_dict=None):
            """Создаёт оверлей с МАСКАМИ (не боксами!) и метриками."""
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import matplotlib.patches as mpatches
            from skimage.segmentation import find_boundaries
            import matplotlib.colors as mcolors

            # Загружаем RAW изображение
            try:
                img_id = None
                for iid in self.test_dataset.image_ids:
                    info = self.test_dataset.image_info[iid]
                    img_name = info["path"].split("/")[-1].rsplit(".", 1)[0]
                    if img_name == name:
                        img_id = iid
                        break

                if img_id is not None:
                    img_path = self.test_dataset.image_info[img_id]["path"]
                    from skimage.io import imread
                    img_raw = imread(img_path)

                    if img_raw.ndim == 3:
                        img = img_raw
                    elif img_raw.ndim == 4:
                        img = img_raw[..., 0]
                    else:
                        img = image[..., 0] if image.ndim == 4 else image
                else:
                    img = image[..., 0] if image.ndim == 4 else image
            except Exception as e:
                print(f"[WARN] Raw image load failed: {e}")
                img = image[..., 0] if image.ndim == 4 else image

            # Переставляем оси если нужно
            if img.ndim == 3:
                if img.shape[0] < 20:
                    img = np.moveaxis(img, 0, -1)
                elif img.shape[1] < 20:
                    img = np.moveaxis(img, 1, -1)

            # MIP проекция
            mip = np.max(img, axis=2) if img.ndim == 3 else img
            mip = mip.astype("float32")

            # Процентильная нормализация
            lo, hi = float(np.percentile(mip, 1)), float(np.percentile(mip, 99))
            if hi > lo:
                mip = np.clip((mip - lo) / (hi - lo), 0, 1)

            # === Создаём 2x2 сетку подграфиков ===
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))

            # === 1. GT MASKS (верхний левый) ===
            ax = axes[0, 0]
            ax.imshow(mip, cmap='gray')
            ax.set_title(f'Ground Truth Masks ({gt_masks.shape[-1] if gt_masks is not None else 0} cells)',
                         fontsize=14, weight='bold')
            ax.axis('off')

            if gt_masks is not None and gt_masks.ndim == 4:
                # Создаем композитную маску для GT
                gt_composite = np.zeros((*mip.shape, 3))
                colors_gt = plt.cm.tab20(np.linspace(0, 1, gt_masks.shape[-1]))

                for i in range(gt_masks.shape[-1]):
                    # MIP проекция маски
                    mask_mip = np.max(gt_masks[..., i], axis=2) if gt_masks.ndim == 4 else gt_masks[..., i]
                    mask_bin = mask_mip > 0.5

                    # Границы маски
                    boundaries = find_boundaries(mask_bin, mode='thick')

                    for c in range(3):
                        gt_composite[mask_bin, c] = colors_gt[i, c] * 0.3  # Полупрозрачная заливка
                        gt_composite[boundaries, c] = colors_gt[i, c]  # Яркие границы

                # Накладываем маски на изображение
                overlay = mip[..., np.newaxis] * 0.7 + gt_composite * 0.3
                ax.imshow(overlay)

            # === 2. PREDICTED MASKS (верхний правый) ===
            ax = axes[0, 1]
            ax.imshow(mip, cmap='gray')

            n_pred_masks = pd_masks.shape[-1] if pd_masks is not None and pd_masks.ndim == 4 else 0
            ax.set_title(f'Predicted Masks ({n_pred_masks} cells)', fontsize=14, weight='bold')
            ax.axis('off')

            if pd_masks is not None and pd_masks.ndim == 4:
                # Создаем композитную маску для предсказаний
                pd_composite = np.zeros((*mip.shape, 3))

                # Сортируем маски по score если есть
                if pd_scores is not None and len(pd_scores) == pd_masks.shape[-1]:
                    sorted_idx = np.argsort(pd_scores)[::-1]
                else:
                    sorted_idx = np.arange(pd_masks.shape[-1])

                colors_pd = plt.cm.tab20(np.linspace(0, 1, len(sorted_idx)))

                for idx, i in enumerate(sorted_idx):
                    # MIP проекция маски
                    mask_mip = np.max(pd_masks[..., i], axis=2) if pd_masks.ndim == 4 else pd_masks[..., i]
                    if pd_masks.ndim == 4:
                        mask_mip = np.max(pd_masks[:, :, :, i], axis=2)  # [H, W]
                    elif pd_masks.ndim == 3:
                        # Формат [H, W, D] - одна маска
                        mask_mip = np.max(pd_masks, axis=2)
                    else:
                        continue
                    mask_bin = mask_mip > 0.5

                    # Границы маски
                    boundaries = find_boundaries(mask_bin, mode='thick')

                    # Цвет по score
                    if pd_scores is not None and i < len(pd_scores):
                        score = pd_scores[i]
                        if score > 0.7:
                            color_multiplier = 1.0
                        elif score > 0.5:
                            color_multiplier = 0.7
                        else:
                            color_multiplier = 0.4
                    else:
                        color_multiplier = 0.7

                    for c in range(3):
                        pd_composite[mask_bin, c] = np.maximum(
                            pd_composite[mask_bin, c],
                            colors_pd[idx, c] * 0.3 * color_multiplier
                        )
                        pd_composite[boundaries, c] = np.maximum(
                            pd_composite[boundaries, c],
                            colors_pd[idx, c] * color_multiplier
                        )

                # Накладываем маски на изображение
                overlay = mip[..., np.newaxis] * 0.7 + pd_composite * 0.3
                ax.imshow(overlay)

                # Добавляем легенду с scores
                if pd_scores is not None:
                    score_text = f"Scores: min={np.min(pd_scores):.2f}, max={np.max(pd_scores):.2f}, mean={np.mean(pd_scores):.2f}"
                    ax.text(0.02, 0.98, score_text, transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=10, verticalalignment='top')

            # === 3. OVERLAY GT + PRED (нижний левый) ===
            ax = axes[1, 0]
            ax.imshow(mip, cmap='gray')
            ax.set_title('Mask Overlay: GT(green) + Pred(red/yellow)', fontsize=14, weight='bold')
            ax.axis('off')

            overlay_composite = np.zeros((*mip.shape, 3))

            # GT маски в зеленом канале
            if gt_masks is not None and gt_masks.ndim == 4:
                gt_union = np.any(gt_masks > 0.5, axis=-1)
                gt_union_mip = np.max(gt_union, axis=2) if gt_union.ndim == 3 else gt_union
                gt_boundaries = find_boundaries(gt_union_mip, mode='thick')
                overlay_composite[gt_union_mip, 1] = 0.5  # Зеленая заливка
                overlay_composite[gt_boundaries, 1] = 1.0  # Яркие зеленые границы

            # Predicted маски с цветом по IoU
            if pd_masks is not None and pd_masks.ndim == 4 and gt_masks is not None:
                # Вычисляем IoU для каждой предсказанной маски
                from core.utils import compute_overlaps_3d

                for i in range(pd_masks.shape[-1]):
                    mask_mip = np.max(pd_masks[..., i], axis=2) if pd_masks.ndim == 4 else pd_masks[..., i]
                    mask_bin = mask_mip > 0.5
                    boundaries = find_boundaries(mask_bin, mode='thick')

                    # Определяем цвет по IoU с ближайшей GT маской
                    best_iou = 0.0
                    if pd_boxes is not None and gt_boxes is not None and len(pd_boxes) > i and len(gt_boxes) > 0:
                        ious = compute_overlaps_3d(pd_boxes[i:i + 1], gt_boxes)
                        best_iou = np.max(ious) if ious.size > 0 else 0.0

                    # Цветовая схема по IoU
                    if best_iou > 0.7:
                        color = [0, 1, 1]  # Cyan
                    elif best_iou > 0.5:
                        color = [1, 1, 0]  # Yellow
                    elif best_iou > 0.3:
                        color = [1, 0.5, 0]  # Orange
                    else:
                        color = [1, 0, 0]  # Red

                    for c in range(3):
                        overlay_composite[mask_bin, c] = np.maximum(
                            overlay_composite[mask_bin, c],
                            color[c] * 0.3
                        )
                        overlay_composite[boundaries, c] = np.maximum(
                            overlay_composite[boundaries, c],
                            color[c] * 0.8
                        )

            # Накладываем композит на изображение
            final_overlay = mip[..., np.newaxis] * 0.6 + overlay_composite * 0.4
            ax.imshow(final_overlay)

            # Легенда
            legend_elements = [
                mpatches.Patch(color='green', label='GT masks'),
                mpatches.Patch(color='cyan', label='Pred IoU>0.7'),
                mpatches.Patch(color='yellow', label='Pred IoU>0.5'),
                mpatches.Patch(color='orange', label='Pred IoU>0.3'),
                mpatches.Patch(color='red', label='Pred IoU<0.3')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

            # === 4. METRICS PANEL (нижний правый) ===
            ax = axes[1, 1]
            ax.axis('off')

            # Формируем текст с метриками
            metrics_text = f"{'=' * 50}\n"
            metrics_text += f"IMAGE: {name}\n"
            metrics_text += f"{'=' * 50}\n\n"

            metrics_text += f"MASK DETECTION METRICS:\n"
            metrics_text += f"  GT masks: {gt_masks.shape[-1] if gt_masks is not None else 0}\n"
            metrics_text += f"  Predicted masks: {pd_masks.shape[-1] if pd_masks is not None else 0}\n"

            if metrics_dict:
                # Pixelwise метрики
                if 'pixelwise' in metrics_dict:
                    pw = metrics_dict['pixelwise']
                    metrics_text += f"\nPIXELWISE SEGMENTATION:\n"
                    metrics_text += f"  Precision: {pw[0]:.3f}\n"
                    metrics_text += f"  Recall: {pw[1]:.3f}\n"
                    metrics_text += f"  F1-score: {pw[2]:.3f}\n"

                # Instance DICE
                if 'instance_dice' in metrics_dict:
                    dice_mean, dice_std, dice_count = metrics_dict['instance_dice']
                    metrics_text += f"\nINSTANCE DICE (IoU>0.5 matching):\n"
                    metrics_text += f"  Mean DICE: {dice_mean:.3f}\n"
                    metrics_text += f"  Std DICE: {dice_std:.3f}\n"
                    metrics_text += f"  Matched instances: {dice_count}\n"

                # IoU метрики если есть боксы
                if 'iou_metrics' in metrics_dict:
                    m = metrics_dict['iou_metrics']
                    metrics_text += f"\nBOX IoU METRICS:\n"
                    metrics_text += f"  Mean IoU: {m.get('mean_iou', 0):.3f}\n"
                    metrics_text += f"  IoU>0.5: {m.get('iou_05', 0)}/{m.get('n_pred', 0)}\n"

                # Detection performance
                if 'detection_performance' in metrics_dict:
                    dp = metrics_dict['detection_performance']
                    metrics_text += f"\nDETECTION PERFORMANCE:\n"
                    metrics_text += f"  TP (matched): {dp['tp']}\n"
                    metrics_text += f"  FP (extra): {dp['fp']}\n"
                    metrics_text += f"  FN (missed): {dp['fn']}\n"
                    metrics_text += f"  Recall: {dp['recall']:.3f}\n"
                    metrics_text += f"  Precision: {dp['precision']:.3f}\n"

            # Статистика scores
            if pd_scores is not None and len(pd_scores) > 0:
                metrics_text += f"\nCONFIDENCE SCORES:\n"
                metrics_text += f"  Mean: {np.mean(pd_scores):.3f}\n"
                metrics_text += f"  Median: {np.median(pd_scores):.3f}\n"
                metrics_text += f"  Max: {np.max(pd_scores):.3f}\n"
                metrics_text += f"  Min: {np.min(pd_scores):.3f}\n"
                metrics_text += f"  >0.7: {np.sum(pd_scores > 0.7)}\n"
                metrics_text += f"  >0.5: {np.sum(pd_scores > 0.5)}\n"

            # Отображаем текст
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.suptitle(f'Cell Segmentation Results: {name}', fontsize=16, weight='bold')
            plt.tight_layout()

            # Сохранение
            os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
            overlay_path = os.path.join(out_dir, "overlays", f"{name}_masks_overlay.png")
            fig.savefig(overlay_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"[SAVED] Mask overlay: {overlay_path}")
            return metrics_dict

        def _compute_mask_metrics(pd_masks, gt_masks, pd_boxes=None, gt_boxes=None):
            """Вычисляет метрики по маскам."""
            metrics = {}

            # Pixelwise метрики
            if pd_masks is not None and gt_masks is not None:
                pd_bin = pd_masks.any(axis=-1) if pd_masks.ndim == 4 else pd_masks > 0
                gt_bin = gt_masks.any(axis=-1) if gt_masks.ndim == 4 else gt_masks > 0
                prec, rec, f1, iou = self._pixelwise_metrics(pd_bin, gt_bin)
                metrics['pixelwise'] = (prec, rec, f1, iou)
                print(f"[PIXELWISE] IoU={iou:.4f}, F1={f1:.4f}, Prec={prec:.3f}, Rec={rec:.3f}")

                # Instance DICE
                metrics['instance_dice'] = self._instance_dice(pd_masks, gt_masks, iou_thr=0.5)

                # Detection metrics на уровне масок
                n_pred = pd_masks.shape[-1] if pd_masks.ndim == 4 else 0
                n_gt = gt_masks.shape[-1] if gt_masks.ndim == 4 else 0

                # Простое сопоставление по IoU масок
                if n_pred > 0 and n_gt > 0:
                    # Вычисляем IoU между всеми парами масок
                    pd_flat = pd_masks.reshape((-1, n_pred))
                    gt_flat = gt_masks.reshape((-1, n_gt))

                    intersection = pd_flat.T @ gt_flat
                    pd_sum = pd_flat.sum(axis=0)
                    gt_sum = gt_flat.sum(axis=0)
                    union = pd_sum[:, None] + gt_sum[None, :] - intersection

                    mask_ious = intersection / np.maximum(union, 1)

                    # Matching с порогом 0.5
                    matched_pred = np.zeros(n_pred, dtype=bool)
                    matched_gt = np.zeros(n_gt, dtype=bool)

                    # Жадное сопоставление
                    while True:
                        # Находим лучшую пару
                        best_i, best_j = np.unravel_index(np.argmax(mask_ious), mask_ious.shape)
                        best_iou = mask_ious[best_i, best_j]

                        if best_iou < 0.5:
                            break

                        matched_pred[best_i] = True
                        matched_gt[best_j] = True

                        # Обнуляем использованные
                        mask_ious[best_i, :] = 0
                        mask_ious[:, best_j] = 0

                    tp = np.sum(matched_pred)
                    fp = n_pred - tp
                    fn = n_gt - np.sum(matched_gt)

                    metrics['detection_performance'] = {
                        'tp': int(tp),
                        'fp': int(fp),
                        'fn': int(fn),
                        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    }

            # Box IoU если есть
            if pd_boxes is not None and gt_boxes is not None and len(pd_boxes) > 0 and len(gt_boxes) > 0:
                from core.utils import compute_overlaps_3d
                ious = compute_overlaps_3d(pd_boxes, gt_boxes)
                max_ious = np.max(ious, axis=1) if ious.size > 0 else np.array([])

                if max_ious.size > 0:
                    metrics['iou_metrics'] = {
                        'mean_iou': np.mean(max_ious),
                        'iou_05': np.sum(max_ious > 0.5),
                        'n_pred': len(pd_boxes)
                    }

            return metrics

        def _names_and_indices():
            """Находит индексы выходов модели."""
            outs = getattr(self.keras_model, "outputs", [])
            names = [t.name.split("/")[0].split(":")[0] for t in outs]
            idx = {n: i for i, n in enumerate(names)}

            def _find_idx(patterns):
                for p in patterns:
                    for name, i in idx.items():
                        if p in name:
                            return i
                return None

            i_det = _find_idx(["mrcnn_detection", "detection"])
            i_mask = _find_idx(["mrcnn_mask", "mask"])
            i_cls = _find_idx(["mrcnn_class", "class"])
            i_rois = _find_idx(["rpn_rois", "ROI", "rois"])

            print(f"[DEBUG] outputs idx: det={i_det} mask={i_mask} cls={i_cls} rois={i_rois}")

            if i_det is None and len(outs) >= 4:
                i_det, i_cls, i_mask, i_rois = 0, 1, 3, 4
                print(f"[FALLBACK] Using default indices")

            return i_det, i_mask, i_cls, i_rois

        def _load_head_weights():
            """Загружает веса головы."""
            head_w = getattr(self.config, "HEAD_WEIGHTS", None)
            if not head_w:
                return
            try:
                self.keras_model.load_weights(head_w, by_name=True)
                print("[HEAD] loaded by_name")
            except Exception as e:
                print(f"[HEAD] load failed: {e}")

        def _to_pixels(boxes_norm, H, W, D):
            """Денормализация боксов."""
            if boxes_norm is None or boxes_norm.size == 0:
                return np.zeros((0, 6), np.int32)
            b = boxes_norm.copy()
            b[:, [0, 3]] *= H
            b[:, [1, 4]] *= W
            b[:, [2, 5]] *= D
            return np.rint(b).astype(np.int32)

        def _as2d(a):
            """Убирает batch dimension."""
            return a[0] if (a is not None and hasattr(a, "ndim") and a.ndim == 3) else a

        # === MAIN EVALUATION ===

        print("\n" + "=" * 70)
        print("MASK-BASED EVALUATION WITH FILTERING")
        print("=" * 70)

        keras.backend.set_learning_phase(0)
        assert self.config.MODE == "inference"

        i_det, i_mask, i_cls, i_rois = _names_and_indices()
        _load_head_weights()

        print(f"\n[PHASE CHECK] Keras learning_phase: {keras.backend.learning_phase()}")

        # Параметры фильтрации
        MIN_CONFIDENCE = float(getattr(self.config, "DETECTION_MIN_CONFIDENCE", 0.9))
        MIN_ROI_SIZE = int(getattr(self.config, "MIN_ROI_SIZE", 100))  # В пикселях
        NMS_THRESHOLD = float(getattr(self.config, "DETECTION_NMS_THRESHOLD", 0.3))

        print(f"\n[FILTER CONFIG]")
        print(f"  MIN_CONFIDENCE: {MIN_CONFIDENCE}")
        print(f"  MIN_ROI_SIZE: {MIN_ROI_SIZE} pixels")
        print(f"  NMS_THRESHOLD: {NMS_THRESHOLD}")

        # Generator
        gen = MrcnnGenerator(
            dataset=self.test_dataset,
            config=self.config,
            shuffle=False,
            batch_size=1,
            training=False
        )

        ids = list(self.test_dataset.image_ids)
        print(f"\nEvaluating {len(ids)} images")

        out_dir = getattr(self.config, "OUTPUT_DIR", "./results")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

        H, W, D = self.config.IMAGE_SHAPE[:3]

        # Метрики по всем изображениям
        all_metrics = {
            'pixelwise_f1': [],
            'instance_dice': [],
            'recall': [],
            'precision': []
        }

        # Статистика фильтрации
        filter_stats = {
            'total_raw': 0,
            'after_confidence': 0,
            'after_size': 0,
            'after_nms': 0
        }

        for image_id in tqdm(ids, desc="Evaluating"):
            try:
                name, inputs = gen.get_input_prediction(image_id)
                image = inputs[0][0]

                # Predict
                outs = self.keras_model.predict(inputs, verbose=0)

                # Диагностика cls_prob
                if i_cls is not None and len(outs) > i_cls:
                    cls_probs = outs[i_cls]
                    if cls_probs.ndim == 3:
                        cls_probs = cls_probs[0]
                    if cls_probs.shape[-1] >= 2:
                        fg_probs = cls_probs[:, 1]
                        print(f"\n[DEBUG] {name}: cls_prob stats: "
                              f"mean={np.mean(fg_probs):.4f} "
                              f"max={np.max(fg_probs):.4f} "
                              f"min={np.min(fg_probs):.4f}")
                        all_confidence_scores.extend(fg_probs.tolist())

                        for score in fg_probs:
                            if score < 0.1:
                                confidence_histogram['0.0-0.1'] += 1
                            elif score < 0.2:
                                confidence_histogram['0.1-0.2'] += 1
                            elif score < 0.3:
                                confidence_histogram['0.2-0.3'] += 1
                            elif score < 0.4:
                                confidence_histogram['0.3-0.4'] += 1
                            elif score < 0.5:
                                confidence_histogram['0.4-0.5'] += 1
                            elif score < 0.6:
                                confidence_histogram['0.5-0.6'] += 1
                            elif score < 0.7:
                                confidence_histogram['0.6-0.7'] += 1
                            elif score < 0.8:
                                confidence_histogram['0.7-0.8'] += 1
                            elif score < 0.9:
                                confidence_histogram['0.8-0.9'] += 1
                            else:
                                confidence_histogram['0.9-1.0'] += 1

                        print(f"[CONFIDENCE ANALYSIS] {name}")
                        print(f"  Total proposals: {len(fg_probs)}")
                        print(f"  Detections at thresholds:")
                        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
                            n = np.sum(fg_probs >= thr)
                            pct = 100.0 * n / len(fg_probs)
                            marker = "👈 CURRENT" if thr == MIN_CONFIDENCE else ""
                            print(f"    {thr:.2f}: {n:>5} ({pct:>5.1f}%) {marker}")
                # Детекции
                det = outs[i_det] if i_det is not None else outs[0]
                det = _as2d(det)

                pd_boxes_norm = np.zeros((0, 6), np.float32)
                pd_scores = np.zeros((0,), np.float32)
                pd_class_ids = np.zeros((0,), np.int32)

                if det is not None and det.size:
                    K = det.shape[1]
                    if K >= 7:
                        cls = det[:, 6].astype(np.int32)
                        keep = cls > 0
                        rows = det[keep] if np.any(keep) else det[:0]
                        if rows.size:
                            pd_boxes_norm = rows[:, :6].astype(np.float32)
                            pd_class_ids = rows[:, 6].astype(np.int32)
                            if K >= 8:
                                pd_scores = rows[:, 7].astype(np.float32)

                n_raw = len(pd_boxes_norm)
                filter_stats['total_raw'] += n_raw
                print(f"[INFO] {name}: {n_raw} raw detections from HEAD")

                if pd_boxes_norm.size == 0:
                    print(f"[INFO] {name}: No foreground detections")
                    continue

                # === ФИЛЬТР 1: CONFIDENCE ===
                if pd_scores.size > 0:
                    conf_mask = pd_scores >= MIN_CONFIDENCE
                    pd_boxes_norm = pd_boxes_norm[conf_mask]
                    pd_class_ids = pd_class_ids[conf_mask]
                    pd_scores = pd_scores[conf_mask]

                    n_after_conf = len(pd_boxes_norm)
                    filter_stats['after_confidence'] += n_after_conf
                    print(f"[FILTER 1/3] Confidence: {n_after_conf}/{n_raw} passed (>={MIN_CONFIDENCE:.2f})")

                if pd_boxes_norm.size == 0:
                    print(f"[INFO] {name}: No detections after confidence filter")
                    continue

                # Денормализация боксов для размерного фильтра
                pd_boxes_px = _to_pixels(pd_boxes_norm, H, W, D)
                print(f"[BBOX DEBUG] {name}:")
                for i in range(min(3, len(pd_boxes_px))):
                    y1, x1, z1, y2, x2, z2 = pd_boxes_px[i]
                    print(f"  Pred bbox #{i}: Y={y1}-{y2}, X={x1}-{x2}, Z={z1}-{z2}")
                # === ФИЛЬТР 2: РАЗМЕР (используем MIN_ROI_SIZE в пикселях) ===
                volumes_px = (pd_boxes_px[:, 3] - pd_boxes_px[:, 0]) * \
                             (pd_boxes_px[:, 4] - pd_boxes_px[:, 1]) * \
                             (pd_boxes_px[:, 5] - pd_boxes_px[:, 2])

                size_mask = volumes_px >= MIN_ROI_SIZE
                n_before_size = len(pd_boxes_norm)

                pd_boxes_norm = pd_boxes_norm[size_mask]
                pd_boxes_px = pd_boxes_px[size_mask]
                pd_class_ids = pd_class_ids[size_mask]
                pd_scores = pd_scores[size_mask]

                n_after_size = len(pd_boxes_norm)
                filter_stats['after_size'] += n_after_size
                print(f"[FILTER 2/3] Size: {n_after_size}/{n_before_size} passed (vol>={MIN_ROI_SIZE} px³)")

                if pd_boxes_norm.size == 0:
                    print(f"[INFO] {name}: No detections after size filter")
                    continue

                # === ФИЛЬТР 3: NMS ===
                if len(pd_boxes_norm) > 1:
                    from core.utils import compute_overlaps_3d

                    # Сортируем по score (высокие первыми)
                    sorted_idx = np.argsort(pd_scores)[::-1]
                    keep = []

                    while len(sorted_idx) > 0:
                        i = sorted_idx[0]
                        keep.append(i)

                        if len(sorted_idx) == 1:
                            break

                        # IoU текущего с остальными
                        ious = compute_overlaps_3d(
                            pd_boxes_norm[i:i + 1],
                            pd_boxes_norm[sorted_idx[1:]]
                        )[0]

                        # Оставляем только те, что слабо перекрываются
                        sorted_idx = sorted_idx[1:][ious < NMS_THRESHOLD]

                    n_before_nms = len(pd_boxes_norm)
                    keep = np.array(keep)

                    pd_boxes_norm = pd_boxes_norm[keep]
                    pd_boxes_px = pd_boxes_px[keep]
                    pd_class_ids = pd_class_ids[keep]
                    pd_scores = pd_scores[keep]

                    n_after_nms = len(pd_boxes_norm)
                    filter_stats['after_nms'] += n_after_nms
                    print(f"[FILTER 3/3] NMS: {n_after_nms}/{n_before_nms} kept (thr={NMS_THRESHOLD})")
                else:
                    filter_stats['after_nms'] += len(pd_boxes_norm)

                print(f"[INFO] {name}: {len(pd_boxes_norm)} FINAL detections after all filters\n")

                if pd_boxes_norm.size == 0:
                    print(f"[INFO] {name}: No detections after all filters")
                    continue

                # GT данные
                try:
                    gt_boxes, gt_class_ids, gt_masks = self.test_dataset.load_data(image_id)
                    gt_boxes = np.asarray(gt_boxes, dtype=np.int32) if gt_boxes is not None else np.zeros((0, 6),
                                                                                                          np.int32)

                    # ✅ ИСПРАВЛЕНИЕ ПОРЯДКА ОСЕЙ GT ДАННЫХ
                    if gt_boxes.size > 0:
                        # (y1,x1,z1,y2,x2,z2) → (y1,z1,x1,y2,z2,x2)
                        gt_boxes = gt_boxes[:, [0, 2, 1, 3, 5, 4]]
                        print(f"[GT FIX] Swapped X↔Z in gt_boxes")



                except:
                    gt_boxes = np.zeros((0, 6), np.int32)
                    gt_masks = None

                # === ГЕНЕРАЦИЯ МАСОК из HEAD ===
                pd_masks = None
                if i_mask is not None and pd_boxes_norm.size:
                    raw_masks = outs[i_mask]
                    raw_masks = raw_masks[0] if hasattr(raw_masks, "ndim") and raw_masks.ndim >= 5 else raw_masks

                    print(f"[DEBUG] raw_masks shape: {raw_masks.shape}")

                    seg = np.zeros((H, W, D), dtype=np.uint16)
                    pd_masks_list = []

                    for j in range(min(pd_boxes_px.shape[0], raw_masks.shape[0])):
                        cid = int(pd_class_ids[j]) if pd_class_ids.size else 1

                        try:
                            m_small = raw_masks[j, ..., cid]
                        except:
                            try:
                                m_small = raw_masks[j, cid, ...]
                            except:
                                m_small = raw_masks[j, ..., 0]

                        # Диагностика первой маски
                        if j == 0:
                            print(f"[DEBUG] mask {j}: shape={m_small.shape}, "
                                  f"mean={np.mean(m_small):.3f}, "
                                  f"max={np.max(m_small):.3f}, "
                                  f"min={np.min(m_small):.3f}")

                        # 👇 ДИАГНОСТИКА BBOX 👇
                        print(f"\n[BBOX COMPARISON] {name}")
                        print(f"Predicted bbox (normalized):")
                        for i in range(min(3, len(pd_boxes_norm))):
                            print(f"  Pred#{i}: {pd_boxes_norm[i]}")

                        print(f"\nGT bbox (normalized):")
                        if gt_boxes is not None and len(gt_boxes) > 0:
                            gt_norm = gt_boxes.astype(float)
                            gt_norm[:, [0, 3]] /= H
                            gt_norm[:, [1, 4]] /= W
                            gt_norm[:, [2, 5]] /= D
                            for i in range(min(3, len(gt_norm))):
                                print(f"  GT#{i}: {gt_norm[i]}")
                        else:
                            print(f"  No GT boxes")
                        full = self.unmold_small_3d_mask(
                            m_small,
                            pd_boxes_px[j],  # ✅ ПРАВИЛЬНО - пиксели!
                            cid,
                            image.shape
                        )

                        if full is not None:
                            seg[full > 0.5] = j + 1
                            pd_masks_list.append(full > 0.5)

                    # Сохранение TIFF сегментации
                    seg_out = np.moveaxis(seg.astype(np.uint8), -1, 0)
                    imsave(
                        os.path.join(out_dir, f"{name}.tiff"),
                        seg_out,
                        check_contrast=False,
                        photometric='minisblack',
                        metadata={'axes': 'ZYX'}
                    )

                    # Преобразуем список масок в 4D массив
                    if pd_masks_list:
                        pd_masks = np.stack(pd_masks_list, axis=-1)
                        print(f"[INFO] Generated {pd_masks.shape[-1]} masks from HEAD")

                # CSV с боксами
                self.save_classes_and_boxes(pd_class_ids, pd_boxes_px, name)

                # Вычисляем метрики по МАСКАМ
                metrics = _compute_mask_metrics(pd_masks, gt_masks, pd_boxes_px, gt_boxes)
                self._print_metrics_summary(name, metrics)
                if pd_masks is not None and pd_masks.size > 0:
                    print(f"\n[EVAL] {name}: Masks stats:")
                    print(f"  pd_masks.shape: {pd_masks.shape}")
                    print(f"  pd_masks.dtype: {pd_masks.dtype}")
                    print(
                        f"  pd_masks sum per mask: {[pd_masks[..., i].sum() if pd_masks.ndim == 4 else pd_masks[i].sum() for i in range(min(3, pd_masks.shape[0] if pd_masks.ndim == 3 else pd_masks.shape[-1]))]}")
                    print(f"  pd_masks max: {np.max(pd_masks)}")
                else:
                    print(f"\n[EVAL] {name}: pd_masks is None or empty!")

                # Также проверь gt_masks:
                if gt_masks is not None and gt_masks.size > 0:
                    print(f"  gt_masks.shape: {gt_masks.shape}")
                    print(f"  gt_masks sum: {gt_masks.sum()}")
                    for g_idx in range(min(gt_masks.shape[-1], 5)):  # Первые 5 масок
                        gt_coords = np.where(gt_masks[..., g_idx] > 0.5)
                        if len(gt_coords[0]) > 0:
                            print(f"  GT mask #{g_idx}: Y={np.min(gt_coords[0])}-{np.max(gt_coords[0])}, "
                                  f"X={np.min(gt_coords[1])}-{np.max(gt_coords[1])}, "
                                  f"Z={np.min(gt_coords[2])}-{np.max(gt_coords[2])}, "
                                  f"pixels={len(gt_coords[0])}")
                    gt_coords = np.where(gt_masks[..., 0] > 0.5)
                    print(f"  GT mask #0 Y range: {np.min(gt_coords[0])}-{np.max(gt_coords[0])}")
                    print(f"  GT mask #0 X range: {np.min(gt_coords[1])}-{np.max(gt_coords[1])}")
                    print(f"  GT mask #0 Z range: {np.min(gt_coords[2])}-{np.max(gt_coords[2])}")
                # Визуализация МАСОК (не боксов!)
                try:
                    _draw_masks_overlay(
                        name, image, pd_masks, gt_masks, pd_boxes_px, gt_boxes,
                        out_dir, pd_scores=pd_scores, metrics_dict=metrics
                    )

                    # Собираем метрики
                    if 'pixelwise' in metrics:
                        all_metrics['pixelwise_f1'].append(metrics['pixelwise'][2])
                    if 'instance_dice' in metrics:
                        all_metrics['instance_dice'].append(metrics['instance_dice'][0])
                    if 'detection_performance' in metrics:
                        all_metrics['recall'].append(metrics['detection_performance']['recall'])
                        all_metrics['precision'].append(metrics['detection_performance']['precision'])

                except Exception as e:
                    print(f"[WARN] overlay failed: {e}")
                    import traceback
                    traceback.print_exc()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] evaluate failed for {image_id}: {e}")
                import traceback
                traceback.print_exc()
        if all_confidence_scores:
            print(f"\n{'=' * 70}")
            print(f"[GLOBAL CONFIDENCE STATISTICS]")
            print(f"{'=' * 70}")
            print(f"Total proposals: {len(all_confidence_scores)}")
            print(f"Mean: {np.mean(all_confidence_scores):.4f}")
            print(f"Median: {np.median(all_confidence_scores):.4f}")
            print(f"Max: {np.max(all_confidence_scores):.4f}")

            print(f"\n[HISTOGRAM]")
            total = sum(confidence_histogram.values())
            for range_name, count in confidence_histogram.items():
                pct = 100.0 * count / total if total > 0 else 0
                bar = '█' * int(pct / 2)
                print(f"  {range_name}: {count:>8} ({pct:>5.1f}%) {bar}")

            p90 = np.percentile(all_confidence_scores, 90)
            print(f"\n💡 RECOMMENDATION:")
            print(f"  Current threshold: {MIN_CONFIDENCE}")
            print(f"  Suggested threshold: {min(0.5, p90):.2f} (90th percentile)")
            print(f"{'=' * 70}\n")
        # Финальная статистика
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY (MASK-BASED)")
        print("=" * 70)

        # Статистика фильтрации
        print(f"\n[FILTERING STATISTICS]")
        print(f"  Total raw detections: {filter_stats['total_raw']}")
        print(f"  After confidence filter: {filter_stats['after_confidence']} "
              f"({100.0 * filter_stats['after_confidence'] / max(1, filter_stats['total_raw']):.1f}%)")
        print(f"  After size filter: {filter_stats['after_size']} "
              f"({100.0 * filter_stats['after_size'] / max(1, filter_stats['total_raw']):.1f}%)")
        print(f"  After NMS: {filter_stats['after_nms']} "
              f"({100.0 * filter_stats['after_nms'] / max(1, filter_stats['total_raw']):.1f}%)")

        if all_metrics['pixelwise_f1']:
            print(f"\n[PIXELWISE SEGMENTATION]")
            print(f"  Mean F1: {np.mean(all_metrics['pixelwise_f1']):.3f}")
            print(f"  Std F1: {np.std(all_metrics['pixelwise_f1']):.3f}")

        if all_metrics['instance_dice']:
            print(f"\n[INSTANCE DICE]")
            print(f"  Mean DICE: {np.mean(all_metrics['instance_dice']):.3f}")
            print(f"  Std DICE: {np.std(all_metrics['instance_dice']):.3f}")

        if all_metrics['recall']:
            print(f"\n[DETECTION METRICS (mask-based)]")
            print(f"  Mean Recall: {np.mean(all_metrics['recall']):.3f}")
            print(f"  Mean Precision: {np.mean(all_metrics['precision']):.3f}")

        print(f"\nResults saved to: {out_dir}")
        print("=" * 70 + "\n")

    def unmold_small_3d_mask(self, mask_small, bbox, class_id, image_shape, label=None):
        """..."""
        import numpy as np
        from core.utils import resize

        m = np.asarray(mask_small, dtype=np.float32)
        while m.ndim > 3:
            m = m.squeeze()

        if m.size == 0 or m.ndim != 3:
            return None

        # Логиты → вероятности
        m_min, m_max = float(np.min(m)), float(np.max(m))
        if m_min < -0.1 or m_max > 1.1:
            m = 1.0 / (1.0 + np.exp(-np.clip(m, -10, 10)))
            m_min, m_max = float(np.min(m)), float(np.max(m))

        m_mean = float(np.mean(m))
        m_std = float(np.std(m))

        if m_std < 1e-6:
            return None

        p95 = float(np.percentile(m, 95))
        if p95 < 0.10:
            return None

        # === BBOX ОБРАБОТКА ===
        b = np.asarray(bbox, dtype=np.float32)
        H, W, D = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])

        # ✅ ДИАГНОСТИКА
        print(f"\n[UNMOLD_MASK] class={class_id}")
        print(f"  bbox raw: {b}")

        y1 = int(np.floor(b[0]))
        x1 = int(np.floor(b[1]))
        z1 = int(np.floor(b[2]))
        y2 = int(np.ceil(b[3]))
        x2 = int(np.ceil(b[4]))
        z2 = int(np.ceil(b[5]))

        print(f"  bbox int: [{y1}, {x1}, {z1}, {y2}, {x2}, {z2}]")

        y1 = np.clip(y1, 0, H - 1)
        y2 = np.clip(y2, y1 + 1, H)
        x1 = np.clip(x1, 0, W - 1)
        x2 = np.clip(x2, x1 + 1, W)
        z1 = np.clip(z1, 0, D - 1)
        z2 = np.clip(z2, z1 + 1, D)

        hh = y2 - y1
        ww = x2 - x1
        dd = z2 - z1

        print(f"  bbox clipped: size=({hh}x{ww}x{dd})")

        # === THRESHOLD ===
        p50 = float(np.percentile(m, 50))
        p90 = float(np.percentile(m, 90))

        if m_mean > 0.4:
            thr = 0.5
        elif m_mean < 0.1:
            active = m[m > p50]
            thr = float(np.percentile(active, 30)) if len(active) > 10 else 0.30
            thr = np.clip(thr, 0.15, 0.45)
        else:
            try:
                from skimage.filters import threshold_otsu
                thr = float(np.clip(threshold_otsu(m), 0.20, 0.6))
            except:
                thr = float(np.clip((p50 + p90) / 2.0, 0.25, 0.55))

        binm = (m >= thr).astype(np.uint8)

        if float(np.sum(binm)) / binm.size < 0.0001:
            return None

        # === CLEANING ===
        if 0.0001 < float(np.sum(binm)) / binm.size < 0.95:
            try:
                from scipy.ndimage import label
                labeled, n_comp = label(binm)
                if n_comp > 1:
                    sizes = np.bincount(labeled.ravel())
                    keep = sizes >= max(2, int(binm.size * 0.0002))
                    keep[0] = False
                    binm = np.isin(labeled, np.where(keep)[0]).astype(np.uint8)
            except:
                pass

        # === RESIZE ===
        try:
            resized = resize(
                binm.astype(np.float32),
                output_shape=(hh, ww, dd),
                order=1,
                mode='constant',
                cval=0,
                clip=True,
                preserve_range=True,
                anti_aliasing=False
            )

            resize_thr = 0.3 if m_mean < 0.15 else 0.4
            binm_resized = (resized >= resize_thr).astype(np.uint8)

        except Exception as e:
            print(f"  resize failed: {e}")
            ih, iw, id_ = binm.shape
            yy = np.linspace(0, ih - 1, hh).astype(int)
            xx = np.linspace(0, iw - 1, ww).astype(int)
            zz = np.linspace(0, id_ - 1, dd).astype(int)
            binm_resized = binm[np.ix_(yy, xx, zz)]

        # ✅ ДИАГНОСТИКА после resize
        density = float(np.sum(binm_resized)) / binm_resized.size if binm_resized.size > 0 else 0.0
        print(f"  after resize: shape={binm_resized.shape}, sum={np.sum(binm_resized)}, density={density:.4f}")

        if density < 0.00001:
            return None

        # === ВСТАВКА ===
        full_mask = np.zeros((H, W, D), dtype=np.uint8)

        actual_h = min(binm_resized.shape[0], hh)
        actual_w = min(binm_resized.shape[1], ww)
        actual_d = min(binm_resized.shape[2], dd)

        full_mask[y1:y1 + actual_h, x1:x1 + actual_w, z1:z1 + actual_d] = \
            binm_resized[:actual_h, :actual_w, :actual_d]

        print(f"  final mask: sum={np.sum(full_mask)}")
        print(f"[UNMOLD_DEBUG] Predicted mask location:")
        print(f"  Non-zero pixels: {np.sum(full_mask > 0.5)}")  # ✅ full_mask
        if np.sum(full_mask > 0.5) > 0:  # ✅ full_mask
            coords = np.where(full_mask > 0.5)  # ✅ full_mask
            print(f"  Y range: {np.min(coords[0])}-{np.max(coords[0])}")
            print(f"  X range: {np.min(coords[1])}-{np.max(coords[1])}")
            print(f"  Z range: {np.min(coords[2])}-{np.max(coords[2])}")
        return full_mask

    def unmold_detections(self, detections, mrcnn_mask):
        """..."""
        import numpy as np
        from core import utils

        boxes = detections[:, :6].astype(np.float32)
        class_ids = detections[:, 6].astype(np.int32)
        scores = detections[:, 7].astype(np.float32) if detections.shape[1] > 7 else np.ones(len(detections),
                                                                                             np.float32)

        H, W, D = int(self.config.IMAGE_SHAPE[0]), int(self.config.IMAGE_SHAPE[1]), int(self.config.IMAGE_SHAPE[2])

        # ✅ ДИАГНОСТИКА
        print(f"\n[UNMOLD_DET] Input boxes (first 2, normalized [0,1]):")
        for i in range(min(2, len(boxes))):
            print(f"  {boxes[i]}")

        # ✅ Денормализация в ПИКСЕЛИ
        boxes_px = utils.denorm_boxes(boxes, (H, W, D))

        print(f"[UNMOLD_DET] After denorm (should be in pixels):")
        for i in range(min(2, len(boxes_px))):
            print(
                f"  {boxes_px[i]} -> size ({boxes_px[i, 3] - boxes_px[i, 0]:.1f}, {boxes_px[i, 4] - boxes_px[i, 1]:.1f}, {boxes_px[i, 5] - boxes_px[i, 2]:.1f})")

        # Фильтр
        keep = class_ids > 0
        h = boxes_px[:, 3] - boxes_px[:, 0]
        w = boxes_px[:, 4] - boxes_px[:, 1]
        d = boxes_px[:, 5] - boxes_px[:, 2]
        keep &= (h > 0.5) & (w > 0.5) & (d > 0.5)  # ✅ минимум 0.5px

        print(f"[UNMOLD_DET] Kept {np.sum(keep)}/{len(keep)} detections")

        boxes_px = boxes_px[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        if mrcnn_mask.ndim != 5:
            raise ValueError(f"mrcnn_mask must be [N,mH,mW,mD,C], got {mrcnn_mask.shape}")

        mrcnn_mask = mrcnn_mask[keep]

        # Генерация масок
        full_masks = []
        for i in range(boxes_px.shape[0]):
            ch = 0 if mrcnn_mask.shape[-1] == 1 else int(class_ids[i])
            ch = min(ch, mrcnn_mask.shape[-1] - 1)

            small = mrcnn_mask[i, :, :, :, ch]

            # Конвертация логитов
            smin, smax = float(np.min(small)), float(np.max(small))
            if smin < -0.1 or smax > 1.1:
                small = 1.0 / (1.0 + np.exp(-small))

            # ✅ КРИТИЧНО: передаём boxes_px (пиксели!)
            full = self.unmold_small_3d_mask(
                small,
                boxes_px[i],  # ← ПИКСЕЛИ, не нормализованные!
                int(class_ids[i]),
                (H, W, D)
            )

            if full is not None:
                full_masks.append(full.astype(bool))
            else:
                full_masks.append(np.zeros((H, W, D), dtype=bool))

        if full_masks:
            masks_nhwd = np.stack(full_masks, axis=0)
            masks_nhwd = np.moveaxis(masks_nhwd, 0, -1)  # теперь (H, W, D, N)
            seg_union = np.any(masks_nhwd, axis=-1)
        else:
            masks_nhwd = np.zeros((H, W, D, 0), dtype=bool)
            seg_union = np.zeros((H, W, D), dtype=bool)

        return boxes_px, scores, class_ids, masks_nhwd, seg_union

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        key = tuple(image_shape)

        if key not in self._anchor_cache:
            # 1) якоря в ПИКСЕЛЯХ
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE
            )

            H, W, D = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])

            # ❌ УБИРАЕМ МОДИФИКАЦИЮ Z - она должна быть в generate_pyramid_anchors!
            # # 2) делаем Z-толщину согласованной с генератором
            # h = (a[:, 3] - a[:, 0]).astype(np.float32)
            # w = (a[:, 4] - a[:, 1]).astype(np.float32)
            # ...

            # ✅ ПРОСТО НОРМАЛИЗУЕМ КАК ЕСТЬ
            scale = np.array([H, W, D, H, W, D], dtype=np.float32)
            a_norm = a / scale
            a_norm = np.clip(a_norm, 0.0, 1.0).astype(np.float32)

            self._anchor_cache[key] = a_norm

        return self._anchor_cache[key]



############################################################
#  Data Formatting
############################################################


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


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32)# - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return normalized_images.astype(np.uint8) #(normalized_images + config.MEAN_PIXEL)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 6] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    ИСПРАВЛЕНО: БЕЗ shift для согласованности.
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1)  # БЕЗ -1!
    return tf.divide(boxes, scale)


def denorm_boxes_graph(boxes, shape):
    """
    Converts boxes from normalized coordinates to pixel coordinates.
    ИСПРАВЛЕНО: БЕЗ shift для согласованности.
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1)  # БЕЗ -1!
    return tf.cast(tf.round(tf.multiply(boxes, scale)), tf.int32)
