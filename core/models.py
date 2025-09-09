import os
import re
import math
import multiprocessing
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave

from core import utils

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# ---- ЕДИНЫЙ TF1-стиль графа и один сеанс ----
tf.compat.v1.disable_eager_execution()
K.clear_session()
_tf_config = tf.compat.v1.ConfigProto()
_tf_config.gpu_options.allow_growth = True
_tf_sess = tf.compat.v1.Session(config=_tf_config)
try:
    # Привязываем TF1-сессию к backend Keras
    K.set_session(_tf_sess)
    # Для некоторых сборок ещё полезно синхронизировать через tf.compat.v1.keras
    tf.compat.v1.keras.backend.set_session(_tf_sess)
except Exception as e:
    print("[tf-compat] set_session warn:", e)



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
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


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
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding3D((3, 3, 3))(input_image)
    # x = KL.Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', use_bias=True)(x)
    x = KL.Conv3D(64, (7, 7, 7), strides=(2, 2, 1), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
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
    Applies the given deltas to the given boxes.

    Args:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] boxes to update
        deltas: [N, (dy, dx, dz, log(dh), log(dw), log(dd))] refinements to apply

    Returns:
        result: the corrected boxes
    """

    # Convert to y, x, z, h, w, d
    height = boxes[:, 3] - boxes[:, 0]
    width = boxes[:, 4] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 2]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 2] + 0.5 * depth

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= tf.exp(deltas[:, 3])
    width *= tf.exp(deltas[:, 4])
    depth *= tf.exp(deltas[:, 5])

    # Convert back to y1, x1, z1, y2, x2, z2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    result = tf.stack([y1, x1, z1, y2, x2, z2], axis=1, name="apply_box_deltas_out")

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
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, dz, log(dh), log(dw), log(dd))]
        anchors: [batch, num_anchors, (y1, x1, z1, y2, x2, z2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, z1, y2, x2, z2)]
    """

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

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # Box deltas [batch, num_rois, 6]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 6])

        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        scores = utils.batch_slice(
            [scores, ix], 
            lambda x, y: tf.gather(x, y), 
            self.images_per_gpu
        )

        deltas = utils.batch_slice(
            [deltas, ix], 
            lambda x, y: tf.gather(x, y), 
            self.images_per_gpu
        )

        pre_nms_anchors = utils.batch_slice(
            [anchors, ix], 
            lambda a, x: tf.gather(a, x),
            self.images_per_gpu,
            names=["pre_nms_anchors"]
        )

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, z1, y2, x2, z2)]
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas], 
            lambda x, y: apply_box_deltas_graph(x, y),
            self.images_per_gpu,
            names=["refined_anchors"]
        )

        def _enforce_min_dz(b):
            # b: [N, 6] normalized [y1, x1, z1, y2, x2, z2]
            y1, x1, z1, y2, x2, z2 = tf.split(b, 6, axis=1)
            # 1 voxel in normalized coords = 1/(D-1)
            min_dz = tf.constant(1.0, dtype=tf.float32) / tf.cast(tf.maximum(1, self.image_depth - 1), tf.float32)
            z2 = tf.maximum(z2, z1 + min_dz)
            return tf.concat([y1, x1, z1, y2, x2, z2], axis=1)

        boxes = utils.batch_slice(
            boxes,
            lambda bb: _enforce_min_dz(bb),
            self.images_per_gpu,
            names=["refined_anchors_min_dz"]
        )
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, z1, y2, x2, z2)]
        window = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.images_per_gpu,
            names=["refined_anchors_clipped"]
        )

        # Non-max suppression
        def nms(boxes, scores):

            indices = custom_op.non_max_suppression_3d(
                boxes, 
                scores, 
                self.proposal_count,
                self.nms_threshold, 
                name="rpn_non_max_suppression"
            )

            proposals = tf.gather(boxes, indices)

            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])

            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.images_per_gpu)

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
    shared = KL.Dropout(0.1, name='rpn_dropout')(shared)

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
                  kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(shared)

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
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
      pool_shape: [pool_height, pool_width, pool_depth]
    Inputs:
      - boxes: [batch, num_boxes, (y1,x1,z1,y2,x2,z2)] in normalized coords
      - image_meta
      - feature_maps: P2..P5, each [batch, H, W, D, C]
    Output:
      [batch, num_boxes, pool_h, pool_w, pool_d, C]
    """
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):

        boxes, image_meta = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        # split & sanitize boxes (клип + гарантированный положительный размер)
        eps = tf.constant(1e-6, dtype=boxes.dtype)
        y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=2)

        def _clip01(t):
            return tf.clip_by_value(t, 0.0, 1.0)

        y1 = _clip01(y1); x1 = _clip01(x1); z1 = _clip01(z1)
        y2 = _clip01(y2); x2 = _clip01(x2); z2 = _clip01(z2)

        y2 = tf.maximum(y2, y1 + eps)
        x2 = tf.maximum(x2, x1 + eps)
        z2 = tf.maximum(z2, z1 + eps)

        boxes = tf.concat([y1, x1, z1, y2, x2, z2], axis=2)

        # Assign each ROI to a level in the pyramid based on the ROI size
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        h = y2 - y1; w = x2 - x1; d = z2 - z1
        image_area = tf.cast(image_shape[0] * image_shape[1] * image_shape[2], tf.float32)
        roi_level = log2_graph(tf.pow(h * w * d, 1.0/3.0) / (224.0 / tf.pow(image_area, 1.0/3.0)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)  # [batch, num_boxes]

        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))  # [K, 2] -> (batch_ix, box_ix)
            level_boxes = tf.gather_nd(boxes, ix)
            # Сохраняем индекс батча, чтобы crop оп понимал из какого экземпляра брать фичи
            box_indices = tf.cast(ix[:, 0], tf.int32)
            # crop_and_resize_3d ожидает нормализованные боксы
            # pooled: [K, ph, pw, pd, C]
            pooled.append(custom_op.crop_and_resize_3d(feature_maps[i], level_boxes, box_indices, self.pool_shape))
            box_to_level.append(ix)

        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)

        # восстановить порядок боксов как в исходных "boxes"
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 1], ix)
        pooled = tf.gather(pooled, ix)

        # [batch, num_boxes, ph, pw, pd, C]
        out_shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, out_shape)

        # простая защита от NaN/Inf (не упадём молча)
        pooled = tf.where(tf.math.is_finite(pooled), pooled, tf.zeros_like(pooled))
        return pooled

    def compute_output_shape(self, input_shape):
        # input_shape: [boxes, image_meta, P2, P3, P4, P5]
        b = input_shape[0][0]
        n = input_shape[0][1]
        c = input_shape[2][-1]
        ph, pw, pd = self.pool_shape
        return (b, n, ph, pw, pd, c)


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.

    Args:
        boxes1, boxes2: [N, (y1, x1, z1, y2, x2, z2)].

    Returns:
        overlaps: matrix of overlap
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)

    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 6])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # 2. Compute intersections
    b1_y1, b1_x1, b1_z1, b1_y2, b1_x2, b1_z2 = tf.split(b1, 6, axis=1)
    b2_y1, b2_x1, b2_z1, b2_y2, b2_x2, b2_z2 = tf.split(b2, 6, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    z1 = tf.maximum(b1_z1, b2_z1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    z2 = tf.minimum(b1_z2, b2_z2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(z2 - z1, 0)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) * (b1_z2 - b1_z1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1) * (b2_z2 - b2_z1)
    union = b1_area + b2_area - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, train_rois_per_image,
                            roi_positive_ratio, bbox_std_dev, use_mini_mask, mask_shape):
    """
    Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, z1, y2, x2, z2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, z1, y2, x2, z2)] in normalized coordinates.
        gt_masks: [height, width, depth, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, dz, log(dh), log(dw), log(dd) )]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, depth]. Masks cropped to bbox
               boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion")
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=3,
                         name="trim_gt_masks")

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs.
    # Positive ROIs
    positive_count = int(train_rois_per_image * roi_positive_ratio)

    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    # Negative ROIs. Add enough to maintain positive:negative ratio.
    # r = 1.0 / config.ROI_POSITIVE_RATIO
    # negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_count = train_rois_per_image - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    negative_overlaps = tf.gather(overlaps, negative_indices)
    roi_gt_box_pos_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_box_neg_assignment = tf.argmax(negative_overlaps, axis=1)

    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_pos_assignment)

    roi_pos_gt_boxes = tf.gather(gt_boxes, roi_gt_box_pos_assignment)
    roi_neg_gt_boxes = tf.gather(gt_boxes, roi_gt_box_neg_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_pos_gt_boxes)
    deltas /= bbox_std_dev

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [3, 0, 1, 2]), -1)

    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_pos_assignment)

    # Compute mask targets
    boxes = positive_rois

    if use_mini_mask:

        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, z1, y2, x2, z2 = tf.split(positive_rois, 6, axis=1)
        gt_y1, gt_x1, gt_z1, gt_y2, gt_x2, gt_z2 = tf.split(roi_pos_gt_boxes, 6, axis=1)
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
    masks = custom_op.crop_and_resize_3d(tf.cast(roi_masks, tf.float32), boxes, box_ids, mask_shape)
    
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=4)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    roi_gt_boxes = tf.concat([roi_pos_gt_boxes, roi_neg_gt_boxes], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(train_rois_per_image - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0), (0, 0)])

    return rois, roi_gt_boxes, roi_gt_class_ids, deltas, masks


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

    def __init__(self, train_rois_per_image, roi_positive_ratio, bbox_std_dev, 
                 use_mini_mask, mask_shape, images_per_gpu, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.mask_shape = mask_shape
        self.images_per_gpu = images_per_gpu

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
                self.train_rois_per_image,
                self.roi_positive_ratio,
                self.bbox_std_dev,
                self.use_mini_mask,
                self.mask_shape
            ),
            self.images_per_gpu, 
            names=names
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 6),  # rois
            (None, self.train_rois_per_image, 6),  # gt_boxes
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
    """
    Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, z1, y2, x2, z2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to apply to
                     proposal boxes
    """

    s = K.int_shape(y)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Reshape((s[1], fc_layers_size), name="pool_reshape")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    x = KL.TimeDistributed(KL.Dense(num_classes * 6, activation='linear'), name='mrcnn_bbox_fc')(shared)

    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw), log(dd))]
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 6), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(y, num_classes, conv_channel, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    # Conv layers
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv1")(y)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)

    return x


# === PATCH === в models.py
def fpn_classifier_graph_with_RoiAlign(rois, feature_maps, image_meta,
                                       pool_size, num_classes, fc_layers_size,
                                       train_bn=True, name_prefix=""):

    import keras.layers as KL
    from keras import backend as K
    N = (lambda n: f"{name_prefix}{n}") if name_prefix else (lambda n: n)

    x = PyramidROIAlign([pool_size, pool_size, pool_size], name=N("roi_align_classifier"))(
        [rois, image_meta] + feature_maps
    )
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name=N("mrcnn_class_conv1"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_class_bn1'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name=N("mrcnn_class_conv2"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_class_bn2'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    def _reshape_shared(t):
        b = tf.shape(t)[0]; n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, fc_layers_size))
    shared = KL.Lambda(_reshape_shared, name=N("pool_reshape"))(x)

    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name=N('mrcnn_class_logits'))(shared)
    mrcnn_probs       = KL.TimeDistributed(KL.Activation("softmax"), name=N("mrcnn_class"))(mrcnn_class_logits)

    x = KL.TimeDistributed(KL.Dense(num_classes * 6, activation='linear'), name=N('mrcnn_bbox_fc'))(shared)

    def _reshape_bbox(t):
        b = tf.shape(t)[0]; n = tf.shape(t)[1]
        return tf.reshape(t, (b, n, num_classes, 6))
    mrcnn_bbox = KL.Lambda(_reshape_bbox, name=N("mrcnn_bbox"))(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph_with_RoiAlign(rois, feature_maps, image_meta,
                                       pool_size, num_classes, conv_channel,
                                       train_bn=True, name_prefix=""):
    import keras.layers as KL
    N = (lambda n: f"{name_prefix}{n}") if name_prefix else (lambda n: n)

    x = PyramidROIAlign([pool_size, pool_size, pool_size], name=N("roi_align_mask"))(
        [rois, image_meta] + feature_maps
    )
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
                           name=N("mrcnn_mask_conv1"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn1'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
                           name=N("mrcnn_mask_conv2"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn2'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
                           name=N("mrcnn_mask_conv3"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn3'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"),
                           name=N("mrcnn_mask_conv4"))(x)
    x = KL.TimeDistributed(BatchNorm(), name=N('mrcnn_mask_bn4'))(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv3DTranspose(conv_channel, (2, 2, 2), strides=2, activation="relu"),
                           name=N("mrcnn_mask_deconv"))(x)
    x = KL.TimeDistributed(KL.Conv3D(num_classes, (1, 1, 1), strides=1, activation="sigmoid"),
                           name=N("mrcnn_mask"))(x)
    return x


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, image_meta,
                            detection_min_confidence,
                            detection_nms_threshold,
                            bbox_std_dev=None,
                            detection_max_instances=None):

    from core import utils as U

    if bbox_std_dev is None:
        bbox_std_dev = BBOX_STD_DEV
    bbox_std_dev = tf.cast(bbox_std_dev, tf.float32)

    # пороги оставляем питоновскими скалярами
    min_conf = float(detection_min_confidence)
    nms_thr  = float(detection_nms_threshold)
    max_inst = int(detection_max_instances) if detection_max_instances is not None else 200

    # image_meta -> [1, META]
    meta = tf.cond(tf.equal(tf.rank(image_meta), 1),
                   lambda: tf.expand_dims(image_meta, 0),
                   lambda: image_meta)

    meta_parsed = parse_image_meta_graph(meta)
    image_shape = tf.cast(meta_parsed['image_shape'][0], tf.float32)   # [H,W,D]
    window      = tf.cast(meta_parsed['window'][0], tf.float32)        # [y1,x1,z1,y2,x2,z2]

    # класс и скор
    class_ids    = tf.argmax(probs, axis=1, output_type=tf.int32)
    class_scores = tf.reduce_max(probs, axis=1)

    # фильтр по min_conf
    keep_ix      = tf.where(class_scores >= min_conf)[:, 0]
    rois_keep    = tf.gather(rois,   keep_ix)       # [K,6] (norm)
    scores_keep  = tf.gather(class_scores, keep_ix)
    class_keep   = tf.gather(class_ids,    keep_ix)

    # дельты выбранного класса
    deltas_keep_all = tf.gather(deltas, keep_ix)    # [K, C, 6]
    Kk = tf.shape(deltas_keep_all)[0]
    idx = tf.stack([tf.range(Kk, dtype=tf.int32), class_keep], axis=1)
    deltas_keep = tf.gather_nd(deltas_keep_all, idx)  # [K,6]
    deltas_keep.set_shape([None, 6])

    # denorm ROIs -> px (3D-версия)
    rois_px  = U.denorm_boxes_3d_graph(rois_keep, image_shape)      # [K,6]

    # применяем дельты
    boxes_px = U.apply_box_deltas_3d_graph(rois_px, deltas_keep, bbox_std_dev)  # [K,6]

    # клип по границам
    H, W, D = image_shape[0], image_shape[1], image_shape[2]
    y1 = tf.clip_by_value(boxes_px[:, 0], 0.0, H - 1.0)
    x1 = tf.clip_by_value(boxes_px[:, 1], 0.0, W - 1.0)
    z1 = tf.clip_by_value(boxes_px[:, 2], 0.0, D - 1.0)
    y2 = tf.clip_by_value(boxes_px[:, 3], 0.0, H - 1.0)
    x2 = tf.clip_by_value(boxes_px[:, 4], 0.0, W - 1.0)
    z2 = tf.clip_by_value(boxes_px[:, 5], 0.0, D - 1.0)
    boxes_px = tf.stack([y1, x1, z1, y2, x2, z2], axis=1)
    boxes_px.set_shape([None, 6])

    # клип по window (после letterbox/cube)
    wy1, wx1, wz1, wy2, wx2, wz2 = window[0], window[1], window[2], window[3], window[4], window[5]
    y1 = tf.maximum(boxes_px[:, 0], wy1); x1 = tf.maximum(boxes_px[:, 1], wx1); z1 = tf.maximum(boxes_px[:, 2], wz1)
    y2 = tf.minimum(boxes_px[:, 3], wy2); x2 = tf.minimum(boxes_px[:, 4], wx2); z2 = tf.minimum(boxes_px[:, 5], wz2)
    boxes_px = tf.stack([y1, x1, z1, y2, x2, z2], axis=1)
    boxes_px.set_shape([None, 6])

    # tiny-боксы
    hh = boxes_px[:, 3] - boxes_px[:, 0]
    ww = boxes_px[:, 4] - boxes_px[:, 1]
    dd = boxes_px[:, 5] - boxes_px[:, 2]
    tiny_ok = tf.where((hh > 1.0) & (ww > 1.0) & (dd > 0.5))[:, 0]
    boxes_px   = tf.gather(boxes_px,   tiny_ok)
    scores_keep= tf.gather(scores_keep,tiny_ok)
    class_keep = tf.gather(class_keep, tiny_ok)

    # per-class NMS (используем Python thr/int, тензоры — только boxes/scores)
    unique_classes = tf.unique(class_keep)[0]

    def body(ci, out_b, out_s, out_c):
        cls = unique_classes[ci]
        cls_mask = tf.where(tf.equal(class_keep, cls))[:, 0]
        b = tf.gather(boxes_px,    cls_mask)
        s = tf.gather(scores_keep, cls_mask)
        b_nms, keep_idx = U.non_max_suppression_3d_graph(b, s, nms_thr, max_inst)
        s_nms = tf.gather(s, keep_idx)
        c_nms = tf.fill([tf.shape(b_nms)[0]], tf.cast(cls, tf.float32))
        return ci + 1, tf.concat([out_b, b_nms], axis=0), tf.concat([out_s, s_nms], axis=0), tf.concat([out_c, c_nms], axis=0)

    def cond(ci, *_):
        return ci < tf.shape(unique_classes)[0]

    out_b0 = tf.zeros([0, 6], dtype=tf.float32)
    out_s0 = tf.zeros([0],    dtype=tf.float32)
    out_c0 = tf.zeros([0],    dtype=tf.float32)

    _, final_boxes_px, final_scores, final_class = tf.while_loop(
        cond, body, [tf.constant(0, dtype=tf.int32), out_b0, out_s0, out_c0],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, 6]),
                          tf.TensorShape([None]),
                          tf.TensorShape([None])]
    )

    # top-K + паддинг до max_inst
    k = tf.minimum(tf.shape(final_scores)[0], tf.constant(max_inst, dtype=tf.int32))
    order = tf.nn.top_k(final_scores, k=k).indices
    final_boxes_px = tf.gather(final_boxes_px, order)
    final_scores   = tf.gather(final_scores,   order)
    final_class    = tf.gather(final_class,    order)

    # в норм-координаты (3D)
    final_boxes_nm = U.norm_boxes_3d_graph(final_boxes_px, image_shape)  # [k,6]
    final_class_id = tf.expand_dims(final_class, 1)                      # [k,1]
    final_scores   = tf.expand_dims(final_scores, 1)                     # [k,1]

    detections_k = tf.concat([final_boxes_nm, final_class_id, final_scores], axis=1)  # [k,8]
    pad = tf.maximum(0, tf.constant(max_inst, tf.int32) - tf.shape(detections_k)[0])
    detections = tf.pad(detections_k, [[0, pad], [0, 0]])
    detections.set_shape([None, 8])
    return detections




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
                 *args,  # <- поглотим лишний позиционный (например batch_size) из старых вызовов
                 **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.bbox_std_dev = bbox_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = detection_max_instances
        self.detection_nms_threshold = detection_nms_threshold
        self.images_per_gpu = images_per_gpu
        # args игнорируем намеренно, чтобы сохранить обратную совместимость

    def call(self, inputs):
        rois         = inputs[0]  # [B, R, 6] (normalized)
        mrcnn_class  = inputs[1]  # [B, R, C]
        mrcnn_bbox   = inputs[2]  # [B, R, C, 6]
        image_meta   = inputs[3]  # [B, META]

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

        # динамический батч
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


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """Упрощенная и исправленная функция потерь для классификации"""
    rpn_match = K.squeeze(rpn_match, -1)
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

    # Выбираем non-neutral якоря
    indices = tf.where(K.not_equal(rpn_match, 0))

    def no_samples():
        return tf.constant(0.0, dtype=tf.float32)

    def compute_loss():
        # Собираем предсказания и целевые значения
        pred_class = tf.gather_nd(rpn_class_logits, indices)
        target_class = tf.gather_nd(anchor_class, indices)

        # Считаем количество позитивных и негативных якорей
        pos_count = tf.reduce_sum(tf.cast(target_class, tf.float32))
        neg_count = tf.cast(tf.shape(target_class)[0], tf.float32) - pos_count

        # Умеренный вес для позитивных примеров
        pos_weight = tf.where(pos_count > 0,
                              3.0 * neg_count / (pos_count + neg_count),
                              tf.constant(2.0))
        neg_weight = 1.0

        # Применяем веса к примерам
        sample_weights = tf.where(
            tf.cast(target_class, tf.bool),
            tf.fill(tf.shape(target_class), pos_weight),
            tf.fill(tf.shape(target_class), neg_weight)
        )

        # Cross entropy с весами
        ce = K.sparse_categorical_crossentropy(
            target=target_class,
            output=pred_class,
            from_logits=True
        )

        # Клиппинг для стабильности
        ce = tf.clip_by_value(ce, 0, 10.0)

        return K.mean(ce * sample_weights)

    return tf.cond(tf.equal(tf.size(indices), 0), no_samples, compute_loss)


def rpn_bbox_loss_graph(images_per_gpu, target_bbox, rpn_match, rpn_bbox):
    """Улучшенная функция потерь для регрессии bbox"""
    rpn_match = K.squeeze(rpn_match, -1)
    pos_indices = tf.where(K.equal(rpn_match, 1))

    def no_pos():
        return tf.constant(0.0, dtype=tf.float32)

    def compute_loss():
        # Предсказания для позитивных якорей
        pred_bbox = tf.gather_nd(rpn_bbox, pos_indices)

        # Собираем GT boxes
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        gt_boxes = batch_pack_graph(target_bbox, batch_counts, images_per_gpu)

        # Разница между GT и pred
        diff = gt_boxes - pred_bbox

        # Ограничиваем большие значения для стабильности
        diff = tf.clip_by_value(diff, -4.0, 4.0)

        # Балансированные веса для разных компонентов bbox
        # Центр более важен, чем размеры
        coord_weights = tf.constant([1., 1., 1., 1., 1., 1.], dtype=tf.float32)

        # Применяем веса
        weighted_diff = diff * coord_weights

        # Huber loss (smooth L1)
        huber_delta = 1.0
        abs_diff = tf.abs(weighted_diff)
        huber_loss = tf.where(
            abs_diff < huber_delta,
            0.5 * tf.square(weighted_diff),
            huber_delta * (abs_diff - 0.5 * huber_delta)
        )

        # Добавляем дополнительный вес позитивным примерам
        # для компенсации их малого количества
        pos_weight = 2.0
        weighted_loss = huber_loss * pos_weight

        return K.mean(weighted_loss)

    return tf.cond(tf.equal(tf.size(pos_indices), 0), no_pos, compute_loss)





def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """
    Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, 
        logits=pred_class_logits
    )

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)

    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 6))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 6))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), 
        tf.int64
    )
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)

    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width, depth].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, depth, num_classes] float32 tensor
                with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]))
    
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 4, 1, 2, 3])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)

    return loss


############################################################
#  Custom Callbacks
############################################################
class BestAndLatestCheckpoint(keras.callbacks.Callback):
    """Хранит два файла в save_path:
       - latest.h5 — перезаписывается каждую эпоху
       - best.h5   — переписывается при улучшении выбранной метрики
       Метрики читаем из logs, которые заполняют EvaluationCallbacks.
       mode:
         - 'RPN'  : выше лучше, сумма rpn_{train,test}_detection_score
         - 'HEAD' : ниже лучше, head_test_total_loss (взвешенная сумма ВСЕХ доступных head-лоссов)
         - иначе  : ниже лучше, logs['loss']
    """
    def __init__(self, save_path="", mode="GENERIC"):
        super(BestAndLatestCheckpoint, self).__init__()
        self.save_path = save_path
        self.mode = (mode or "GENERIC").upper()
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        # всегда пишем latest
        latest_path = f"{self.save_path}latest.h5"
        self.model.save_weights(latest_path)

        logs = logs or {}
        metric = None
        better = False

        if self.mode == 'RPN':
            train_score = logs.get('rpn_train_detection_score', 0.0)
            test_score  = logs.get('rpn_test_detection_score',  0.0)
            metric = float(train_score) + float(test_score)
            better = (self.best is None) or (metric > self.best)

        elif self.mode == 'HEAD':
            # приоритет — total (он уже взвешен LOSS_WEIGHTS)
            metric = logs.get('head_test_total_loss', None)

            # fallback: суммируем все head_test_*_mean, если есть (class/bbox/mask/obj/margin)
            if metric is None:
                acc = 0.0
                any_found = False
                for short in ("class", "bbox", "mask", "obj", "margin"):
                    k = f"head_test_{short}_mean"
                    if k in logs:
                        acc += float(logs[k])
                        any_found = True
                if any_found:
                    metric = acc

            if metric is not None:
                better = (self.best is None) or (float(metric) < self.best)

        else:
            metric = logs.get('loss', None)
            if metric is not None:
                better = (self.best is None) or (float(metric) < self.best)

        if better and (metric is not None):
            self.best = float(metric)
            best_path = f"{self.save_path}best.h5"
            self.model.save_weights(best_path)

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
            # соберём пару ключевых метрик из logs, если есть
            extra = {}
            if logs:
                for k in ("rpn_train_detection_score", "rpn_test_detection_score",
                          "rpn_train_mean_coord_error", "rpn_test_mean_coord_error",
                          "head_test_total_loss", "loss"):
                    if k in logs:
                        # приводим к float, чтобы json был сериализуемым
                        try:
                            extra[k] = float(logs[k])
                        except Exception:
                            pass

            # ВАЖНО: правильный импорт внутри пакета core
            from core.utils import Telemetry
            Telemetry.snapshot_and_reset(epoch + 1, self.save_dir, extra=extra)
        except Exception as e:
            # на всякий случай выведем предупреждение один раз в эпоху
            print(f"[Telemetry] snapshot failed: {e}")

# class SaveWeightsCallback(keras.callbacks.Callback):
#     def __init__(self, save_path):
#         super(SaveWeightsCallback, self).__init__()
#         self.save_path = save_path
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.model.save_weights(f"{self.save_path}epoch_{str(epoch+1).zfill(3)}.h5")


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


class HeadEvaluationCallback(keras.callbacks.Callback):
    """
    Прогоняет голову по всему train/test датасетам порционно (чанки по T ROI),
    собирает средние по каждому loss слою, печатает сводку и пишет в logs:
      - head_{subset}_{lossname}_mean  (в т.ч. obj/margin, если есть)
      - head_{subset}_total_loss       (ВЗВЕШЕННАЯ сумма по LOSS_WEIGHTS)
    """
    def __init__(self, model, config, train_dataset, test_dataset):
        super(HeadEvaluationCallback, self).__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Построим отображение имя_выхода -> индекс
        self._name_to_idx = {name: i for i, name in enumerate(getattr(self.model, "output_names", []))}

        # Какие лоссы потенциально есть у головы
        self._possible_losses = [
            "mrcnn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss",
            "mrcnn_obj_loss",
            "mrcnn_margin_loss",
        ]

        # Оставим только реально присутствующие в outputs модели
        self._present_losses = [n for n in self._possible_losses if n in self._name_to_idx]

        # Веса из конфига (по умолчанию 1.0)
        self._lw = {n: float(getattr(self.config, "LOSS_WEIGHTS", {}).get(n, 1.0)) for n in self._possible_losses}

    def _eval_subset(self, dataset, subset_name: str):
        import numpy as np, time

        # ===== ЛОКАЛЬНЫЕ КОНСТАНТЫ =====
        T = int(getattr(self.config, "TRAIN_ROIS_PER_IMAGE", 128))
        MAX_IMGS = 12
        MAX_CHUNKS_PER_IMG = 4
        POS_CHUNK_BIAS = 0.7
        COMPUTE_STD = False
        TIME_BUDGET_SEC = 0

        # ===== подготовка источника данных =====
        gen = HeadGenerator(dataset=dataset, config=self.config, shuffle=False, training=False)

        img_ids = list(dataset.image_ids)
        if isinstance(MAX_IMGS, int) and MAX_IMGS > 0 and len(img_ids) > MAX_IMGS:
            idx = np.linspace(0, len(img_ids) - 1, num=MAX_IMGS, dtype=np.int32)
            img_ids = [int(img_ids[i]) for i in idx]

        # ===== берём только выходы loss-слоёв =====
        loss_layers, loss_order = [], []
        for lname in self._present_losses:
            try:
                loss_layers.append(self.model.get_layer(lname).output)
                loss_order.append(lname)
            except Exception:
                pass
        if not loss_layers:
            loss_order = list(self._present_losses)
            loss_layers = [self.model.outputs[self._name_to_idx[l]] for l in loss_order]

        # Пытаемся подготовить слой/выход objness для AUC
        obj_name = "mrcnn_obj"
        obj_layer = None
        try:
            obj_layer = self.model.get_layer(obj_name).output
        except Exception:
            try:
                obj_layer = self.model.outputs[self._name_to_idx[obj_name]]
            except Exception:
                obj_layer = None

        # Список тензоров, которые попробуем тянуть одной функцией
        fetch_layers = list(loss_layers)
        obj_fetch_idx = None
        if obj_layer is not None:
            obj_fetch_idx = len(fetch_layers)
            fetch_layers.append(obj_layer)

        # Функция для инференса выбранных тензоров (если получится)
        try:
            fetch_fn = K.function(self.model.inputs, fetch_layers)
        except Exception:
            fetch_fn = None  # откатимся на predict_on_batch / отдельный K.function по необходимости

        # ==== безопасная выборка obj-оценок ====
        def _safe_fetch_obj_scores(model, inputs_b, outs, obj_fetch_idx, obj_name, name_to_idx):
            # 1) если obj шёл в fetch_layers и вернулся — взять из outs
            if outs is not None and obj_fetch_idx is not None:
                if 0 <= obj_fetch_idx < len(outs):
                    try:
                        return np.asarray(outs[obj_fetch_idx])
                    except Exception:
                        pass
            # 2) попробовать достать из predict_on_batch по имени выхода
            try:
                full = model.predict_on_batch(inputs_b)
                if obj_name in name_to_idx:
                    return np.asarray(full[name_to_idx[obj_name]])
            except Exception:
                pass
            # 3) отдельная функция только на obj-слой (если он есть)
            try:
                layer = model.get_layer(obj_name)
                fn = K.function(model.inputs, [layer.output])
                return np.asarray(fn(inputs_b)[0])
            except Exception:
                return None

        # ===== аккумулируем взвешенные суммы =====
        sum_by_loss = {n: 0.0 for n in loss_order}
        den_by_loss = {n: 0 for n in loss_order}
        chunk_vals = {n: [] for n in loss_order} if COMPUTE_STD else None

        # Для OBJNESS_AUC собираем по-ROI
        obj_scores_all, obj_labels_all = [], []

        pos_total, roi_total = 0, 0
        used_imgs, used_chunks_total = 0, 0

        # Утилита AUC (Mann–Whitney)
        def _binary_auc(scores, labels):
            s = np.asarray(scores, dtype=np.float64)
            y = np.asarray(labels, dtype=np.int32)
            if s.size == 0:
                return np.nan
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos == 0 or n_neg == 0:
                return np.nan
            order = np.argsort(s)
            s_sorted = s[order]
            y_sorted = y[order]
            ranks = np.empty_like(s_sorted, dtype=np.float64)
            n = s_sorted.size
            i = 0
            while i < n:
                j = i + 1
                while j < n and s_sorted[j] == s_sorted[i]:
                    j += 1
                r = 0.5 * ((i + 1) + j)  # средний ранг при тай-нах
                ranks[i:j] = r
                i = j
            sum_ranks_pos = ranks[y_sorted == 1].sum()
            auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            return float(auc)

        t0 = time.time()
        prev_lp = None
        try:
            try:
                prev_lp = K.learning_phase()
            except Exception:
                prev_lp = None
            try:
                K.set_learning_phase(0)
            except Exception:
                pass

            # ===== основной цикл =====
            for image_id in img_ids:
                if TIME_BUDGET_SEC and (time.time() - t0) > TIME_BUDGET_SEC:
                    break

                rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask = \
                    gen.load_image_gt(int(image_id))

                total = int(rois_aligned.shape[0])
                if total == 0:
                    continue

                # разбиение на чанки
                num_chunks = (total + T - 1) // T
                pos_mask_list = []
                n_roi_vec = np.empty((num_chunks,), dtype=np.int32)
                n_pos_vec = np.empty((num_chunks,), dtype=np.int32)
                for ci, start in enumerate(range(0, total, T)):
                    end = min(start + T, total)
                    tci = target_class_ids[start:end]
                    n_roi = int(end - start)
                    n_pos = int((tci > 0).sum())
                    n_roi_vec[ci] = n_roi
                    n_pos_vec[ci] = n_pos
                    pos_mask_list.append(n_pos > 0)

                # выбираем ограниченное число чанков
                if isinstance(MAX_CHUNKS_PER_IMG, int) and MAX_CHUNKS_PER_IMG > 0 and num_chunks > MAX_CHUNKS_PER_IMG:
                    pos_idx = [i for i, f in enumerate(pos_mask_list) if f]
                    neg_idx = [i for i, f in enumerate(pos_mask_list) if not f]
                    n_take_pos = min(len(pos_idx), int(round(MAX_CHUNKS_PER_IMG * POS_CHUNK_BIAS)))
                    n_take_neg = MAX_CHUNKS_PER_IMG - n_take_pos
                    take = (pos_idx[:n_take_pos] + neg_idx[:n_take_neg]) if n_take_neg > 0 else pos_idx[:n_take_pos]
                    take = sorted(take)
                else:
                    take = list(range(num_chunks))

                B = len(take)
                if B == 0:
                    continue

                # --- батч на изображение (B чанков) ---
                ra_b = np.zeros((B, T,) + rois_aligned.shape[1:], dtype=np.float32)
                ma_b = np.zeros((B, T,) + mask_aligned.shape[1:], dtype=np.float32)
                tci_b = np.full((B, T), -1, dtype=np.int32)
                tb_b = np.zeros((B, T, 6), dtype=np.float32)

                if target_mask.ndim == 5 and target_mask.shape[-1] == 1:
                    tm_tail = target_mask.shape[1:]  # (28,28,28,1)
                else:
                    tm_tail = target_mask.shape[1:] + (1,)  # (28,28,28,1) из (28,28,28)
                tm_b = np.zeros((B, T) + tm_tail, dtype=np.float32)

                im_b = np.repeat(image_meta[np.newaxis, ...], B, axis=0)

                n_roi_sel = np.empty((B,), dtype=np.int32)
                n_pos_sel = np.empty((B,), dtype=np.int32)

                for bi, ci in enumerate(take):
                    start = ci * T
                    end = min(start + T, total)

                    ra = rois_aligned[start:end]
                    ma = mask_aligned[start:end]
                    tci = target_class_ids[start:end]
                    tb = target_bbox[start:end]
                    tm = target_mask[start:end]

                    n_roi = ra.shape[0]
                    n_pos = int((tci > 0).sum())
                    n_roi_sel[bi] = n_roi
                    n_pos_sel[bi] = n_pos

                    ra_b[bi, :n_roi] = ra
                    ma_b[bi, :n_roi] = ma
                    tci_b[bi, :n_roi] = tci
                    tb_b[bi, :n_roi] = tb

                    if tm.ndim == 4:
                        tm = tm[..., np.newaxis]
                    tm_b[bi, :n_roi] = tm.astype(np.float32)

                # POS/RATE считаем по выбранным чанкам
                roi_total += int(n_roi_sel.sum())
                pos_total += int(n_pos_sel.sum())

                inputs_b = [ra_b, ma_b, im_b, tci_b, tb_b, tm_b]

                # тянем выбранные тензоры
                outs = None
                if fetch_fn is not None:
                    try:
                        outs = fetch_fn(inputs_b)
                    except Exception:
                        outs = None

                # если не удалось — fallback на predict_on_batch для лоссов
                if outs is None:
                    full = self.model.predict_on_batch(inputs_b)
                    outs = [np.asarray(full[self._name_to_idx[l]]) for l in loss_order]

                # аккумулируем лоссы
                for lidx, lname in enumerate(loss_order):
                    arr = np.asarray(outs[lidx])
                    if arr.ndim == 0 or arr.size == 1:
                        vals = np.full((B,), float(arr), dtype=np.float32)
                    else:
                        arr = arr.astype(np.float32, copy=False)
                        if arr.shape[0] != B:
                            try:
                                arr = arr.reshape(B, -1)
                            except Exception:
                                arr = np.broadcast_to(arr.reshape(1, -1), (B, -1))
                        else:
                            arr = arr.reshape(B, -1)
                        vals = arr.mean(axis=1)

                    denom = n_pos_sel if (("bbox" in lname) or ("mask" in lname)) else n_roi_sel
                    valid = denom > 0
                    if np.any(valid):
                        sum_by_loss[lname] += float(np.dot(vals[valid], denom[valid].astype(np.float64)))
                        den_by_loss[lname] += int(denom[valid].sum())
                    if COMPUTE_STD and chunk_vals is not None:
                        chunk_vals[lname].extend(vals.tolist())

                # собираем objness для AUC безопасно
                obj = _safe_fetch_obj_scores(self.model, inputs_b, outs, obj_fetch_idx, obj_name, self._name_to_idx)
                if obj is not None:
                    obj = np.asarray(obj)
                    # ожидаем (B,T,1) или (B,T); приведём к (B,T,1)
                    if obj.ndim == 2:
                        obj = obj[..., np.newaxis]
                    if obj.ndim == 3 and obj.shape[0] == B:
                        for bi in range(B):
                            n = int(n_roi_sel[bi])
                            if n <= 0:
                                continue
                            scores = obj[bi, :n, 0]
                            labels = (tci_b[bi, :n] > 0).astype(np.int32)
                            obj_scores_all.append(scores)
                            obj_labels_all.append(labels)

                used_imgs += 1
                used_chunks_total += int(B)

        finally:
            try:
                if prev_lp is not None:
                    K.set_learning_phase(prev_lp)
            except Exception:
                pass

        # ===== финальная сводка =====
        stats = {}
        for lname in loss_order:
            mean = float(sum_by_loss[lname] / den_by_loss[lname]) if den_by_loss[lname] > 0 else 0.0
            if COMPUTE_STD and chunk_vals is not None and len(chunk_vals[lname]) > 0:
                arr = np.asarray(chunk_vals[lname], dtype=np.float32)
                std = float(arr.std())
            else:
                std = 0.0
            short = lname.replace("mrcnn_", "").replace("_loss", "")
            stats[f"head_{subset_name}_{short}_mean"] = mean
            if COMPUTE_STD:
                stats[f"head_{subset_name}_{short}_std"] = std

        total = 0.0
        for lname in loss_order:
            mean = stats[f"head_{subset_name}_{lname.replace('mrcnn_', '').replace('_loss', '')}_mean"]
            total += mean * self._lw.get(lname, 1.0)
        stats[f"head_{subset_name}_total_loss"] = total

        pos_rate = float(pos_total) / max(int(roi_total), 1)
        stats[f"head_{subset_name}_pos_rate"] = pos_rate

        # ==== OBJNESS_AUC ====
        if len(obj_scores_all) > 0 and len(obj_labels_all) > 0:
            s = np.concatenate(obj_scores_all, axis=0)
            y = np.concatenate(obj_labels_all, axis=0)
            auc = _binary_auc(s, y)
        else:
            auc = np.nan
        stats[f"head_{subset_name}_obj_auc"] = float(auc) if np.isfinite(auc) else np.nan

        tag = "TRAIN SUBSET" if subset_name == "train" else "TEST SUBSET"
        parts = []
        for lname in ["mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss", "mrcnn_obj_loss", "mrcnn_margin_loss"]:
            if lname in loss_order:
                short = lname.replace("mrcnn_", "").replace("_loss", "").upper()
                mean = stats[f"head_{subset_name}_{short.lower()}_mean"]
                parts.append(f"{short}: {mean:.6f}")
        auc_str = f"OBJ_AUC: {auc:.3f}" if np.isfinite(auc) else "OBJ_AUC: n/a"

        print(tag)
        if parts:
            print("    " + "    ".join(parts))
        print(f"TOTAL(w): {total:.6f}    POS_RATE: {pos_rate:.3f}    {auc_str}    "
              f"[used {used_imgs}/{len(img_ids)} imgs, {used_chunks_total} chunks, "
              f"time {int(time.time() - t0)}s]")
        return stats

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        train_stats = self._eval_subset(self.train_dataset, "train")
        test_stats  = self._eval_subset(self.test_dataset,  "test")
        logs.update(train_stats)
        logs.update(test_stats)

class AutoTuneRPNCallback(keras.callbacks.Callback):
    """Акуратный автоподбор anchors: добавляем 1–2 значения из suggest и пересобираем якоря.
       Работает только в режиме RPN, не трогает NMS/IoU/лоссы.
    """
    def __init__(self, train_generator, config, every=1, warmup=1, max_changes=3):
        super().__init__()
        self.gen = train_generator
        self.config = config
        self.every = int(every)
        self.warmup = int(warmup)
        self.left = int(max_changes)

    def on_epoch_end(self, epoch, logs=None):
        if not getattr(self.config, "AUTO_TUNE_RPN", False):
            return
        ep = int(epoch) + 1
        if ep <= self.warmup or self.left <= 0:
            return

        # забираем последний снапшот/совет
        try:
            from core.utils import Telemetry
            # если ты сохранил suggest в snapshot_and_reset
            suggest = getattr(Telemetry, "_last_suggest", None)
        except Exception:
            suggest = None

        # fallback: просто читаем последний файл suggest_patch_*.json
        if suggest is None:
            import glob, json, os
            try:
                files = sorted(glob.glob(os.path.join(self.config.WEIGHT_DIR, "suggest_patch_*_epoch*.json")))
                if files:
                    with open(files[-1], "r", encoding="utf-8") as f:
                        patch = json.load(f)
                        suggest = patch
            except Exception:
                pass

        if not suggest or ep % self.every != 0:
            return

        # «grow-only»: добавим не более 1–2 новых scale/ratio
        cur_sc = list(self.config.RPN_ANCHOR_SCALES)
        cur_rt = list(self.config.RPN_ANCHOR_RATIOS)
        add_sc = [s for s in suggest.get("RPN_ANCHOR_SCALES", suggest.get("scales", [])) if s not in cur_sc]
        add_rt = [r for r in suggest.get("RPN_ANCHOR_RATIOS", suggest.get("ratios", [])) if r not in cur_rt]

        changed = False
        if add_sc:
            cur_sc += sorted(add_sc)[:1]   # добавим ОДИН «мостик»
            changed = True
        if add_rt:
            cur_rt += sorted(add_rt)[:1]   # и ОДИН ratio
            changed = True

        if changed:
            # ограничим длину (держим максимум как в конфиге)
            sc_limit = int(getattr(self.config, "AUTO_TUNE_SCALES_LIMIT", 8))
            rt_limit = int(getattr(self.config, "AUTO_TUNE_RATIOS_LIMIT", 8))
            self.config.RPN_ANCHOR_SCALES = sorted(cur_sc)[:sc_limit]
            self.config.RPN_ANCHOR_RATIOS = sorted(set(cur_rt))[:rt_limit]

            # пересобрать якоря у генератора
            try:
                self.gen.rebuild_anchors()
                print(f"[autotune] epoch={ep} applied scales={self.config.RPN_ANCHOR_SCALES} ratios={self.config.RPN_ANCHOR_RATIOS}")
                self.left -= 1
            except Exception as e:
                print(f"[autotune] rebuild_anchors failed: {e}")
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

        # Create Datasets
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
            # KL.UpSampling3D(size=(2, 2, 2), name="fpn_p5upsampled")(P5),
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p5upsampled")(P5),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c4p4')(C4)
        ])

        P3 = KL.Add(name="fpn_p3add")([
            # KL.UpSampling3D(size=(2, 2, 2), name="fpn_p4upsampled")(P4),
            KL.UpSampling3D(size=(2, 2, 1), name="fpn_p4upsampled")(P4),
            KL.Conv3D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = KL.Add(name="fpn_p2add")([
            # KL.UpSampling3D(size=(2, 2, 2), name="fpn_p3upsampled")(P3),
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
        anchors = self.get_anchors(self.config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

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
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None], name="input_gt_masks", dtype=bool)

            rois, target_gt_boxes, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.ROI_POSITIVE_RATIO,
                self.config.BBOX_STD_DEV,
                self.config.USE_MINI_MASK,
                self.config.MASK_SHAPE,
                self.config.IMAGES_PER_GPU,
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
        if self.config.OPTIMIZER["name"] == "SGD":
            # Получаем параметры из конфига
            optimizer_params = self.config.OPTIMIZER["parameters"].copy()

            # Создаем Keras оптимизатор с clipnorm для стабильности
            optimizer = keras.optimizers.SGD(
                learning_rate=optimizer_params.get("learning_rate", 0.0001),
                momentum=optimizer_params.get("momentum", 0.9),
                decay=optimizer_params.get("decay", 1e-5),
                clipnorm=1.0  # Ограничение нормы градиентов
            )
        else:
            optimizer_params = self.config.OPTIMIZER["parameters"].copy()
            optimizer = keras.optimizers.Adam(
                learning_rate=optimizer_params.get("learning_rate", 0.0001),
                beta_1=optimizer_params.get("beta_1", 0.9),
                beta_2=optimizer_params.get("beta_2", 0.999),
                epsilon=optimizer_params.get("epsilon", 1e-7),
                clipnorm=1.0  # Ограничение нормы градиентов
            )

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
            config=self.config,
            every=int(getattr(self.config, "AUTO_TUNE_EVERY", 1)),
            warmup=int(getattr(self.config, "AUTO_TUNE_WARMUP_EPOCHS", 1)),
            max_changes=2  # максимально 2 вмешательства за прогон
        )
        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Конфигурация модели
        # Устанавливаем правильные пороги IoU в конфиге
        self.config.RPN_POSITIVE_IOU = float(getattr(self.config, "RPN_POSITIVE_IOU", 0.20))  # Было 0.7, затем 0.6
        self.config.RPN_NEGATIVE_IOU = float(getattr(self.config, "RPN_NEGATIVE_IOU", 0.05))  # Было 0.3

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

        lr_scheduler = keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

        # Early Stopping для предотвращения переобучения
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        )

        # Снижение LR при плато
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks = [evaluation, save_weights, telemetry_cb, lr_scheduler, reduce_lr, early_stopping]
        if getattr(self.config, "AUTO_TUNE_RPN", False):
            callbacks.insert(0, autotune_cb)
        # Training loop с оптимизированными параметрами
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
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:3])
        return self._anchor_cache[tuple(image_shape)]

    def head_target_generation(self):

        assert self.config.MODE == "targeting", "Create model in targeting mode."

        # Create Data Generators
        train_generator = RPNGenerator(dataset=self.train_dataset, config=self.config)
        test_generator = RPNGenerator(dataset=self.test_dataset, config=self.config)

        # Create target folders
        base_path = f"{self.config.OUTPUT_DIR}"
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}datasets/", exist_ok=True)

        target_dirs = [
            f"{base_path}rois/",
            f"{base_path}rois_aligned/",
            f"{base_path}mask_aligned/",
            f"{base_path}target_class_ids/",
            f"{base_path}target_bbox/",
            f"{base_path}target_mask/",
        ]
        for target_dir in target_dirs:
            os.makedirs(target_dir, exist_ok=True)

        # Load RPN_WEIGHTS
        self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)

        # Proper target generation
        for generator, set_type in zip([train_generator, test_generator], ["train", "test"]):
            
            print(f"TARGET GENERATION FOR {set_type} DATASET...")

            # Initiate example dataframe
            example_dataframe = pd.DataFrame({
                "rois": [], 
                "rois_aligned": [],
                "mask_aligned": [], 
                "target_class_ids": [],
                "target_bbox": [], 
                "target_mask": [],
            })

            # Loop over examples
            n = int(self.config.TARGET_RATIO * len(generator.image_ids))
            for ex_id in tqdm(range(n)):

                item_info = generator.dataset.image_info[ex_id]
                name = item_info["path"].split("/")[-1].split(".")[0]
                res = generator.__getitem__(ex_id)
                if isinstance(res[0], str):
                    gen_name, inputs = res  # (name, inputs) из генератора
                    # при желании можно использовать gen_name вместо вычисленного name
                else:
                    inputs, _ = res

                    # Note that outputs is [rois, rois_aligned, mask_aligned, _, target_class_ids, target_bbox, target_mask]
                outputs = self.keras_model.predict(inputs)
                # We don't want the fourth element
                del outputs[3]

                dataframe_info = []
                for target_dir, output in zip(target_dirs, outputs):

                    save_path = f"{target_dir}{name}.npy"

                    # Save example
                    np.save(save_path, output[0])

                    dataframe_info.append(save_path)

                example_dataframe.loc[len(example_dataframe.index)] = dataframe_info
            
            # Save dataset dataframe
            example_dataframe.to_csv(f"{base_path}datasets/{set_type}.csv", index=None)

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

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, show_summary):
        """
        config: RPN configuration
        weight_dir: training weight directory
        train_dataset: data.Dataset train object
        test_dataset: data.Dataset test object
        """
        
        self.config = config

        self.keras_model = self.build()
        print("[HEAD.__init__] model built", flush=True)
        self.epoch = self.config.FROM_EPOCH

        self.train_dataset, self.test_dataset = self.prepare_datasets()
        print("[HEAD.__init__] datasets prepared", flush=True)
        if show_summary:
            print("[HEAD.__init__] printing summary ...", flush=True)
            self.print_summary()


    def prepare_datasets(self):

        # Create Datasets
        train_dataset = ToyHeadDataset()

        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)

        train_dataset.prepare()

        train_dataset.filter_positive()
        test_dataset = ToyHeadDataset()
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
        Build Head Mask R-CNN architecture.
        Обучаем голову по заранее подготовленным патчам (rois_aligned/mask_aligned),
        без добавления новых конфигурационных полей.
        """
        import keras as KM
        import keras.layers as KL

        assert self.config.MODE in ["training", "targeting"], "HEAD expects training/targeting mode"

        # === Inputs (ровно под генератор HeadGenerator) ===
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
        input_target_class_ids = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE], name="input_target_class_ids")
        input_target_bbox = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, 6], name="input_target_bbox")
        input_target_mask = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, *self.config.MASK_SHAPE, 1],
                                     name="input_target_mask")

        active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

        # === Heads ===
        # ВАЖНО: fc_layers_size ДОЛЖЕН совпадать с инференсной головой ⇒ FPN_CLASSIF_FC_LAYERS_SIZE
        mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = fpn_classifier_graph(
            y=input_rois_aligned,
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,  # <-- фикс
            train_bn=False  # BN фризим для стабильности на маленьких батчах
        )
        mrcnn_mask = build_fpn_mask_graph(
            y=input_mask_aligned,
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,
            train_bn=False
        )

        # === Losses (обязательно оборачиваем через lambda x: func(*x)) ===
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [input_target_class_ids, mrcnn_class_logits, active_class_ids]
        )
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [input_target_bbox, input_target_class_ids, mrcnn_bbox]
        )
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [input_target_mask, input_target_class_ids, mrcnn_mask]
        )

        # === Model ===
        inputs = [
            input_rois_aligned, input_mask_aligned, input_image_meta,
            input_target_class_ids, input_target_bbox, input_target_mask
        ]
        outputs = [
            mrcnn_class_logits, mrcnn_prob, mrcnn_bbox, mrcnn_mask,
            class_loss, bbox_loss, mask_loss
        ]
        model = KM.Model(inputs, outputs, name='head_training')

        # Multi-GPU (как у тебя)
        if self.config.GPU_COUNT > 1:
            from core.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)

        return model

    def compile(self):
        """
        Компиляция HEAD (обучение по патчам):
        - добавляем лоссы через add_loss (class/bbox/mask и др., если есть),
        - L2-регуляризация (кроме BN gamma/beta),
        - compile без явных y (Keras берёт лоссы из add_loss).
        """
        import keras
        import keras.backend as K


        m = self.keras_model

        # Сброс ранее добавленных лоссов (на случай повторной компиляции)
        try:
            m._losses.clear()
        except Exception:
            m._losses = []
        m._per_input_losses = {}

        # Какие лоссы пытаться прикрепить (добавим только если слой существует)
        loss_layer_names = [
            "mrcnn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss",
            "mrcnn_obj_loss",
            "mrcnn_margin_loss",
        ]

        for lname in loss_layer_names:
            try:
                layer = m.get_layer(lname)
            except Exception:
                continue
            weight = float(getattr(self.config, "LOSS_WEIGHTS", {}).get(lname, 1.0))
            m.add_loss(K.mean(layer.output) * weight)

        # L2 регуляризация весов (исключая BN gamma/beta)
        wd = float(getattr(self.config, "WEIGHT_DECAY", 0.0))
        if wd > 0.0:
            l2_terms = []
            for w in m.trainable_weights:
                wn = w.name
                if "gamma" in wn or "beta" in wn:
                    continue
                # нормируем на число элементов, чтобы L2 не зависел от формы тензора
                l2_terms.append(keras.regularizers.l2(wd)(w) / K.cast(K.prod(K.shape(w)), K.floatx()))
            if l2_terms:
                m.add_loss(tf.add_n(l2_terms))

        # Оптимайзер из конфига (SGD/Adadelta/Adam); дефолт — SGD(lr=1e-3, momentum=0.9)
        opt_cfg = getattr(self.config, "OPTIMIZER",
                          {"name": "SGD", "parameters": {"learning_rate": 1e-3, "momentum": 0.9}})
        oname = str(opt_cfg.get("name", "SGD")).upper()
        oparams = dict(opt_cfg.get("parameters", {}))
        if oname == "SGD":
            optimizer = keras.optimizers.SGD(**oparams)
        elif oname == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**oparams)
        else:
            optimizer = keras.optimizers.Adam(**oparams)

        # Компиляция: лоссы уже добавлены через add_loss
        m.compile(optimizer=optimizer, loss=[None] * len(m.outputs))

    def train(self):
        assert self.config.MODE == "training", "Create model in training mode."

        # Улучшение: разделяем датасет на train/val (80/20)
        train_ids = self.train_dataset.image_ids
        np.random.shuffle(train_ids)
        split = int(0.8 * len(train_ids))
        train_ids, val_ids = train_ids[:split], train_ids[split:]

        # Генераторы с улучшенной нормализацией (z-score)
        def normalize_volume(vol):
            mu, sigma = np.mean(vol), np.std(vol)
            return (vol - mu) / sigma if sigma > 0 else vol

        class NormalizedHeadGenerator(HeadGenerator):
            def load_image_gt(self, image_id):
                rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask = super().load_image_gt(
                    image_id)
                # Normalize if needed, but for head, inputs are ROIs/masks – assume pre-normalized
                return rois_aligned, mask_aligned, image_meta, target_class_ids, target_bbox, target_mask

        train_generator = NormalizedHeadGenerator(dataset=self.train_dataset.subset(train_ids), config=self.config,
                                                  training=True)
        val_generator = NormalizedHeadGenerator(dataset=self.train_dataset.subset(val_ids), config=self.config,
                                                training=False) # Eval mode for val

        # Callback для сохранения
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='MRCNN')

        # Улучшение: дополнительные callbacks
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join(self.config.WEIGHT_DIR, 'logs'))

        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Загрузка весов (как было, but for heads load RPN first if needed)
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Workers=0 for stable
        workers = 0

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(train_ids),
            callbacks=[save_weights, early_stop, reduce_lr, tensorboard],
            validation_data=val_generator,
            validation_steps=len(val_ids),
            max_queue_size=30,
            workers=workers,
            use_multiprocessing=False,
        )


from keras import backend as K

def _ensure_tf1_graph():
    """Переводим среду в режим TF1-графа: disable eager, clear_session, set Session(allow_growth)."""
    try:
        if hasattr(tf, "executing_eagerly") and tf.executing_eagerly():
            tf1.disable_eager_execution()
    except Exception:
        pass
    try:
        K.clear_session()
    except Exception:
        pass
    try:
        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        sess = tf1.Session(config=cfg)

    except Exception as e:
        print("[tf-compat] session init warn:", e)

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

    def _infer_head_params_from_h5(self, weights_path):
        """
        Пробуем достать из H5 формы ключевых слоёв головы и вернуть словарь:
          {"pool": <POOL_SIZE>, "fc": <FPN_CLASSIF_FC_LAYERS_SIZE>, "mask": <HEAD_CONV_CHANNEL>}
        Если чего-то нет — кидаем исключение, чтобы сработал fallback на config.
        """
        import os, h5py
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"no head weights: {weights_path}")

        with h5py.File(weights_path, "r") as f:
            root = f["model_weights"] if "model_weights" in f.keys() else f

            def _shape(layer):
                if layer not in root:
                    raise KeyError(f"layer '{layer}' not found in H5")
                g = root[layer]
                # берём первый тензор (kernel) — у Conv3D/Conv3DTranspose он всегда есть
                ds = [g[k] for k in g.keys() if isinstance(g[k], h5py.Dataset)]
                if not ds:
                    raise KeyError(f"no tensors in layer '{layer}'")
                return tuple(ds[0].shape)

            # Классификатор: mrcnn_class_conv1: [1,1,1,Cin, fc]
            s_cls1 = _shape("mrcnn_class_conv1")
            # Маск-ветка: mrcnn_mask_conv1: [3,3,3,Cin, mask_ch]
            s_m1 = _shape("mrcnn_mask_conv1")

            # POOL_SIZE восстанавливаем из conv1 kernel shape для класификатора:
            # у нас conv(kernel_size=(POOL,POOL,POOL), padding='valid') → Cin = pool^3 * TOP_DOWN_PYRAMID_SIZE,
            # но проще взять из ROIAlign-веса нельзя; поэтому используем config.POOL_SIZE как «pool»,
            # а fc берём из выхода conv1/conv2 (последняя размерность).
            # Если у тебя conv1 стоит (POOL,POOL,POOL), то первая тройка элементов ядра == (POOL,POOL,POOL).
            try:
                pool = int(s_cls1[0])  # kernel_depth
            except Exception:
                pool = int(getattr(self.config, "POOL_SIZE", 7))

            fc = int(s_cls1[-1])
            mask = int(s_m1[-1])

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
        """Посчитать, для скольких слоёв по именам и по форме есть совместимые веса в H5.
           Поддерживает оба layout'а: корень и подгруппу 'model_weights'.
        """
        try:
            import h5py
            f = h5py.File(weights_path, 'r')
        except Exception:
            return 0

        # Где лежат слои в H5
        root = f
        if 'model_weights' in f.keys():
            root = f['model_weights']

        # Словарь: имя_слоя_в_модели -> список форм тензоров
        name_to_shapes = {}
        for layer in model.layers:
            w = layer.weights
            if not w:
                continue
            shapes = [tuple(K.int_shape(v)) for v in w]
            name_to_shapes[layer.name] = shapes

        matched = 0
        for lname, shapes in name_to_shapes.items():
            if lname not in root:
                continue
            g = root[lname]
            # извлекаем датасеты (веса) в группе
            h5_tensors = [g[k] for k in g.keys() if isinstance(g[k], h5py.Dataset)]
            if len(h5_tensors) != len(shapes):
                continue
            ok = True
            for ds, shp in zip(h5_tensors, shapes):
                # сравниваем, разрешая None в модели
                target = tuple((d if d is not None else s) for d, s in zip(shp, ds.shape))
                if tuple(ds.shape) != target:
                    ok = False
                    break
            if ok:
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
        keys = [
            "mrcnn_class_conv1", "mrcnn_class_conv2",
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
            "mrcnn_mask_conv1", "mrcnn_mask_conv2", "mrcnn_mask_conv3", "mrcnn_mask_conv4",
            "mrcnn_mask_deconv", "mrcnn_mask",
        ]
        print("[HEAD] Weights healthcheck (L2-norms):")
        for suf in keys:
            l = self._get_layer_by_suffix(suf)
            if l is None:
                print(f"  {suf:22s} : missing")
                continue
            try:
                ws = l.get_weights()
                if not ws:
                    print(f"  {suf:22s} : empty")
                    continue
                norms = [float(np.linalg.norm(w)) for w in ws]
                print(f"  {suf:22s} : " + ", ".join(f"{n:.3f}" for n in norms))
            except Exception as e:
                print(f"  {suf:22s} : error {e}")

    def prepare_datasets(self):

        # Create Datasets
        train_dataset = ToyDataset()
        train_dataset.config = self.config
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
        train_dataset.prepare()
        train_dataset.filter_positive()
        test_dataset = ToyDataset()
        test_dataset.config = self.config
        test_dataset.load_dataset(data_dir=self.config.DATA_DIR, is_train=False)
        test_dataset.prepare()

        return train_dataset, test_dataset

    def print_summary(self):
        
        # Model summary
        self.keras_model.summary(line_length=140)

        # Number of example in Datasets
        print("\nTrain dataset contains:", len(self.train_dataset.image_info), " elements.")
        print("Test dataset contains:", len(self.test_dataset.image_info), " elements.")

        # Configuration
        self.config.display()

    # В файле core/models.py, внутри класса MaskRCNN:


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
        # if hasattr(self, "_pick_head_channels"):
        #     try:
        #         fc_ch = int(self._pick_head_channels())
        #     except Exception as e:
        #         print("[build] _pick_head_channels() failed; fallback", fc_ch, ":", e)

        # СНОВА чистый граф после пробы, чтобы не было коллизий имён
        _ensure_tf1_graph()

        # ---------- (2) Inputs ----------
        input_image = KL.Input(shape=[*self.config.IMAGE_SHAPE], name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_anchors = KL.Input(shape=[None, 6], name="input_anchors", dtype=tf.float32)

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
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.ROI_POSITIVE_RATIO,
                self.config.BBOX_STD_DEV,
                self.config.USE_MINI_MASK,
                self.config.MASK_SHAPE,
                self.config.IMAGES_PER_GPU,
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
                train_bn=False,
                name_prefix=""  # каноничные имена mrcnn_*
            )

            # --- Mask head ---
            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
                rois=rois,
                feature_maps=mrcnn_feature_maps,
                image_meta=input_image_meta,
                pool_size=self.config.MASK_POOL_SIZE,
                num_classes=self.config.NUM_CLASSES,
                conv_channel=self.config.HEAD_CONV_CHANNEL,  # <-- фикс: ширина mask-ветки = HEAD_CONV_CHANNEL
                train_bn=False,
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
            # Подстроим только голову, RPN и FPN не трогаем
            pool_from_h5 = int(hp["pool"])
            fc_class_ch = int(hp["fc"])
            mask_ch = int(hp["mask"])
            # ВАЖНО: ROIAlign classifier должен иметь тот же pool_size, что в обученных весах
            self.config.POOL_SIZE = pool_from_h5
            print(
                f"[HEAD][infer] using POOL_SIZE={pool_from_h5}, classifier_ch={fc_class_ch}, mask_ch={mask_ch} from H5")
        except Exception as e:
            print(f"[HEAD][infer] H5 introspection failed: {e} ; fallback to config")
            fc_class_ch = int(
                getattr(self.config, "FPN_CLASSIF_FC_LAYERS_SIZE", getattr(self.config, "HEAD_CONV_CHANNEL", 128)))
            mask_ch = int(getattr(self.config, "HEAD_CONV_CHANNEL", fc_class_ch))
        # --- classifier+bbox на rpn_rois ---
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(
            rois=rpn_rois, feature_maps=mrcnn_feature_maps, image_meta=input_image_meta,
            pool_size=self.config.POOL_SIZE, num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,  # <-- фикс: как в тренинге головы
            train_bn=False, name_prefix=""
        )

        # --- DetectionLayer: refine + NMS ---
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

        # --- mask head по финальным боксам ---
        mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
            rois=detection_boxes,  # <-- фикс: финальные боксы
            feature_maps=mrcnn_feature_maps, image_meta=input_image_meta,
            pool_size=self.config.MASK_POOL_SIZE, num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,  # <-- фикс: как в тренинге головы
            train_bn=False, name_prefix=""
        )

        # --- Основная инференс-модель ---
        infer_inputs = [input_image, input_image_meta, input_anchors]
        infer_outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois]
        self.keras_model = KM.Model(infer_inputs, infer_outputs, name="mask_rcnn_inference")

        # ----- Вспомогательные eval-модели (всегда с префиксами!) -----
        # 1) core: detections + P2..P5
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

        # 2) head_eval: classifier+bbox на произвольных ROIs + (P2..P5)
        eval_rois = KL.Input(shape=[None, 6], name="eval_rois")
        eval_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="eval_meta")
        P2_in = KL.Input(shape=K.int_shape(mrcnn_feature_maps[0])[1:], name="eval_P2")
        P3_in = KL.Input(shape=K.int_shape(mrcnn_feature_maps[1])[1:], name="eval_P3")
        P4_in = KL.Input(shape=K.int_shape(mrcnn_feature_maps[2])[1:], name="eval_P4")
        P5_in = KL.Input(shape=K.int_shape(mrcnn_feature_maps[3])[1:], name="eval_P5")

        cls_logits_eval, cls_probs_eval, bbox_deltas_eval = fpn_classifier_graph_with_RoiAlign(
            rois=eval_rois, feature_maps=[P2_in, P3_in, P4_in, P5_in], image_meta=eval_meta,
            pool_size=self.config.POOL_SIZE, num_classes=self.config.NUM_CLASSES,
            fc_layers_size=fc_ch, train_bn=False, name_prefix="eval_"
        )
        self.keras_head_eval = KM.Model(
            [eval_rois, eval_meta, P2_in, P3_in, P4_in, P5_in],
            [cls_probs_eval, bbox_deltas_eval],
            name="head_eval"
        )

        # 3) mask_head_eval: маски на произвольных ROIs + (P2..P5)
        eval_rois_m = KL.Input(shape=[None, 6], name="eval_rois_mask")
        mask_logits_eval = build_fpn_mask_graph_with_RoiAlign(
            rois=eval_rois_m, feature_maps=[P2_in, P3_in, P4_in, P5_in], image_meta=eval_meta,
            pool_size=self.config.MASK_POOL_SIZE, num_classes=self.config.NUM_CLASSES,
            conv_channel=fc_ch, train_bn=False, name_prefix="mask_eval_"
        )
        self.keras_mask_head_eval = KM.Model(
            [eval_rois_m, eval_meta, P2_in, P3_in, P4_in, P5_in],
            mask_logits_eval,
            name="mask_head_eval"
        )

        return self.keras_model

    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        self.keras_model.metrics_tensors = []

        # Use Adam by default for better convergence
        if self.config.OPTIMIZER["name"] == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])
        elif self.config.OPTIMIZER["name"] == "SGD":
            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Default Adam

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
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join(self.config.WEIGHT_DIR, 'logs'))

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
            callbacks=[save_weights, early_stop, reduce_lr, tensorboard],
            validation_data=val_generator,
            validation_steps=len(val_ids),
            max_queue_size=30,
            workers=workers,
            use_multiprocessing=False,
        )

    def _refine_detections_numpy(self, rpn_rois_batch, mrcnn_class, mrcnn_bbox, image_meta,
                                 min_conf=None, nms_thr=None, max_inst=None):
        """
        rpn_rois_batch: [1,R,6] norm
        mrcnn_class:    [R,C]
        mrcnn_bbox:     [R,C,6] — в стандартизованных deltas
        image_meta:     [1, meta]
        Возвращает: detections [N,8] = (y1,x1,z1,y2,x2,z2, cls_id, score), norm coords
        """
        import numpy as np
        from core import utils

        min_conf = float(getattr(self.config, "DETECTION_MIN_CONFIDENCE", 0.15) if min_conf is None else min_conf)
        nms_thr = float(getattr(self.config, "DETECTION_NMS_THRESHOLD", 0.45) if nms_thr is None else nms_thr)
        max_inst = int(getattr(self.config, "DETECTION_MAX_INSTANCES", 200) if max_inst is None else max_inst)

        # batch=1
        rois_nm = rpn_rois_batch[0]  # [R,6] norm
        probs = mrcnn_class  # [R,C]
        deltas = mrcnn_bbox  # [R,C,6]

        # размеры кадра для перевода в пиксели
        img_shape = self.config.IMAGE_SHAPE[:3]
        rois_px = utils.denorm_boxes(rois_nm, img_shape).astype(np.float32)  # [R,6]

        C = probs.shape[1]
        H, W, D = img_shape
        detections_all = []

        # по классам (1..C-1), 0 — фон
        for c in range(1, C):
            cls_scores = probs[:, c]
            keep_ix = np.where(cls_scores >= min_conf)[0]
            if keep_ix.size == 0:
                continue

            # decode deltas для этого класса
            class_deltas = deltas[keep_ix, c, :]  # [K,6]
            boxes_ref = rois_px[keep_ix]  # [K,6]
            refined_px = utils.apply_box_deltas_3d(boxes_ref, class_deltas, self.config.BBOX_STD_DEV)
            refined_px = refined_px.astype(np.float32)

            # клиппим
            refined_px[:, 0] = np.clip(refined_px[:, 0], 0, H - 1)
            refined_px[:, 1] = np.clip(refined_px[:, 1], 0, W - 1)
            refined_px[:, 2] = np.clip(refined_px[:, 2], 0, D - 1)
            refined_px[:, 3] = np.clip(refined_px[:, 3], 0, H - 1)
            refined_px[:, 4] = np.clip(refined_px[:, 4], 0, W - 1)
            refined_px[:, 5] = np.clip(refined_px[:, 5], 0, D - 1)

            # tiny-фильтр
            hh = refined_px[:, 3] - refined_px[:, 0]
            ww = refined_px[:, 4] - refined_px[:, 1]
            dd = refined_px[:, 5] - refined_px[:, 2]
            ok = np.where((hh > 1) & (ww > 1) & (dd > 0.5))[0]
            if ok.size == 0:
                continue
            refined_px = refined_px[ok]
            sc = cls_scores[keep_ix][ok]

            # 3D-NMS
            refined_px_nms, keep_local = utils.non_max_suppression_3d(refined_px, sc, threshold=nms_thr,
                                                                      max_boxes=max_inst)
            if refined_px_nms.shape[0] == 0:
                continue
            sc_nms = sc[keep_local]

            # обратно в норм. координаты + склейка
            refined_nm = utils.norm_boxes(refined_px_nms, (H, W, D)).astype(np.float32)
            cls_id_col = np.full((refined_nm.shape[0], 1), float(c), dtype=np.float32)
            det_c = np.concatenate([refined_nm, cls_id_col, sc_nms[:, None]], axis=1)  # [M,8]
            detections_all.append(det_c)

        if len(detections_all) == 0:
            return np.zeros((0, 8), dtype=np.float32)

        det = np.vstack(detections_all)  # [N,8]
        order = det[:, 7].argsort()[::-1]
        det = det[order]
        if det.shape[0] > max_inst:
            det = det[:max_inst]
        return det

    def process_image(self, image_id, generator, result_dir, roi_probe=None, cls_probe=None, det_idx=0, mask_idx=3):
        """
        Process a single image for evaluation.

        Args:
            image_id: ID of the image to process.
            generator: Instance of NormalizedMrcnnGenerator.
            result_dir: Directory to save results.
            roi_probe: Keras model to extract ROIs (optional).
            cls_probe: Keras model to extract class probabilities (optional).
            det_idx: Index of detections in model output.
            mask_idx: Index of masks in model output.

        Returns:
            List of metrics and metadata for the image.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from core.utils import compute_ap
        from skimage.io import imsave

        # Get inputs
        name, inputs = generator.get_input_prediction(image_id)
        name = name.rsplit(".", 1)[0]
        print(f"[DEBUG] Processing image: {name}, ID: {image_id}")

        # Fixed z-score normalization
        def normalize_volume(vol):
            mu, sigma = np.mean(vol), np.std(vol)
            return (vol - mu) / sigma if sigma > 0 else vol

        inputs[0][0] = normalize_volume(inputs[0][0])
        print(
            f"[DEBUG] Image shape: {inputs[0][0].shape}, mean: {np.mean(inputs[0][0]):.2f}, std: {np.std(inputs[0][0]):.2f}")

        # Debugging: Save ROIs and class probs
        roiN = None
        maxClsProb = None
        if roi_probe:
            try:
                rois = roi_probe.predict(inputs, batch_size=1, verbose=0)[0]
                roiN = int(rois.shape[0])
                np.save(os.path.join(result_dir, f"{name}_proposals.npy"), rois)
                print(f"[DEBUG] {name}: {roiN} proposals saved")
                # Compute IoU with GT boxes for debugging
                try:
                    gt_boxes, _, _ = self.test_dataset.load_data(image_id)
                    if gt_boxes is not None and gt_boxes.size > 0:
                        # Normalize gt_boxes to [0,1] for IoU with normalized rois
                        image_shape = self.config.IMAGE_SHAPE[:3]
                        gt_boxes_norm = utils.norm_boxes(gt_boxes.astype(np.float32), image_shape)
                        overlaps = overlaps_graph(rois, gt_boxes_norm)
                        mean_iou = float(np.mean(np.max(overlaps, axis=1))) if overlaps.size else 0.0
                        print(f"[DEBUG] {name} proposals mean IoU with GT: {mean_iou:.3f}")
                except Exception as e:
                    print(f"[DEBUG] IoU computation failed: {e}")
            except Exception as e:
                print(f"[WARN] ROI probe failed for {name}: {e}")
        if cls_probe:
            try:
                cls = cls_probe.predict(inputs, batch_size=1, verbose=0)[0]
                maxClsProb = float(np.max(cls[:, 1])) if cls.shape[1] > 1 and cls.size else 0.0
                np.save(os.path.join(result_dir, f"{name}_class_probs.npy"), cls)
                print(f"[DEBUG] {name}: max class prob: {maxClsProb:.3f}")
            except Exception as e:
                print(f"[WARN] Class probe failed for {name}: {e}")

        # Main inference
        outs = self.keras_model.predict(inputs, batch_size=1, verbose=0)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        detections = outs[det_idx]
        mrcnn_mask = outs[mask_idx]

        # Debug raw detections
        max_raw_score = np.max(detections[:, 7]) if detections.size else 0.0
        print(f"[DEBUG] {name}: Max raw detection score: {max_raw_score:.3f}")

        # Unmold detections
        pd_boxes, pd_scores, pd_class_ids, pd_masks, pd_segs = self.unmold_detections(detections[0], mrcnn_mask[0])
        pred_inst = int(pd_boxes.shape[0])
        print(f"[DEBUG] {name}: {pred_inst} detections after unmold")

        # Ground truth
        try:
            gt_boxes, gt_class_ids, gt_masks = self.test_dataset.load_data(image_id, masks_needed=True)
            print(f"[DEBUG] {name}: GT boxes: {gt_boxes.shape[0] if gt_boxes is not None else 0}, "
                  f"GT masks: {gt_masks.shape[-1] if gt_masks is not None else 0}")
        except TypeError:
            gt_boxes, gt_class_ids, gt_masks = self.test_dataset.load_data(image_id)
            print(f"[DEBUG] {name}: GT boxes (fallback): {gt_boxes.shape[0] if gt_boxes is not None else 0}")

        # Metrics with NaN handling
        if (gt_masks is None or gt_masks.size == 0 or gt_masks.shape[-1] == 0) and pred_inst == 0:
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
                prec50 = prec50 if not np.isnan(prec50) else 0.0  # Handle NaN
                miou = float(np.mean(ious)) if len(ious) else 0.0

            if gt_masks is None or gt_masks.size == 0 or pd_masks is None or pd_masks.size == 0:
                pxP = pxR = pxF1 = 0.0
                dice_vals = []
            else:
                pxP, pxR, pxF1 = self._pixelwise_metrics(pd_masks > 0.5, gt_masks > 0)  # Use binary
                dice_vals = self._instance_dice(pd_masks, gt_masks)
            dice_mean = float(np.mean(dice_vals)) if len(dice_vals) else 0.0
            dice_std = float(np.std(dice_vals)) if len(dice_vals) else 0.0
            dice_n = int(len(dice_vals))

        # Print per-image metrics
        print(f"{name} AP50:{ap50:.3f} P:{prec50:.3f} R:{rec50:.3f} mIoU:{miou:.3f} "
              f"pxP:{pxP:.3f} pxR:{pxR:.3f} pxF1:{pxF1:.3f} Dice:{dice_mean:.3f}±{dice_std:.3f} inst:{pred_inst}")

        # Save visualization for debugging
        try:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            input_slice = inputs[0][0][..., 0].mean(axis=2)  # Mean over Z for vis
            ax[0].imshow(input_slice, cmap='gray')
            ax[0].set_title("Input Image")

            # GT mask: sum over instances if exists, else zeros
            gt_vis = np.any(gt_masks > 0, axis=-1).mean(axis=2) if gt_masks is not None and gt_masks.shape[
                -1] > 0 else np.zeros_like(input_slice)
            ax[1].imshow(gt_vis, cmap='jet')
            ax[1].set_title("GT Mask")

            # Pred seg: mean over Z
            ax[2].imshow(pd_segs.mean(axis=2), cmap='jet')
            ax[2].set_title("Predicted Seg")
            plt.savefig(os.path.join(result_dir, f"{name}_vis.png"))
            plt.close()
            print(f"[DEBUG] Visualization saved for {name}")
        except Exception as e:
            print(f"[WARN] Visualization failed for {name}: {e}")

        # Save outputs
        try:
            self.save_classes_and_boxes(pd_class_ids, pd_boxes, name)
            imsave(os.path.join(result_dir, f"{name}.tiff"), pd_segs.astype(np.uint8), check_contrast=False)
            print(f"[DEBUG] Outputs saved for {name}")
        except Exception as e:
            print(f"[WARN] Save outputs failed for {name}: {e}")

        return [name, pred_inst, ap50, prec50, rec50, miou, pxP, pxR, pxF1, dice_mean, dice_std, dice_n,
                (roiN if roiN is not None else -1), (maxClsProb if maxClsProb is not None else -1.0)]

    def evaluate(self):
        """
        Inference and evaluation with metrics, using fixed z-score normalization,
        proper multiprocessing, and debugging outputs.
        """
        import os
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import keras
        from core.data_generators import MrcnnGenerator
        from core.utils import compute_ap
        import multiprocessing as mp
        from functools import partial

        assert self.config.MODE == "inference", "Create model in inference mode."

        # Load weights
        rpn_w = getattr(self.config, "RPN_WEIGHTS", None)
        if rpn_w:
            try:
                self.keras_model.load_weights(rpn_w, by_name=True)
                print("[RPN] loaded by_name")
            except Exception as e:
                print(f"[RPN] load failed ({type(e).__name__}): {e}")

        head_w = getattr(self.config, "HEAD_WEIGHTS", None)
        if head_w:
            try:
                self.keras_model.load_weights(head_w, by_name=True)
                print("[HEAD] loaded by_name")
            except Exception as e:
                print(f"[HEAD] load failed ({type(e).__name__}): {e}")

        # Force detection thresholds
        try:
            det_layer = self.keras_model.get_layer("mrcnn_detection")
            if hasattr(self.config, "DETECTION_MIN_CONFIDENCE"):
                det_layer.detection_min_confidence = float(self.config.DETECTION_MIN_CONFIDENCE)
            if hasattr(self.config, "DETECTION_NMS_THRESHOLD"):
                det_layer.detection_nms_threshold = float(self.config.DETECTION_NMS_THRESHOLD)
            print(f"[INFO] Detection thresholds (forced): min_conf={det_layer.detection_min_confidence} "
                  f"nms_thr={det_layer.detection_nms_threshold}")
        except Exception:
            print("[WARN] Failed to set detection thresholds")

        # Generator with fixed z-score normalization
        class NormalizedMrcnnGenerator(MrcnnGenerator):
            def load_image_gt(self, dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = super().load_image_gt(
                    dataset, config, image_id, augment, augmentation, use_mini_mask
                )
                mu, sigma = np.mean(image), np.std(image)
                image = (image - mu) / sigma if sigma > 0 else image
                return image, image_meta, gt_class_ids, gt_boxes, gt_masks

        gen = NormalizedMrcnnGenerator(dataset=self.test_dataset, config=self.config,
                                       shuffle=False, batch_size=1, training=False)

        # Output indices
        out_names = [t.name.split("/")[0] for t in self.keras_model.outputs]
        det_idx = out_names.index("mrcnn_detection") if "mrcnn_detection" in out_names else 0
        mask_idx = out_names.index("mrcnn_mask") if "mrcnn_mask" in out_names else min(3, len(out_names) - 1)

        # Probe models for debugging
        KM = keras.models
        roi_probe = None
        cls_probe = None
        try:
            roi_probe = KM.Model(self.keras_model.inputs, self.keras_model.get_layer("ROI").output)
        except Exception:
            print("[WARN] Failed to create ROI probe")
        try:
            cls_probe = KM.Model(self.keras_model.inputs, self.keras_model.get_layer("mrcnn_class").output)
        except Exception:
            print("[WARN] Failed to create class probe")

        result_dir = getattr(self.config, "OUTPUT_DIR", "./")
        os.makedirs(result_dir, exist_ok=True)
        cols = ["name", "inst", "AP50", "P50", "R50", "mIoU", "pxP", "pxR", "pxF1", "DiceMean", "DiceStd", "DiceN",
                "roiN", "maxCls"]
        df = pd.DataFrame(columns=cols)

        # Sort image IDs to ensure consistent order
        ids = sorted(list(self.test_dataset.image_ids))  # Explicit sorting
        print(f"Evaluating {len(ids)} images")

        # Multiprocessing or single-threaded
        results = []
        if os.name != 'nt':
            try:
                process_func = partial(self.process_image, generator=gen, result_dir=result_dir,
                                       roi_probe=roi_probe, cls_probe=cls_probe, det_idx=det_idx, mask_idx=mask_idx)
                with mp.Pool(processes=mp.cpu_count() // 2) as pool:
                    results = list(tqdm(pool.imap(process_func, ids), total=len(ids), desc="Evaluating"))
            except Exception as e:
                print(f"[WARN] Multiprocessing failed: {e}. Falling back to single-threaded.")
                results = [self.process_image(id, gen, result_dir, roi_probe, cls_probe, det_idx, mask_idx)
                           for id in tqdm(ids, desc="Evaluating")]
        else:
            results = [self.process_image(id, gen, result_dir, roi_probe, cls_probe, det_idx, mask_idx)
                       for id in tqdm(ids, desc="Evaluating")]

        # Populate DataFrame
        for res in results:
            df.loc[len(df.index)] = res

        # Save report
        csv_path = os.path.join(result_dir, "report.csv")
        try:
            df.to_csv(csv_path, index=False)
            print(f"[EVAL] Saved report: {csv_path}")
        except Exception as e:
            print(f"[WARN] CSV save failed: {e}")

        # Print mean metrics to console
        try:
            with np.errstate(invalid="ignore"):
                mean_metrics = df[["AP50", "P50", "R50", "mIoU", "pxP", "pxR", "pxF1", "DiceMean"]].mean(
                    numeric_only=True)
                print("\n[EVAL] Mean Metrics:")
                for metric, value in mean_metrics.items():
                    print(f"{metric}: {value:.4f}")
                # Additional debugging stats
                print(f"Mean instances per image: {df['inst'].mean():.2f}")
                print(f"Mean ROIs per image: {df['roiN'].mean():.2f}")
                print(f"Mean max class prob: {df['maxCls'].mean():.3f}")
        except Exception as e:
            print(f"[WARN] Failed to print metrics: {e}")
            print("[EVAL] DataFrame contents:")
            print(df)

        return df

    def _pixelwise_metrics(self, pred_bin, gt_bin):
        """Calculate pixelwise precision, recall, and F1 score."""
        tp = np.logical_and(pred_bin, gt_bin).sum(dtype=np.int64)
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum(dtype=np.int64)
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum(dtype=np.int64)
        prec = (tp / (tp + fp + 1e-9)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn + 1e-9)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec + 1e-9)) if (prec + rec) > 0 else 0.0
        return float(prec), float(rec), float(f1)

    def _instance_dice(self, pred_masks, gt_masks, iou_thr=0.5):
        """Calculate instance-level DICE score with IoU-based matching."""
        K = int(pred_masks.shape[-1]) if pred_masks is not None and pred_masks.ndim == 4 else 0
        G = int(gt_masks.shape[-1]) if gt_masks is not None and gt_masks.ndim == 4 else 0
        if K == 0 or G == 0:
            return 0.0, 0.0, 0
        P = pred_masks.reshape((-1, K)).astype(np.bool_)
        T = gt_masks.reshape((-1, G)).astype(np.bool_)
        P_sum = P.sum(axis=0).astype(np.int64)
        T_sum = T.sum(axis=0).astype(np.int64)
        inter = (P.T @ T).astype(np.int64)
        union = (P_sum[:, None] + T_sum[None, :]) - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = inter / np.maximum(union, 1)
        used_p = np.zeros((K,), dtype=np.bool_)
        used_g = np.zeros((G,), dtype=np.bool_)
        dices = []
        while True:
            iou_mask = iou.copy()
            iou_mask[used_p, :] = -1.0
            iou_mask[:, used_g] = -1.0
            k, g = np.unravel_index(np.argmax(iou_mask), iou_mask.shape)
            if iou_mask[k, g] < iou_thr:
                break
            denom = P_sum[k] + T_sum[g]
            d = (2.0 * inter[k, g] / denom) if denom > 0 else 0.0
            dices.append(float(d))
            used_p[k] = True
            used_g[g] = True
        if len(dices) == 0:
            return 0.0, 0.0, 0
        return float(np.mean(dices)), float(np.std(dices)), int(len(dices))

    def save_classes_and_boxes(self, pd_class_ids, pd_boxes, name):
        
        cab_df = pd.DataFrame(
            {
                "class": [],
                "y1" : [],
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
            
        cab_df.to_csv(f"{self.config.OUTPUT_DIR}{name}.csv")

    def unmold_detections(self, detections, mrcnn_mask):
        """
        Process raw network outputs into usable detection results.

        Args:
            detections: [N, (y1, x1, z1, y2, x2, z2, class_id, score)] in normalized coordinates
            mrcnn_mask: [N, height, width, depth, num_classes]

        Returns:
            boxes: [N, (y1, x1, z1, y2, x2, z2)] in pixel coordinates
            scores: [N] confidence scores
            class_ids: [N] class IDs
            masks: [height, width, depth, N] binary instance masks
            segs: [height, width, depth] segmentation map with instance IDs
        """
        # Get valid detections (class_id > 0)
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        print(f"[DEBUG] Detections before class filter: {detections.shape[0]}, after: {N}")

        # Extract data from detections
        boxes = detections[:N, :6]
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]

        # Get class-specific masks (use class_id to index into mask channels)
        masks = mrcnn_mask[np.arange(N), ..., class_ids]

        # Convert normalized boxes to pixel coordinates
        original_image_shape = self.config.IMAGE_SHAPE[:3]
        boxes = utils.denorm_boxes(boxes, original_image_shape)

        # Filter out boxes with zero area
        exclude_ix = np.where(
            (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0
        )[0]
        print(f"[DEBUG] Detections before zero-area filter: {N}, excluded: {exclude_ix.shape[0]}")

        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = boxes.shape[0]

        # Create full-sized instance masks
        full_masks = np.zeros((*original_image_shape, N), dtype=np.float32)
        for i in range(N):
            full_masks[..., i] = utils.unmold_mask(masks[i], boxes[i], original_image_shape)

        # Create segmentation map with instance IDs
        segs = np.zeros(original_image_shape, dtype=np.uint16)
        for i in range(N):
            # Process in reverse order so earlier instances (lower indices) appear on top
            mask = full_masks[..., N - i - 1] > 0.5
            segs = np.where(mask, i + 1, segs)

        return boxes, scores, class_ids, full_masks, segs

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:3])
        return self._anchor_cache[tuple(image_shape)]


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
        [image_id] +  # size=1
        list(original_image_shape) +  # size=4
        list(image_shape) +  # size=4
        list(window) +  # size=6 (y1, x1, z1, y2, x2, z2) in image coordinates
        [scale] +  # size=1
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

    Args:
        boxes: [..., (y1, x1, z1, y2, x2, z2)] in pixel coordinates
        shape: [..., (height, width, depth)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    shape: [..., (height, width, depth)] in pixels

    Note: In pixel coordinates (y2, x2, z2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, z1, y2, x2, z2)] in pixel coordinates
    """
    h, w, d = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([h, w, d, h, w, d], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
