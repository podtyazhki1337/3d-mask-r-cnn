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
try:
    # если TF2 и eager уже включён, выключим его заранее
    if hasattr(tf, "executing_eagerly") and tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
except Exception as _e:
    pass
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
try:
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    K.set_session(sess)
except Exception:
    pass
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
        import tensorflow as tf
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


def fpn_classifier_graph_with_RoiAlign(rois, feature_maps, image_meta,
                         pool_size, num_classes, fc_layers_size, train_bn=True):
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
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, dz, log(dh), log(dw) log(dd))] Deltas to apply to
                     proposal boxes
    """
    
    s = K.int_shape(rois)

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
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


def build_fpn_mask_graph_with_RoiAlign(rois, feature_maps, image_meta,
                         pool_size, num_classes, conv_channel, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, z1, y2, x2, z2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv3D(conv_channel, (3, 3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
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


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, bbox_std_dev, 
                            detection_min_confidence, detection_max_instances,
                            detection_nms_threshold):
    """
    Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, dz, log(dh), log(dw), log(dd) )]. Class-specific
                bounding box deltas.
        window: (y1, x1, z1, y2, x2, z2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, z1, y2, x2, z2, class_id, score)] where
        coordinates are normalized.
    """

    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)

    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * bbox_std_dev)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]

    # Filter out low confidence boxes
    if detection_min_confidence:
        conf_keep = tf.where(class_scores >= detection_min_confidence)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""

        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        # Apply NMS
        class_keep = custom_op.non_max_suppression_3d(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=detection_max_instances,
            iou_threshold=detection_nms_threshold
        )

        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        # Pad with -1 so returned tensors have the same shape
        gap = detection_max_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)

        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([detection_max_instances])

        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)

    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]

    # Keep top detections
    roi_count = detection_max_instances
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat(
        [
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = detection_max_instances - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

    return detections


class DetectionLayer(KE.Layer):
    """
    Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, z1, y2, x2, z2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, bbox_std_dev, detection_min_confidence, 
                 detection_max_instances, detection_nms_threshold, 
                 images_per_gpu, batch_size, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.bbox_std_dev = bbox_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = detection_max_instances
        self.detection_nms_threshold = detection_nms_threshold
        self.images_per_gpu = images_per_gpu
        self.batch_size = batch_size

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:3])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(
                x, 
                y, 
                w, 
                z, 
                self.bbox_std_dev,
                self.detection_min_confidence,
                self.detection_max_instances,
                self.detection_nms_threshold
            ),
            self.images_per_gpu
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, z1, y2, x2, z2, class_id, class_scores)] in
        # normalized coordinates
        detections_batch = tf.reshape(detections_batch, [self.batch_size, self.detection_max_instances, 8])

        return detections_batch

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
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
        train_dataset.prepare()
        train_dataset.filter_positive()
        test_dataset = ToyDataset()
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
            workers=3,
            use_multiprocessing=True,
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
        pos_scores, neg_scores = [], []
        for i in range(self.steps):
            try:
                x, _ = self._get_batch(i)
            except StopIteration:
                break
            outs = self.model.predict_on_batch(x)
            names = getattr(self.model, "output_names", [])
            oi = names.index("mrcnn_obj") if "mrcnn_obj" in names else -3
            mrcnn_obj = outs[oi]   # [B,T]
            tci = x[3]               # target_class_ids [B,T]
            pos = mrcnn_obj[tci > 0]; neg = mrcnn_obj[tci <= 0]
            pos_scores.extend(list(pos)); neg_scores.extend(list(neg))
        if pos_scores and neg_scores:
            pos_scores, neg_scores = np.array(pos_scores), np.array(neg_scores)
            auc = (np.argsort(np.argsort(np.r_[pos_scores, neg_scores])).astype(np.float64)[:len(pos_scores)].sum()
                   - len(pos_scores)*(len(pos_scores)+1)/2.0) / (len(pos_scores)*len(neg_scores) + 1e-9)
            print(f"[HEAD][epoch {epoch+1}] obj_pos_mean={pos_scores.mean():.3f}  "
                  f"obj_neg_mean={neg_scores.mean():.3f}  AUC≈{auc:.3f}")

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
        import keras as KM
        import keras.layers as KL

        assert self.config.MODE in ["training", "targeting"], "HEAD expects training/targeting mode"

        # Входы
        input_rois_aligned = KL.Input(
            shape=[self.config.TRAIN_ROIS_PER_IMAGE,
                   self.config.POOL_SIZE,
                   self.config.POOL_SIZE,
                   self.config.POOL_SIZE,
                   self.config.TOP_DOWN_PYRAMID_SIZE],
            name="input_rois_aligned"
        )
        input_mask_aligned = KL.Input(
            shape=[self.config.TRAIN_ROIS_PER_IMAGE,
                   self.config.MASK_POOL_SIZE,
                   self.config.MASK_POOL_SIZE,
                   self.config.MASK_POOL_SIZE,
                   self.config.TOP_DOWN_PYRAMID_SIZE],
            name="input_mask_aligned"
        )
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")
        input_target_class_ids = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, ], name="input_target_class_ids")
        input_target_bbox = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, 6], name="input_target_bbox")
        input_target_mask = KL.Input(shape=[self.config.TRAIN_ROIS_PER_IMAGE, *self.config.MASK_SHAPE, 1],
                                     name="input_target_mask")

        active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

        # === Heads (BN фризим жёстко для стабильности при маленьком batch) ===
        mrcnn_class_logits, mrcnn_prob, mrcnn_bbox = fpn_classifier_graph(
            y=input_rois_aligned,
            pool_size=self.config.POOL_SIZE,
            num_classes=self.config.NUM_CLASSES,
            fc_layers_size=self.config.HEAD_CONV_CHANNEL,
            train_bn=False
        )
        mrcnn_mask = build_fpn_mask_graph(
            y=input_mask_aligned,
            num_classes=self.config.NUM_CLASSES,
            conv_channel=self.config.HEAD_CONV_CHANNEL,
            train_bn=False
        )

        # Лоссы (как у тебя)
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [input_target_bbox, input_target_class_ids, mrcnn_bbox])
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [input_target_class_ids, mrcnn_class_logits, active_class_ids])
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [input_target_mask, input_target_class_ids, mrcnn_mask])
        # === ДОП. «DETECTION SCORE» (objness) + BCE-лосс ===
        # s_obj = 1 - p(bg). Целевая метка: 1 для (class_id>0), 0 для BG.
        mrcnn_obj = KL.Lambda(lambda p: 1.0 - p[..., 0], name="mrcnn_obj")(mrcnn_prob)  # [B, T]
        target_obj = KL.Lambda(lambda y: K.cast(K.greater(y, 0), "float32"), name="target_obj")(input_target_class_ids)

        def bce_obj_loss(args):
            y_true, y_pred = args
            return K.mean(keras.losses.binary_crossentropy(y_true, y_pred))
        obj_loss_raw = KL.Lambda(bce_obj_loss, name="mrcnn_obj_loss")([target_obj, mrcnn_obj])
        margin=0.0

        def margin_loss_fn(args):
            # args: [target_class_ids, class_logits]
            tci, logits = args  # [B,T], [B,T,C]
            # выберем логит целевого класса и логит фона
            tci_i = K.cast(tci, "int32")
            batch = K.shape(logits)[0]
            T = K.shape(logits)[1]
            C = K.shape(logits)[2]
            # индексы для gather_nd
            b_idx = K.tile(K.reshape(K.arange(0, batch), (-1, 1, 1)), (1, T, 1))  # [B,T,1]
            t_idx = K.tile(K.reshape(K.arange(0, T), (1, -1, 1)), (batch, 1, 1))  # [B,T,1]
            idx = K.concatenate([b_idx, t_idx, K.expand_dims(tci_i, -1)], axis=-1)  # [B,T,3]
            tgt_logit = tf.gather_nd(logits, idx)  # [B,T]
            bg_logit = logits[..., 0]  # [B,T]

            pos_mask = K.cast(K.greater(tci_i, 0), "float32")
            neg_mask = 1.0 - pos_mask

            # Для позитивов хотим: tgt_logit - bg_logit >= margin
            pos_term = K.relu(margin - (tgt_logit - bg_logit)) * pos_mask
            # Для негативов хотим: bg_logit - max(logits_fg) >= margin
            fg_logits = logits[..., 1:]
            max_fg = K.max(fg_logits, axis=-1)
            neg_term = K.relu(margin - (bg_logit - max_fg)) * neg_mask

            # усредняем только по присутствующим
            pos_den = K.maximum(K.sum(pos_mask), K.epsilon())
            neg_den = K.maximum(K.sum(neg_mask), K.epsilon())
            return (K.sum(pos_term) / pos_den + K.sum(neg_term) / neg_den) * 0.5

        if margin > 0.0:
            margin_loss = KL.Lambda(margin_loss_fn, name="mrcnn_margin_loss")(
                [input_target_class_ids, mrcnn_class_logits])
        else:
            margin_loss = KL.Lambda(lambda x: K.constant(0.0), name="mrcnn_margin_loss")(mrcnn_class_logits)

        model = KM.Model(
            [input_rois_aligned, input_mask_aligned, input_image_meta,
             input_target_class_ids, input_target_bbox, input_target_mask],
            [mrcnn_class_logits, mrcnn_prob, mrcnn_bbox, mrcnn_mask,
             class_loss, bbox_loss, mask_loss, mrcnn_obj, obj_loss_raw, margin_loss],
            name='head_training'
        )

        # Регистрируем лоссы
        loss_names = ["mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss",
                      "mrcnn_obj_loss", "mrcnn_margin_loss"]
        for name in loss_names:
            layer = model.get_layer(name)
            # scalar per batch
            weighted = tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.0)
            model.add_loss(weighted)

        # L2 regularization (кроме gamma/beta BN)
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name
        ]
        if reg_losses:
            model.add_loss(tf.add_n(reg_losses))

        # Optimizer
        opt_cfg = getattr(self.config, "OPTIMIZER",
                          {"name": "SGD", "parameters": {"learning_rate": 1e-3, "momentum": 0.9}})
        if str(opt_cfg.get("name", "SGD")).lower() == "adam":
            optimizer = keras.optimizers.Adam(**opt_cfg.get("parameters", {}))
        else:
            optimizer = keras.optimizers.SGD(**opt_cfg.get("parameters", {}))

        model.compile(optimizer=optimizer, loss=[None] * len(model.outputs))

        # Немного инфо
        try:
            print("[HEAD.build] BN is FROZEN (train_bn=False)")
            print(
                f"[HEAD.build] NUM_CLASSES={self.config.NUM_CLASSES} | TRAIN_ROIS_PER_IMAGE={self.config.TRAIN_ROIS_PER_IMAGE} "
                f"| POOL={self.config.POOL_SIZE} | MASK_POOL={self.config.MASK_POOL_SIZE}")
        except Exception:
            pass

        print("[HEAD.build] exit (model created)", flush=True)
        return model

    def compile(self):
        """
        Готовим модель к обучению: добавляем все доступные head-лоссы,
        L2 регуляризацию и компилируем. Метрики — такие же лоссы (с весами).
        """
        self.keras_model.metrics_tensors = []

        # Оптимайзер (оставляю твою схему выбора)
        opt_cfg = getattr(self.config, "OPTIMIZER",
                          {"name": "SGD", "parameters": {"learning_rate": 1e-3, "momentum": 0.9}})
        if str(opt_cfg.get("name", "SGD")).upper() == "SGD":
            optimizer = keras.optimizers.SGD(**opt_cfg.get("parameters", {}))
        elif str(opt_cfg.get("name", "SGD")).upper() == "ADADELTA":
            optimizer = keras.optimizers.Adadelta(**opt_cfg.get("parameters", {}))
        else:
            optimizer = keras.optimizers.Adam(**opt_cfg.get("parameters", {}))

        # Сбросим ранее добавленные лоссы у модели (во избежание дублирования)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        # Потенциальные лоссы головы (добавим те, чьи слои действительно есть)
        candidate_losses = [
            "mrcnn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss",
            "mrcnn_obj_loss",
            "mrcnn_margin_loss",
        ]
        present = []
        for name in candidate_losses:
            try:
                self.keras_model.get_layer(name)
                present.append(name)
            except Exception:
                pass

        # Добавляем найденные лоссы с весами
        for name in present:
            layer = self.keras_model.get_layer(name)
            weight = float(getattr(self.config, "LOSS_WEIGHTS", {}).get(name, 1.0))
            loss = tf.reduce_mean(layer.output, keepdims=True) * weight
            self.keras_model.add_loss(loss)

        # L2 regularization (кроме gamma/beta BN)
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        if reg_losses:
            self.keras_model.add_loss(tf.add_n(reg_losses))

        # Компиляция без явного указания целевых лоссов (они добавлены через add_loss)
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Метрики = те же лоссы (взвешенные), чтобы видеть их в прогрессе
        for name in present:
            if name not in self.keras_model.metrics_names:
                layer = self.keras_model.get_layer(name)
                weight = float(getattr(self.config, "LOSS_WEIGHTS", {}).get(name, 1.0))
                self.keras_model.metrics_names.append(name)
                self.keras_model.metrics_tensors.append(layer.output * weight)

    def train(self):
        
        assert self.config.MODE == "training", "Create model in training mode."
        print("[HEAD.train] enter", flush=True)
        # Create Data Generators
        train_generator = HeadGenerator(dataset=self.train_dataset, config=self.config,training=True)



        evaluation = HeadEvaluationCallback(self.keras_model, self.config, self.train_dataset, self.test_dataset)
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='HEAD')
        # Model compilation
        print("[HEAD.train] compile()", flush=True)
        self.compile()
        print("[HEAD.train] compiled", flush=True)
        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # Load weights if self.config.HEAD_WEIGHTS is not None
        if self.config.HEAD_WEIGHTS:
            print("[HEAD] Loading HEAD_WEIGHTS:", self.config.HEAD_WEIGHTS)
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        num_items = len(self.train_dataset.image_info)
        first_ids = list(self.train_dataset.image_ids[:min(5, num_items)])
        print("[HEAD] First image_ids:", first_ids)

        # === Проба ЗАГРУЗКИ ОДНОГО ЭЛЕМЕНТА ИЗ DATASET ===
        try:
            probe_id = int(self.train_dataset.image_ids[0])
            print(f"[HEAD] Probing dataset.load_data(image_id={probe_id}) ...")
            _ra, _ma, _tci, _tb, _tm = self.train_dataset.load_data(probe_id)
            print("[HEAD] dataset.load_data() OK.",
                  "rois_aligned", getattr(_ra, "shape", None),
                  "mask_aligned", getattr(_ma, "shape", None),
                  "target_class_ids", getattr(_tci, "shape", None),
                  "target_bbox", getattr(_tb, "shape", None),
                  "target_mask", getattr(_tm, "shape", None))
        except Exception as e:
            import traceback as _tbx
            print("[HEAD][ERROR] dataset.load_data() failed with exception:", repr(e))
            print(_tbx.format_exc())
            raise

        # === Проба ГЕНЕРАТОРА: выдернуть первый батч ===
        try:
            print("[HEAD] Probing train_generator.__len__() ...")
            gen_len = len(train_generator)
            print(f"[HEAD] HeadGenerator len={gen_len}, BATCH_SIZE={self.config.BATCH_SIZE}")
            if gen_len <= 0:
                raise RuntimeError("[HEAD] HeadGenerator has zero length. Проверь image_ids и BATCH_SIZE.")

            print("[HEAD] Fetching first batch via train_generator.__getitem__(0) ...")
            inputs, outputs = train_generator.__getitem__(0)
            print("[HEAD] First batch fetched.")
            for i, arr in enumerate(inputs):
                shp = getattr(arr, "shape", None)
                dtype = getattr(arr, "dtype", None)
                print(f"[HEAD]   input[{i}]: shape={shp}, dtype={dtype}")
            print("[HEAD] outputs len:", len(outputs))
        except Exception as e:
            import traceback as _tbx
            print("[HEAD][ERROR] train_generator first batch failed:", repr(e))
            print(_tbx.format_exc())
            raise

        # === Явные шаги эпохи и «безопасный» fit ===
        steps_per_epoch = max(1, gen_len)
        print(f"[HEAD] steps_per_epoch={steps_per_epoch}")
        # Training loop
        print("[HEAD.train] start fit_generator", flush=True)
        val_seq = HeadGenerator(dataset=self.train_dataset, config=self.config, shuffle=False, training=False)
        HeadDet = HeadObjScoreMonitor(val_seq, steps=20)
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=[evaluation, save_weights,HeadDet],
            validation_data=None,
            max_queue_size=1,
            workers=0,
            use_multiprocessing=False,
            verbose=1,
        )
        print("[HEAD.train] fit_generator returned", flush=True)

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

    def _get_layer_by_suffix(self, suffix):
        for layer in self.keras_model.layers:
            if layer.name.endswith(suffix):
                return layer
        return None

    def _copy_weights_by_suffix(self, src_model, suffixes):
        imported, missed = [], []
        for suf in suffixes:
            src = None
            for l in src_model.layers:
                if l.name.endswith(suf):
                    src = l
                    break
            dst = self._get_layer_by_suffix(suf)
            if src is not None and dst is not None:
                try:
                    dst.set_weights(src.get_weights())
                    imported.append((src.name, dst.name))
                except Exception as e:
                    missed.append((suf, f"shape mismatch: {str(e)}"))
            else:
                missed.append((suf, "not found (src or dst)"))
        return imported, missed

    def _force_load_head_weights(self):
        """
        Гарантированная загрузка веса головы из HEAD_WEIGHTS:
        строим мини-модель-скелет (те же имена слоёв), грузим в неё файл весов by_name,
        а затем копируем веса в self.keras_model по суффиксам.
        """
        import keras as KM
        import keras.layers as KL
        head_path = getattr(self.config, "HEAD_WEIGHTS", None)
        if not head_path:
            print("[HEAD] no HEAD_WEIGHTS specified — skip force load")
            return

        try:
            # Входы «как в HEAD.build», но без датасетов
            T = int(getattr(self.config, "TRAIN_ROIS_PER_IMAGE", 128))
            Ctd = int(getattr(self.config, "TOP_DOWN_PYRAMID_SIZE", 256))
            P = int(getattr(self.config, "POOL_SIZE", 7))
            Mp = int(getattr(self.config, "MASK_POOL_SIZE", 14))
            Ms = tuple(getattr(self.config, "MASK_SHAPE", (28, 28, 28)))
            Nc = int(self.config.NUM_CLASSES)

            inp_ra = KL.Input(shape=(T, P, P, P, Ctd), name="input_rois_aligned")
            inp_ma = KL.Input(shape=(T, Mp, Mp, Mp, Ctd), name="input_mask_aligned")
            inp_meta = KL.Input(shape=(self.config.IMAGE_META_SIZE,), name="input_image_meta")
            inp_tci = KL.Input(shape=(T,), name="input_target_class_ids")
            inp_tb = KL.Input(shape=(T, 6), name="input_target_bbox")
            inp_tm = KL.Input(shape=(*Ms, 1,), name="input_target_mask_dummy")  # не используется

            # Heads (важны ИМЕНА слоёв!)
            from core.models import fpn_classifier_graph, build_fpn_mask_graph, parse_image_meta_graph
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(inp_meta)

            cls_logits, cls_prob, bbox = fpn_classifier_graph(
                y=inp_ra,
                pool_size=self.config.POOL_SIZE,
                num_classes=Nc,
                fc_layers_size=self.config.HEAD_CONV_CHANNEL,
                train_bn=False  # всегда фриз BN при HEAD-тренировке/скомпиливании скелета
            )
            mrcnn_mask = build_fpn_mask_graph(
                y=inp_ma,
                num_classes=Nc,
                conv_channel=self.config.HEAD_CONV_CHANNEL,
                train_bn=False
            )

            head_skeleton = KM.Model(
                [inp_ra, inp_ma, inp_meta, inp_tci, inp_tb, inp_tm],
                [cls_logits, cls_prob, bbox, mrcnn_mask],
                name="head_skeleton"
            )
            head_skeleton.load_weights(head_path, by_name=True)
        except Exception as e:
            print(f"[HEAD] skeleton load failed: {e}")
            return

        # Копируем веса по суффиксам имён слоёв
        def _get_layer_by_suffix(model, suffix):
            for l in model.layers:
                if l.name.endswith(suffix):
                    return l
            return None

        imported, missed = [], []
        suffixes = [
            # classifier & bbox
            "mrcnn_class_conv1", "mrcnn_class_bn1",
            "mrcnn_class_conv2", "mrcnn_class_bn2",
            "mrcnn_class_logits", "mrcnn_class",
            "mrcnn_bbox_fc", "mrcnn_bbox",
            # mask
            "mrcnn_mask_conv1", "mrcnn_mask_bn1",
            "mrcnn_mask_conv2", "mrcnn_mask_bn2",
            "mrcnn_mask_conv3", "mrcnn_mask_bn3",
            "mrcnn_mask_conv4", "mrcnn_mask_bn4",
            "mrcnn_mask_deconv", "mrcnn_mask",
        ]
        for suf in suffixes:
            src = _get_layer_by_suffix(head_skeleton, suf)
            dst = _get_layer_by_suffix(self.keras_model, suf)
            if src is not None and dst is not None:
                try:
                    dst.set_weights(src.get_weights())
                    imported.append((src.name, dst.name))
                except Exception as e:
                    missed.append((suf, f"shape mismatch: {e}"))
            else:
                missed.append((suf, "not found (src or dst)"))

        print(f"[HEAD] imported {len(imported)} head layers via skeleton")
        if missed:
            print("[HEAD] missed layers:")
            for suf, why in missed:
                print("   -", suf, "→", why)

    def _head_weight_healthcheck(self):
        """
        Печатаем нормы весов ключевых слоёв головы, чтобы убедиться, что веса загружены (не константы).
        """
        import numpy as np
        keys = [
            "mrcnn_class_conv1", "mrcnn_class_conv2",
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
            "mrcnn_mask_conv1", "mrcnn_mask_conv2", "mrcnn_mask_conv3", "mrcnn_mask_conv4",
            "mrcnn_mask_deconv", "mrcnn_mask",
        ]
        print("[HEAD] weights healthcheck (L2-norms):")
        for suf in keys:
            l = self._get_layer_by_suffix(suf)
            if l is None:
                print(f"  {suf:22s} : MISSING")
                continue
            try:
                ws = l.get_weights()
                if not ws:
                    print(f"  {suf:22s} : empty")
                    continue
                norms = [float(np.linalg.norm(w)) for w in ws]
                print(f"  {suf:22s} : " + ", ".join(f"{n:.4f}" for n in norms))
            except Exception as e:
                print(f"  {suf:22s} : error {e}")
    def prepare_datasets(self):

        # Create Datasets
        train_dataset = ToyDataset()
        train_dataset.load_dataset(data_dir=self.config.DATA_DIR)
        train_dataset.prepare()
        train_dataset.filter_positive()
        test_dataset = ToyDataset()
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

    def build(self):
        """
        Build Mask R-CNN architecture.
        """

        assert self.config.MODE in ['training', 'inference']
        
        # Image size must be dividable by 2 multiple times
        # h, w, d = self.config.IMAGE_SHAPE[:3]
        # if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6) or d / 2 ** 6 != int(d / 2 ** 6):
        #     raise Exception("Image size must be dividable by 2 at least 6 times "
        #                     "to avoid fractions when downscaling and upscaling."
        #                     "For example, use 256, 320, 384, 448, 512, ... etc. ")
        h, w, d = self.config.IMAGE_SHAPE[:3]
        if (h % 64) or (w % 64):
            raise ValueError("IMAGE_SHAPE height & width must be multiples of 64")
        # Inputs
        input_image = KL.Input(shape=[*self.config.IMAGE_SHAPE], name="input_image")
        input_image_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="input_image_meta")

        if self.config.MODE == "training":
            
            # RPN targets
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 6], name="input_rpn_bbox", dtype=tf.float32)

            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

            if self.config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[*self.config.MINI_MASK_SHAPE, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[*self.config.IMAGE_SHAPE[:-1], None], name="input_gt_masks", dtype=bool)

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
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if self.config.MODE == "training":

            anchors = self.get_anchors(self.config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

        elif self.config.MODE == "inference":

            input_anchors = KL.Input(shape=[None, 6], name="input_anchors")
            anchors = input_anchors

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
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.config.MODE == "training" else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            images_per_gpu=self.config.IMAGES_PER_GPU,
            rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
            image_depth=self.config.IMAGE_DEPTH,
            name="ROI"
        )([rpn_class, rpn_bbox, anchors])

        if self.config.MODE == "training":

            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            # Generate detection targets
            rois, _, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.ROI_POSITIVE_RATIO,
                self.config.BBOX_STD_DEV,
                self.config.USE_MINI_MASK,
                self.config.MASK_SHAPE,
                self.config.IMAGES_PER_GPU,
                name="proposal_targets"
            )([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads: classifier and regressor
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(
                rois=rois, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta, 
                pool_size=self.config.POOL_SIZE,
                num_classes=self.config.NUM_CLASSES, 
                fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,
                train_bn=self.config.TRAIN_BN                
            )

            # Network Heads: segmentation
            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(
                rois=rois, 
                feature_maps=mrcnn_feature_maps, 
                image_meta=input_image_meta,
                pool_size=self.config.MASK_POOL_SIZE, 
                num_classes=self.config.NUM_CLASSES,
                conv_channel=self.config.HEAD_CONV_CHANNEL,
                train_bn=self.config.TRAIN_BN
            )

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config.IMAGES_PER_GPU, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks,
                      input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            
            model = KM.Model(inputs, outputs, name='mask_rcnn_training')



        elif self.config.MODE == "inference":

            # === Heads: classifier & regressor ===

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_with_RoiAlign(

                rois=rpn_rois,

                feature_maps=mrcnn_feature_maps,

                image_meta=input_image_meta,

                pool_size=self.config.POOL_SIZE,

                num_classes=self.config.NUM_CLASSES,

                fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE,

                train_bn=self.config.TRAIN_BN

            )

            # === Detections ===

            # [batch, num_detections, (y1,x1,z1,y2,x2,z2, class_id, score)]

            detections = DetectionLayer(

                self.config.BBOX_STD_DEV,

                self.config.DETECTION_MIN_CONFIDENCE,

                self.config.DETECTION_MAX_INSTANCES,

                self.config.DETECTION_NMS_THRESHOLD,

                self.config.IMAGES_PER_GPU,

                self.config.BATCH_SIZE,

                name="mrcnn_detection"

            )([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            detection_boxes = KL.Lambda(lambda x: x[..., :6])(detections)

            # === Mask head на финальных боксах ===

            mrcnn_mask = build_fpn_mask_graph_with_RoiAlign(

                rois=detection_boxes,

                feature_maps=mrcnn_feature_maps,

                image_meta=input_image_meta,

                pool_size=self.config.MASK_POOL_SIZE,

                num_classes=self.config.NUM_CLASSES,

                conv_channel=self.config.HEAD_CONV_CHANNEL,

                train_bn=self.config.TRAIN_BN

            )

            # === Основная инференс-модель (как у тебя) ===

            inputs = [input_image, input_image_meta, input_anchors]

            outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,

                       rpn_rois, rpn_class, rpn_bbox]

            model = KM.Model(inputs, outputs, name='mask_rcnn_inference')
            # DEBUG: модель, которая отдаёт признаки после ROIAlign (classifier head)
            try:
                self.keras_roi_debug = KM.Model(
                    [input_image, input_image_meta, input_anchors],
                    model.get_layer("roi_align_classifier").output,
                    name="roi_align_classifier_debug"
                )
            except Exception as e:
                print(f"[DEBUG] ROI debug model build failed: {e}")
                self.keras_roi_debug = None
            # ------------------------------------------------------------------

            # ДОП. модели для "оконного" инференса головы (ничего не ломают):

            # 1) core: выдаёт detections + FPN P2..P5

            self.keras_infer_core = KM.Model(

                [input_image, input_image_meta, input_anchors],

                [detections, P2, P3, P4, P5],

                name="mrcnn_infer_core"

            )

            # 2) mask head eval: из любых ROI + meta + P2..P5 → mrcnn_mask

            #    Собираем по именованным слоям, чтобы использовать те же веса.

            eval_rois = KL.Input(shape=[None, 6], name="eval_rois")  # norm coords

            eval_meta = KL.Input(shape=[self.config.IMAGE_META_SIZE], name="eval_image_meta")

            p2_in = KL.Input(shape=K.int_shape(P2)[1:], name="eval_P2")

            p3_in = KL.Input(shape=K.int_shape(P3)[1:], name="eval_P3")

            p4_in = KL.Input(shape=K.int_shape(P4)[1:], name="eval_P4")

            p5_in = KL.Input(shape=K.int_shape(P5)[1:], name="eval_P5")

            x = model.get_layer("roi_align_mask")([eval_rois, eval_meta, p2_in, p3_in, p4_in, p5_in])

            x = model.get_layer("mrcnn_mask_conv1")(x)

            x = model.get_layer("mrcnn_mask_bn1")(x, training=False);
            x = KL.Activation('relu')(x)

            x = model.get_layer("mrcnn_mask_conv2")(x)

            x = model.get_layer("mrcnn_mask_bn2")(x, training=False);
            x = KL.Activation('relu')(x)

            x = model.get_layer("mrcnn_mask_conv3")(x)

            x = model.get_layer("mrcnn_mask_bn3")(x, training=False);
            x = KL.Activation('relu')(x)

            x = model.get_layer("mrcnn_mask_conv4")(x)

            x = model.get_layer("mrcnn_mask_bn4")(x, training=False);
            x = KL.Activation('relu')(x)

            x = model.get_layer("mrcnn_mask_deconv")(x)

            mrcnn_mask_eval = model.get_layer("mrcnn_mask")(x)

            self.keras_mask_head_eval = KM.Model(

                [eval_rois, eval_meta, p2_in, p3_in, p4_in, p5_in],

                mrcnn_mask_eval,

                name="mask_head_eval"

            )

            # ------------------------------------------------------------------

            return model

    def compile(self):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        self.keras_model.metrics_tensors = []

        if self.config.OPTIMIZER["name"] == "ADADELTA":

            optimizer = keras.optimizers.Adadelta(**self.config.OPTIMIZER["parameters"])

        elif self.config.OPTIMIZER["name"] == "SGD":

            optimizer = keras.optimizers.SGD(**self.config.OPTIMIZER["parameters"])

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        if self.config.LEARNING_LAYERS == "rpn":

            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]

        elif self.config.LEARNING_LAYERS == "heads":

            loss_names = [
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
            
        elif self.config.LEARNING_LAYERS == "all":

            loss_names = [
                "rpn_class_loss", "rpn_bbox_loss",
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
            
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.)
                )
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))
                
    def train(self):

        assert self.config.MODE == "training", "Create model in training mode."

        # Create Data Generators
        train_generator = MrcnnGenerator(dataset=self.train_dataset, config=self.config)

        # Callback for saving weights
        # save_weights = SaveWeightsCallback(self.config.WEIGHT_DIR)
        save_weights = BestAndLatestCheckpoint(save_path=self.config.WEIGHT_DIR, mode='MRCNN')
        # Model compilation
        self.compile()

        # Initialize weight dir
        os.makedirs(self.config.WEIGHT_DIR, exist_ok=True)

        # These conditions allows to customize the weights imports
        if self.config.MASK_WEIGHTS:
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if self.config.RPN_WEIGHTS:
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if self.config.HEAD_WEIGHTS:
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.config.FROM_EPOCH,
            epochs=self.config.FROM_EPOCH + self.config.EPOCHS,
            steps_per_epoch=len(self.train_dataset.image_info),
            callbacks=[save_weights],
            validation_data=None,
            max_queue_size=30,
            workers=workers,
            use_multiprocessing=True,
        )

    def _refine_detections_numpy(self, rpn_rois, mrcnn_class, mrcnn_bbox, image_meta):
        """
        ВНЕГРАФОВЫЙ детектор (fallback), если DetectionLayer дал 0.
        Входы:
          rpn_rois:    [1, R, 6] норм. (y1,x1,z1,y2,x2,z2)
          mrcnn_class: [1, R, C] softmax по классам (bg=0)
          mrcnn_bbox:  [1, R, C, 6] или [1, R, C*6] дельты
          image_meta:  [1, META] (здесь не используем, размеры берём из config)
        Вывод:
          detections_np: [N, 8]  (y1,x1,z1,y2,x2,z2, class_id, score) — НОРМАЛИЗОВАННЫЕ координаты
        """
        import numpy as np
        from core import utils

        # --- параметры ---
        min_conf = float(getattr(self.config, "DETECTION_MIN_CONFIDENCE", 0.15))
        nms_thr = float(getattr(self.config, "DETECTION_NMS_THRESHOLD", 0.45) or 0.45)
        max_inst = int(getattr(self.config, "DETECTION_MAX_INSTANCES", 100))
        bbox_std = np.asarray(self.config.BBOX_STD_DEV, dtype=np.float32)
        H, W, D = self.config.IMAGE_SHAPE[:3]

        # --- приведение форм ---
        rois = np.asarray(rpn_rois[0], dtype=np.float32)  # [R,6] norm
        scores = np.asarray(mrcnn_class[0], dtype=np.float32)  # [R,C]
        deltas = np.asarray(mrcnn_bbox[0], dtype=np.float32)  # [R,C,6] | [R,C*6]
        if deltas.ndim == 2:
            C = scores.shape[1]
            deltas = deltas.reshape((rois.shape[0], C, 6))

        C = scores.shape[1]
        detections_all = []

        # --- по каждому НЕ-фоновому классу ---
        for c in range(1, C):
            cls_scores = scores[:, c]  # [R]
            keep_ix = np.where(cls_scores >= min_conf)[0]
            if keep_ix.size == 0:
                continue

            rois_c = rois[keep_ix]  # [K,6] norm
            deltas_c = deltas[keep_ix, c, :] * bbox_std  # [K,6]

            # denorm → применить дельты → клип в пикселях
            rois_px = utils.denorm_boxes(rois_c, (H, W, D)).astype(np.float32)  # [K,6] pixels
            ref_px = utils.apply_box_deltas_3d(rois_px, deltas_c)  # [K,6] pixels

            # клип к границам изображения
            ref_px[:, 0] = np.clip(ref_px[:, 0], 0, H - 1)
            ref_px[:, 1] = np.clip(ref_px[:, 1], 0, W - 1)
            ref_px[:, 2] = np.clip(ref_px[:, 2], 0, D - 1)
            ref_px[:, 3] = np.clip(ref_px[:, 3], 0, H - 1)
            ref_px[:, 4] = np.clip(ref_px[:, 4], 0, W - 1)
            ref_px[:, 5] = np.clip(ref_px[:, 5], 0, D - 1)

            # фильтр tiny-боксов
            hh = ref_px[:, 3] - ref_px[:, 0]
            ww = ref_px[:, 4] - ref_px[:, 1]
            dd = ref_px[:, 5] - ref_px[:, 2]
            ok = np.where((hh > 1) & (ww > 1) & (dd > 0.5))[0]
            if ok.size == 0:
                continue
            ref_px = ref_px[ok]
            sc = cls_scores[keep_ix][ok]

            # NMS 3D (ваш utils.non_max_suppression_3d)
            ref_px_nms, keep_local = utils.non_max_suppression_3d(ref_px, sc, threshold=nms_thr, max_boxes=max_inst)
            if ref_px_nms.shape[0] == 0:
                continue
            sc_nms = sc[keep_local]

            # нормализация обратно в [0,1], формирование записей
            ref_nm = utils.norm_boxes(ref_px_nms, (H, W, D)).astype(np.float32)
            cls_id = np.full((ref_nm.shape[0], 1), float(c), dtype=np.float32)
            det_c = np.concatenate([ref_nm, cls_id, sc_nms[:, None]], axis=1)  # [M,8]
            detections_all.append(det_c)

        if len(detections_all) == 0:
            return np.zeros((0, 8), dtype=np.float32)

        det = np.vstack(detections_all)  # [N,8]
        order = det[:, 7].argsort()[::-1]
        det = det[order]
        if det.shape[0] > max_inst:
            det = det[:max_inst]
        return det

    def evaluate(self):
        """
        Полный eval с диагностикой ROIAlign и fallback-детектором.
        """
        import os
        import numpy as np
        import pandas as pd
        from skimage.io import imsave
        from core.utils import compute_ap

        assert self.config.MODE == "inference", "Create model in inference mode."

        # --- Загрузка весов (как у тебя) ---
        if getattr(self.config, "MASK_WEIGHTS", None):
            self.keras_model.load_weights(self.config.MASK_WEIGHTS, by_name=True)
        if getattr(self.config, "RPN_WEIGHTS", None):
            self.keras_model.load_weights(self.config.RPN_WEIGHTS, by_name=True)
        if getattr(self.config, "HEAD_WEIGHTS", None):
            self.keras_model.load_weights(self.config.HEAD_WEIGHTS, by_name=True)

        # Проба принудительной прокопировки (не критично, но пусть будет)
        if hasattr(self, "_force_load_head_weights"):
            self._force_load_head_weights()
        if hasattr(self, "_head_weight_healthcheck"):
            self._head_weight_healthcheck()

        # --- Генератор ---
        data_generator = MrcnnGenerator(dataset=self.test_dataset, config=self.config)

        # --- Папка результатов ---
        result_dir = self.config.OUTPUT_DIR
        os.makedirs(result_dir, exist_ok=True)

        # --- Пороговые параметры ---
        MIN_CONF = float(getattr(self.config, "DETECTION_MIN_CONFIDENCE", 0.15))
        NMS_THR = float(getattr(self.config, "DETECTION_NMS_THRESHOLD", 0.45) or 0.45)
        print(f"[INFO] Detection thresholds — MIN_CONF={MIN_CONF:.2f}, NMS_THR={NMS_THR:.2f}")

        # --- Заглушка-объём для пустых случаев ---
        H, W, D = self.config.IMAGE_SHAPE[:3]
        blank_zyx = np.zeros((D, H, W), dtype=np.uint16)

        # --- Таблица результатов ---
        result_dataframe = pd.DataFrame({
            "name": [], "instance_nb": [], "map-50": [],
            "precision-50": [], "recall-50": [], "iou-50": [],
        })

        num_imgs = len(self.test_dataset.image_info)
        printed_debug = False  # диагностику ROI печатаем один раз, чтобы лог не распух

        for i in range(num_imgs):
            # ----- входы -----
            name, inputs = data_generator.get_input_prediction(i)
            name = name.split(".")[0]

            # ----- GT -----
            _, _, gt_boxes, gt_class_ids, gt_masks = data_generator.load_image_gt(i)
            gt_inst = int(gt_masks.shape[-1]) if hasattr(gt_masks, "shape") else 0

            # ----- (опц.) Диагностика ROIAlign -----
            if (not printed_debug) and getattr(self, "keras_roi_debug", None) is not None:
                try:
                    roi_feat = self.keras_roi_debug.predict(inputs, verbose=0)  # [1, R, ph,pw,pd,C]
                    rf = roi_feat[0]  # [R,ph,pw,pd,C]
                    rf_mean = float(np.mean(rf))
                    rf_std = float(np.std(rf))
                    # по каналам усредним дисперсию — важно, чтобы она не была ~0
                    rf_var_per_ch = np.var(rf, axis=(0, 1, 2, 3))  # [C]
                    rf_var_med = float(np.median(rf_var_per_ch))
                    print(
                        f"[DEBUG] ROIAlign features: mean={rf_mean:.4f}, std={rf_std:.4f}, median var/ch={rf_var_med:.6e}")
                    if rf_std < 1e-6 or rf_var_med < 1e-10:
                        print(
                            "[ALERT] ROIAlign features look nearly constant! Check boxes normalization and crop_and_resize_3d.")
                    printed_debug = True
                except Exception as e:
                    print(f"[DEBUG] ROI features probe failed: {e}")
                    printed_debug = True

            # ----- Инференс графа -----
            # out = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]
            out = self.keras_model.predict(inputs, verbose=0)
            detections_batch = out[0]
            mrcnn_class = out[1]
            mrcnn_bbox = out[2]
            mrcnn_mask_batch = out[3]
            rpn_rois_batch = out[4]

            # разворот batch=1
            detections = detections_batch[0]
            probs = mrcnn_class[0]
            rpn_rois = rpn_rois_batch[0]
            mrcnn_mask = mrcnn_mask_batch[0]

            # ---- Диагностика головы ДО DetectionLayer ----
            R = int(rpn_rois.shape[0]) if hasattr(rpn_rois, "shape") else 0
            C = int(probs.shape[1]) if hasattr(probs, "shape") and probs.ndim == 2 else 0
            if C >= 2 and R > 0:
                non_bg = probs[:, 1:]
                best_scores = non_bg.max(axis=1) if non_bg.size else np.zeros((R,), dtype=np.float32)
            else:
                best_scores = np.zeros((R,), dtype=np.float32)
            cnt_ge_min = int((best_scores >= MIN_CONF).sum()) if best_scores.size else 0
            cnt_ge_020 = int((best_scores >= 0.20).sum()) if best_scores.size else 0
            top5 = np.round(np.sort(best_scores)[-5:][::-1], 3).tolist() if best_scores.size else []

            # сколько прошло через DetectionLayer (class_id > 0)
            raw_det = int(np.sum(detections[:, 6] > 0)) if detections.size else 0

            # ----- FALLBACK: если граф дал 0 детекций, пытаемся вне графа -----
            used_fallback = False
            if raw_det <= 0:
                try:
                    core_out = self.keras_infer_core.predict(inputs, verbose=0)  # [detections, P2..P5]
                    P2, P3, P4, P5 = core_out[1], core_out[2], core_out[3], core_out[4]

                    det_np = self._refine_detections_numpy(rpn_rois_batch, mrcnn_class, mrcnn_bbox, inputs[1])
                    if det_np.shape[0] > 0:
                        boxes_nm = det_np[np.newaxis, :, :6]  # [1,N,6]
                        mrcnn_mask = self.keras_mask_head_eval.predict(
                            [boxes_nm, inputs[1], P2, P3, P4, P5], verbose=0
                        )[0]
                        detections = det_np
                        raw_det = int(np.sum(detections[:, 6] > 0))
                        used_fallback = True
                except Exception as e:
                    print(f"[FALLBACK] error: {e}")

            # ----- Если совсем пусто -----
            if raw_det <= 0:
                imsave(f"{result_dir}{name}.tiff", blank_zyx, check_contrast=False)
                print(f"{i + 1}/{num_imgs}  Example: {name}  GT:{gt_inst}  RPN-ROI:{R}  "
                      f"HEAD>={MIN_CONF:.2f}:{cnt_ge_min}  >={0.20:.2f}:{cnt_ge_020}  >0:{R}  "
                      f"RAW det:0  FINAL det:0  avg score:0.000  "
                      f"mAP:0.000  Prec:0.000  Rec:0.000  mIoU:0.000  top5:{top5}")
                if gt_inst > 0:
                    result_dataframe.loc[len(result_dataframe.index)] = [name, gt_inst, 0.0, 0.0, 0.0, 0.0]
                continue

            # ----- Развёртка в пиксели и маски -----
            pd_boxes, pd_scores, pd_class_ids, pd_masks, pd_segs = self.unmold_detections(detections, mrcnn_mask)
            final_det = int(pd_boxes.shape[0]) if hasattr(pd_boxes, "shape") else 0
            avg_sc = float(pd_scores.mean()) if final_det > 0 else 0.0

            # сохранить боксы/классы .csv
            self.save_classes_and_boxes(pd_class_ids, pd_boxes, name)

            # сохранить сегментацию .tiff (Z,Y,X)
            if hasattr(pd_segs, "shape") and pd_segs.ndim == 3 and pd_segs.shape[2] > 0:
                imsave(f"{result_dir}{name}.tiff",
                       pd_segs.transpose(2, 0, 1).astype(np.uint16),
                       check_contrast=False)
            else:
                imsave(f"{result_dir}{name}.tiff", blank_zyx, check_contrast=False)

            # метрики
            map50, precision50, recall50, ious = compute_ap(
                gt_boxes, gt_class_ids, gt_masks, pd_boxes, pd_class_ids, pd_scores, pd_masks, iou_threshold=0.5
            )
            map50 = float(map50) if np.isfinite(map50) else 0.0
            precision50 = float(precision50) if np.isfinite(precision50) else 0.0
            recall50 = float(recall50) if np.isfinite(recall50) else 0.0
            mean_iou = float(np.mean(ious)) if (isinstance(ious, (list, np.ndarray)) and len(ious) > 0) else 0.0
            mean_iou = mean_iou if np.isfinite(mean_iou) else 0.0

            result_dataframe.loc[len(result_dataframe.index)] = [name, gt_inst, map50, precision50, recall50, mean_iou]

            print(f"{i + 1}/{num_imgs}  Example: {name}  GT:{gt_inst}  RPN-ROI:{R}  "
                  f"HEAD>={MIN_CONF:.2f}:{cnt_ge_min}  >={0.20:.2f}:{cnt_ge_020}  >0:{R}  "
                  f"RAW det:{raw_det}  FINAL det:{final_det}  avg score:{avg_sc:.3f}  "
                  f"mAP:{map50:.3f}  Prec:{precision50:.3f}  Rec:{recall50:.3f}  "
                  f"mIoU:{mean_iou:.3f}  top5:{top5}" + ("  [FALLBACK]" if used_fallback else ""))

        # --- сохранить сводный отчёт ---
        result_dataframe.to_csv(f"{result_dir}report.csv", index=None)
        print(result_dataframe.mean())

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
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, z1, y2, x2, z2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, depth, num_classes]
        original_image_shape: [H, W, D, C] Original image shape before resizing
        image_shape: [H, W, D, C] Shape of the image after resizing and padding
        window: [y1, x1, z1, y2, x2, z2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, depth, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        original_image_shape = self.config.IMAGE_SHAPE[:3]
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :6]
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), ..., class_ids]

        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape)

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
        N = boxes.shape[0]

        unmold_masks = np.zeros((*original_image_shape, masks.shape[0]))
        for i in range(N): 
            unmold_masks[..., i] = utils.unmold_mask(masks[i], boxes[i], original_image_shape)

        # Resize masks to original image size and set boundary threshold.
        segs = np.zeros((original_image_shape)).astype(np.uint16)
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[-i-1], boxes[-i-1], original_image_shape)
            segs = np.where(full_mask, i+1, segs)

        return boxes, scores, class_ids, unmold_masks, segs

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
