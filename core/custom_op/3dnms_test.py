import os
import numpy as np
import tensorflow as tf

# Comment the following line to debug TF or libcuda issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


nms3d_module = tf.load_op_library('./non_max_suppression_3d_op.so')

#TestSelectFromThreeClusters
boxes1 = np.array([[0, 0, 0, 1, 1, 1], [0, 0.1, 0, 1, 1.1, 1], [0, -0.1, 0, 1, 0.9, 1],
                   [0, 10, 0, 1, 11, 1], [0, 10.1, 0, 1, 11.1, 1], [0, 100, 0, 1, 101, 1]])
scores1 = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3])
max_output_size1 = 3
boxes1 = tf.dtypes.cast(boxes1, tf.float32)
scores1 = tf.dtypes.cast(scores1, tf.float32)
results1 = nms3d_module.non_max_suppression3d(boxes=boxes1, scores=scores1, max_output_size=max_output_size1)
if (results1.numpy() == np.array([3, 0, 5])).all:
    print('TestSelectFromThreeClusters is OK.')

#TestSelectFromThreeClustersFlippedCoordinates
boxes2 = np.array([[1, 1, 1, 0, 0, 0], [0, 0.1, 1, 1, 1.1, 0], [0, .9, 1, 1, -0.1, 0],
                  [0, 10, 1, 1, 11, 0], [1, 10.1, 1, 0, 11.1, 0], [1, 101, 1, 0, 100, 0]])
boxes2 = tf.dtypes.cast(boxes2, tf.float32)
scores2 = scores1
max_output_size2 = max_output_size1
results2 = nms3d_module.non_max_suppression3d(boxes=boxes2, scores=scores2, max_output_size=max_output_size2)
if (results2.numpy() == np.array([3, 0, 5])).all:
    print('TestSelectFromThreeClustersFlippedCoordinates is OK.')

#TestSelectAtMostTwoBoxesFromThreeClusters
boxes3 = boxes1
scores3 = scores1
max_output_size3 = 2
results3 = nms3d_module.non_max_suppression3d(boxes=boxes3, scores=scores3, max_output_size=max_output_size3)
if (results3.numpy() == np.array([3, 0])).all:
    print('TestSelectAtMostTwoBoxesFromThreeClusters is OK.')

#TestSelectWithNegativeScores
boxes4 = boxes1
scores4 = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]) - 10
scores4 = tf.dtypes.cast(scores4, tf.float32)
max_output_size4 = 6
results4 = nms3d_module.non_max_suppression3d(boxes=boxes4, scores=scores4, max_output_size=max_output_size4)
if (results4.numpy() == np.array([3, 0, 5])).all:
    print('TestSelectWithNegativeScores is OK.')

#TestSelectAtMostThirtyBoxesFromThreeClusters
boxes5 = boxes1
scores5 = scores1
max_output_size5 = 30
results5 = nms3d_module.non_max_suppression3d(boxes=boxes5, scores=scores5, max_output_size=max_output_size5)
if (results5.numpy() == np.array([3, 0, 5])).all:
    print('TestSelectAtMostThirtyBoxesFromThreeClusters is OK.')

#TestSelectSingleBox
boxes6 = np.array([[0, 0, 0, 1, 1, 1]])
scores6 = np.array([0.9])
boxes6 = tf.dtypes.cast(boxes6, tf.float32)
scores6 = tf.dtypes.cast(scores6, tf.float32)
max_output_size6 = 3
results6 = nms3d_module.non_max_suppression3d(boxes=boxes6, scores=scores6, max_output_size=max_output_size6)
if (results6.numpy() == np.array([0])).all:
    print('TestSelectSingleBox is OK.')

#TestSelectFromTenIdenticalBoxes
num_boxes = 10 #10 000 000
boxes7 = np.zeros((num_boxes,6))
for i in range(num_boxes):
    boxes7[i] = np.array([0, 0, 0, 1, 1, 1])
scores7 = 0.9*np.ones(num_boxes)
boxes7 = tf.dtypes.cast(boxes7, tf.float32)
scores7 = tf.dtypes.cast(scores7, tf.float32)
max_output_size7 = 3
results7 = nms3d_module.non_max_suppression3d(boxes=boxes7, scores=scores7, max_output_size=max_output_size7)
if (results7.numpy() == np.array([0])).all:
    print('TestSelectFromTenIdenticalBoxes is OK.')

#TestInconsistentBoxAndScoreShapes
boxes8 = boxes1
scores8 = np.array([0.9, 0.75, 0.6, 0.95, 0.5])
scores8 = tf.dtypes.cast(scores8, tf.float32)
max_output_size8 = 30
try:
    results8 = nms3d_module.non_max_suppression3d(boxes=boxes8, scores=scores8, max_output_size=max_output_size8)
except Exception as e:
    if 'scores has incompatible shape' in str(e):
        print('TestInconsistentBoxAndScoreShapes is OK.')

#TestInvalidIOUThreshold
boxes9 = boxes6
scores9 = scores6
max_output_size9 = max_output_size1
iou9 = 1.2
try:
    results9 = nms3d_module.non_max_suppression3d(boxes=boxes9, scores=scores9, max_output_size=max_output_size9, iou_threshold=iou9)
except Exception as e:
    if 'iou_threshold must be in [0, 1]' in str(e):
        print('TestInvalidIOUThreshold is OK.')

#TestEmptyInput
boxes10 = np.empty([0,6])
scores10 = np.empty([0])
boxes10 = tf.dtypes.cast(boxes10, tf.float32)
scores10 = tf.dtypes.cast(scores10, tf.float32)
max_output_size10 = 30
results10 = nms3d_module.non_max_suppression3d(boxes=boxes10, scores=scores10, max_output_size=max_output_size10)
if (results10.numpy() == np.array([])).all:
    print('TestEmptyInput is OK.')
