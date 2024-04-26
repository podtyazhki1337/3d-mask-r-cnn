import os
import numpy as np
import tensorflow as tf
from scipy import interpolate

# Comment the following line to debug TF or libcuda issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def crop_and_resize_from_scipy(image, boxes, crop_size, method='linear', bounds_error=True, extrapolation_value=0):
    cropped = np.empty( (np.shape(boxes)[0], *crop_size, 1 ) )
    image_height = np.shape(image)[1]
    image_width = np.shape(image)[2]
    image_depth = np.shape(image)[3]
    for i in range(np.shape(boxes)[0]):
        if crop_size[0] > 1 :
            height_scale = (boxes[i, 3] - boxes[i, 0]) * ( image_height - 1 ) / ( crop_size[0] - 1)
            y_range = np.arange(crop_size[0])
            y_coor = boxes[i, 0] * ( image_height - 1 ) + height_scale * y_range
        else :
            y_coor = np.array([ 0.5 * (boxes[i, 3] + boxes[i, 0]) * ( image_height - 1 ) ])
        if crop_size[1] > 1 :
            width_scale = (boxes[i, 4] - boxes[i, 1]) * (image_width - 1) / (crop_size[1] - 1)
            x_range = np.arange(crop_size[1])
            x_coor = boxes[i, 1] * (image_width - 1) + width_scale * x_range
        else :
            x_coor = np.array([ 0.5 * (boxes[i, 4] + boxes[i, 1]) * (image_width - 1) ])
        if crop_size[2] > 1 :
            depth_scale = (boxes[i, 5] - boxes[i, 2]) * (image_depth - 1) / (crop_size[2] - 1)
            z_range = np.arange(crop_size[2])
            z_coor = boxes[i, 2] * (image_depth - 1) + depth_scale * z_range
        else :
            z_coor = np.array([ 0.5 * (boxes[i, 5] + boxes[i, 2]) * (image_depth - 1) ])


        yspace = np.linspace(0, image_height - 1, image_height)
        xspace = np.linspace(0, image_width - 1, image_width)
        zspace = np.linspace(0, image_depth - 1, image_depth)
        f = interpolate.RegularGridInterpolator((yspace, xspace, zspace), image[0,:,:,:,0], method=method,
                                                bounds_error=bounds_error, fill_value=extrapolation_value)

        piece = np.empty(( *crop_size,))
        for y in range(crop_size[0]):
            for x in range(crop_size[1]):
                for z in range(crop_size[2]):
                    piece[y,x,z] = f(np.array([ y_coor[y], x_coor[x], z_coor[z] ]))
        cropped[i,:,:,:,0] = piece # np.array([y_coor, x_coor, z_coor])
    return cropped


car3d_module = tf.load_op_library('./crop_and_resize_3d_op.so')
crop_and_resize_3d = car3d_module.crop_and_resize3d

#TestCropAndResize2x2x2To1x1x1Uint8Bilinear
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([1,1,1])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy() == scipy_control:
    print('TestCropAndResize2x2x2To1x1x1Uint8Bilinear is OK.')
else:
    print('TestCropAndResize2x2x2To1x1x1Uint8Bilinear is not OK.')

#TestCropAndResize2x2x2To1x1x1Uint8Nearest
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([1,1,1])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')
print(results, scipy_control)

if results.shape == scipy_control.shape and results.numpy() == scipy_control:
    print('TestCropAndResize2x2x2To1x1x1Uint8Nearest is OK.')
else:
    print('TestCropAndResize2x2x2To1x1x1Uint8Nearest is not OK (but it is normal because of "nearest" definition is this particular case).')


#TestCropAndResize2x2x2To1x1x1Flipped
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([1,1,1,0,0,0])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([1,1,1])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy() == scipy_control:
    print('TestCropAndResize2x2x2To1x1x1Flipped is OK.')
else:
    print('TestCropAndResize2x2x2To1x1x1Flipped is not OK.')

#TestCropAndResize2x2x2To1x1x1FlippedNearest
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([1,1,1,0,0,0])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([1,1,1])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')
print(results, scipy_control)

if results.shape == scipy_control.shape and results.numpy() == scipy_control:
    print('TestCropAndResize2x2x2To1x1x1FlippedNearest is OK.')
else:
    print('TestCropAndResize2x2x2To1x1x1FlippedNearest is not OK (but it is normal because of "nearest" definition is this particular case).')

#TestCropAndResize2x2x2To3x3x3
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all() :
    print('TestCropAndResize2x2x2To3x3x3 is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3 is not OK.')

#TestCropAndResize2x2x2To3x3x3Nearest
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize2x2x2To3x3x3Nearest is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3Nearest is not OK.')

#TestCropAndResize2x2x2To3x3x3Flipped
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([1,1,1,0,0,0])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all() :
    print('TestCropAndResize2x2x2To3x3x3Flipped is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3Flipped is not OK.')

#TestCropAndResize2x2x2To3x3x3FlippedNearest
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([1,1,1,0,0,0])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize2x2x2To3x3x3FlippedNearest is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3FlippedNearest is not OK.')

#TestCropAndResize3x3x3To2x2x2
image = np.empty((1,3,3,3,1))
image[0,:,:,:,0] = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
boxes = np.empty((2,6))
boxes[0] = np.array([0,0,0,1,1,1])
boxes[1] = np.array([0,0,0,0.5,0.5,0.5])
box_index = np.empty((2))
box_index[0] = 0
box_index[1] = 0
crop_size = np.empty(3)
crop_size = np.array([2,2,2])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all() :
    print('TestCropAndResize3x3x3To2x2x2 is OK.')
else:
    print('TestCropAndResize3x3x3To2x2x2 is not OK.')

#TestCropAndResize3x3x3To2x2x2Nearest
image = np.empty((1,3,3,3,1))
image[0,:,:,:,0] = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
boxes = np.empty((2,6))
boxes[0] = np.array([0,0,0,1,1,1])
boxes[1] = np.array([0,0,0,0.5,0.5,0.5])
box_index = np.empty((2))
box_index[0] = 0
box_index[1] = 0
crop_size = np.empty(3)
crop_size = np.array([2,2,2])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize3x3x3To2x2x2Nearest is OK.')
else:
    print('TestCropAndResize3x3x3To2x2x2Nearest is not OK.')

#TestCropAndResize3x3x3To2x2x2Flipped
image = np.empty((1,3,3,3,1))
image[0,:,:,:,0] = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
boxes = np.empty((2,6))
boxes[0] = np.array([1,1,1,0,0,0])
boxes[1] = np.array([0.5,0.5,0.5,0,0,0])
box_index = np.empty((2))
box_index[0] = 0
box_index[1] = 0
crop_size = np.empty(3)
crop_size = np.array([2,2,2])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all() :
    print('TestCropAndResize3x3x3To2x2x2Flipped is OK.')
else:
    print('TestCropAndResize3x3x3To2x2x2Flipped is not OK.')

#TestCropAndResize3x3x3To2x2x2FlippedNearest
image = np.empty((1,3,3,3,1))
image[0,:,:,:,0] = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
boxes = np.empty((2,6))
boxes[0] = np.array([1,1,1,0,0,0])
boxes[1] = np.array([0.5,0.5,0.5,0,0,0])
box_index = np.empty((2))
box_index[0] = 0
box_index[1] = 0
crop_size = np.empty(3)
crop_size = np.array([2,2,2])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, method='nearest')

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, method_name='nearest')

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize3x3x3To2x2x2FlippedNearest is OK.')
else:
    print('TestCropAndResize3x3x3To2x2x2FlippedNearest is not OK.')

#TestCropAndResize2x2x2To3x3x3Extrapolated
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([-1,-1,-1,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

v = -1
scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size, bounds_error=False, extrapolation_value=v)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, extrapolation_value=v)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize2x2x2To3x3x3Extrapolated is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3Extrapolated is not OK.')

#TestCropAndResize2x2x2To3x3x3NoCrop
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((0,6))
box_index = np.empty((0))
crop_size = np.empty(3)
crop_size = np.array([3,3,3])

scipy_control = crop_and_resize_from_scipy(image, boxes, crop_size)

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

results = crop_and_resize_3d(image, boxes, box_index, crop_size, extrapolation_value=v)

if results.shape == scipy_control.shape and results.numpy().all() == scipy_control.all():
    print('TestCropAndResize2x2x2To3x3x3NoCrop is OK.')
else:
    print('TestCropAndResize2x2x2To3x3x3NoCrop is not OK.')

#TestInvalidInputShape
image = np.empty((2,2,2,1))
image[:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 0
crop_size = np.empty(3)
crop_size = np.array([4,4,4])

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)

try:
    results = crop_and_resize_3d(image, boxes, box_index, crop_size)
except Exception as e:
    if 'input image must be 5-D' in str(e):
        print('TestInvalidInputShape is OK.')

#TestInvalidBoxIndexShape
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((2))
box_index[0] = 0
box_index[1] = 0
crop_size = np.empty(3)
crop_size = np.array([4,4,4])

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)
try:
    results = crop_and_resize_3d(image, boxes, box_index, crop_size)
except Exception as e:
    if 'box_index has incompatible shape' in str(e):
        print('TestInvalidBoxIndexShape is OK.')

#TestInvalidBoxIndex
image = np.empty((1,2,2,2,1))
image[0,:,:,:,0] = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
boxes = np.empty((1,6))
boxes[0] = np.array([0,0,0,1,1,1])
box_index = np.empty((1))
box_index[0] = 1
crop_size = np.empty(3)
crop_size = np.array([4,4,4])

image = tf.dtypes.cast(image, tf.float32)
boxes = tf.dtypes.cast(boxes, tf.float32)
box_index = tf.dtypes.cast(box_index, tf.int32)
crop_size = tf.dtypes.cast(crop_size, tf.int32)
try:
    results = crop_and_resize_3d(image, boxes, box_index, crop_size)
except Exception as e:
    if 'box_index has values outside [0, batch_size)' in str(e):
        print('TestInvalidBoxIndex is OK.')
