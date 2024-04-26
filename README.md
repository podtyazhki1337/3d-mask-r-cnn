# 3D Mask R-CNN

Based on the [2D implementation](https://github.com/matterport/Mask_RCNN) by Matterport, Inc, [this update](https://github.com/ahmedfgad/Mask-RCNN-TF2) and [this fork](https://github.com/matterport/Mask_RCNN/pull/1611/files).

This 3D implementation was written by Gabriel David (LIRMM, Montpellier, France). Most of the code inherits from the MIT Licence edicted by Matterport, Inc (see core/LICENCE). The custom operation codes is under Apache TensorFlow licence (see core/custom_op/LICENCE).

This repository is linked to the paper:


The main branch contains the code and procedure to redo the benchmark over the so-called toy dataset and should be taken as an example in case of a new project. The branch called morphogenesis concerns the application of the 3D Mask R-CNN to the instance segmentation of Phallusia mammillatta embryo cells, and is in consequence more specific to this task.

## Preliminary steps

### Compile TensorFlow sources

The source compilation of TensorFlow 2.1.4 can be achieved following [this tutorial](https://medium.com/@gabriel.david57/from-scratch-to-cuda-installation-and-tensorflow-compilation-by-the-sources-on-ubuntu-20-04-ac3e9a3d6e89).

### Compile 3D Non Max Suppression and 3D Crop And Resize

Before any training or inference, user must compile the 3D Non Max Suppression and Crop And Resize methods within TensorFlow sources as custom operations. After having cloned this repo, do:

```
cd 3d-mask-r-cnn
cp core/custom_op/CropAndResize3D path/to/tensorflow/source/dir/tensorflow/core/user_ops/
cp core/custom_op/NonMaxSuppression3D path/to/tensorflow/source/dir/tensorflow/core/user_ops/

cd path/to/tensorflow/source/dir/tensorflow/core/user_ops/NonMaxSuppression3D
./compilation_script

cd ../CropAndResize3D
./compilation_script
```

The two compilation steps generate four *.so* files

```
path/to/tensorflow/source/dir/tensorflow/core/user_ops/NonMaxSuppression3D/non_max_suppression_3d_op.so
path/to/tensorflow/source/dir/tensorflow/core/user_ops/CropAndResize3D/crop_and_resize_3d_op.so
path/to/tensorflow/source/dir/tensorflow/core/user_ops/CropAndResize3D/crop_and_resize_3d_grad_boxes_op.so
path/to/tensorflow/source/dir/tensorflow/core/user_ops/CropAndResize3D/crop_and_resize_3d_grad_image_op.so
```

Copy each *.so* in the *3d-mask-r-cnn/core/custom_op* directory. We provide test script for these custom operations, *3dcar_test.py* and *3dnms_test.py*, respectively for Crop And Resize and Non Max Suppression. The performed tests are a direct translation of those that can be found in TensorFlow sources for the 2D case. We compare our 3D implemementation of the Crop And Resize op with a method based on the *scipy.interpolate.RegularGridInterpolator* function.

Note: two tests of the Crop And Resize appear as "not Ok" but actually are. The difference of results between our 3D Crop And Resize and the *scipy.interpolate.RegularGridInterpolator* simply highlights that the choices made by these two methods of "what is nearest?" is not the same in this very particular case.


# Toy Dataset

This section aims to reproduce the results of the paper mentioned above on the toy dataset.

Representation of pair ground truth instance segmentation and input image:

![Instance segmentation](example/segmentation.gif)![Image](example/input_image.gif)

## Generate toy data and datasets

You can execute the script *generate_data.py* with command line arguments to customize the parameters for your experiment. Here is how you can do it:

`
python generate_data.py --train_dir "./data/" --nb_thread 1 --n_train_images 10000 --image_size 128
`

The parameters are explained as follows:

+ --train_dir: This argument specifies the directory where the data for training will be stored. The default value is './data/'.
+ --nb_thread: This argument specifies the number of threads you want to run for your experiment. The default value is 1.
+ --n_train_images: This argument specifies the number of training images you want to generate for your experiment. The default value is 10000.
+ --image_size: This argument specifies the size of the images to be generated for your experiment. The default value is 128.

Default generation will produce ~18Go of data.

The train and test datasets are generated thanks to the script *generate_datasets.py* following the command line syntaxe:

`
python generate_datasets.py --data_dir "./data/" --test_size 0.05
`

The parameters are:

+ --data_dir: This argument specifies the directory where the data for training are stored. The default value is './data/'.
+ --test_size: This argument specifies the ratio of training examples assigned to the test subset. The default value is 0.05.

## Region Proposal Network training

In order to run the RPN training, one must follow the example in *rpn_training.py* script and the training config in *configs/rpn/scp_rpn_config.json*.

The script is run following the command:

`
python rpn_training.py --config_path "configs/rpn/scp_rpn_config.json" --gpus "0" --summary True
`

The parameters being:

+ --config: RPN training config. See *scp_rpn_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display RPN keras model summary, number of examples in train and test datasets, as well as the training config.

By default, training weights are stored in *weights/scp_rpn/*.

## Head target generation

As we train the Mask R-CNN in two steps (RPN first and Head after), we advice to generate the proper targets of the Head in order to speed its training. These targets correspond to the output of the RPN including the RoIAlign step. It can be easily achieved by running the *generate_head_targets.py* script as follows:

`
python generate_head_targets.py --config_path "configs/targeting/scp_target_config.json" --gpus "0" --summary True
`

The parameters being:

+ --config: RPN targeting config. See *scp_target_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display RPN keras model summary, number of examples in train and test datasets, as well as the given config.

By default, targets are generated from the 20th epoch of the RPN training and are saved under *data/scp_target/* location.

## Head training

The script *head_training.py* allows to train the Head part of the Mask R-CNN thanks to:

`
python head_training.py --config_path "configs/heads/scp_heads_config.json" --gpus "0" --summary True
`

with

+ --config: Head training config. See *scp_heads_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display Head keras model summary, number of examples in train and test datasets, as well as the training config.

## Mask R-CNN evaluation

Once the RPN and the Head have been trained, one can evaluate the performance of the whole Mask R-CNN with the script *mrcnn_evaluation.py* and the command line:

`
python mrcnn_evaluate.py --config_path "configs/mrcnn/scp_mrcnn_config.json" --gpus "0" --summary True
`

where

+ --config: whole Mask R-CNN config. See *scp_mrcnn_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display the Mask R-CNN keras model summary, number of examples in train and test datasets, as well as the given config.

## Appendix 1: Compare prediction and ground truth instance segmentation



## Appendix 2: Mask R-CNN training

The whole Mask R-CNN is trainable as an end-to-end network with the Python script:

`
python mrcnn_training.py --config_path "configs/mrcnn/scp_mrcnn_config.json" --gpus "0" --summary True
`

where

+ --config: whole Mask R-CNN config. See *scp_mrcnn_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display the Mask R-CNN keras model summary, number of examples in train and test datasets, as well as the training config.

## Appendix 3: Mask R-CNN prediction

The whole Mask R-CNN is trainable as an end-to-end network with the Python script:

`
python mrcnn_training.py --config_path "configs/mrcnn/scp_mrcnn_config.json" --gpus "0" --summary True
`

where

+ --config: whole Mask R-CNN config. See *scp_mrcnn_config.json* or *core/config.py* for more details.
+ --gpus: GPUs to use for training. "0" or "1" to use only one gpu, "0,1" for two gpus for instance.
+ --summary: if True, display the Mask R-CNN keras model summary, number of examples in train and test datasets, as well as the training config.
