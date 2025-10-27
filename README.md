<h1 align="center"> 3D Mask R-CNN </h1>

Modified version of the Mask R-CNN for 3D image data (volumes). Now it supports different image sizes, gives telemetry during training and evaluation of RPN, and has a more flexible configuration system.

Also added support for chunked data loading and training for head training and evaluation.

Also contains preprocessing script for the datasets used for my trainings.

## New: End-to-End HEAD Training Mode

Traditional Mask R-CNN training requires 3 stages:
1. Train RPN
2. Generate targets from RPN proposals
3. Train HEAD on pre-generated targets

New `training_head_e2e` mode that trains HEAD with frozen RPN using live proposals, matching inference distribution exactly.

### Key Benefits:
- **No distribution mismatch** - HEAD sees same proposals during training as in inference
- **No target generation step** - trains directly on raw images
- **Better generalization** - learns to handle actual RPN proposal quality
- **Seamless weight transfer** - unified layer names between training and inference

### Usage

Simply set `"MODE": "training_head_e2e"` in your HEAD config:

```json
{
  "MODE": "training_head_e2e",
  "RPN_WEIGHTS": "weights/rpn/best.h5",
  ...
}
```

For traditional training with pre-generated targets, use `"MODE": "training"`.

## Key Improvements

### 1. Architecture Changes for E2E Compatibility

Modified HEAD and MASKRCNN to use explicit `PyramidROIAlign` layers with consistent names:

```python
# Both training and inference now use:
rois_aligned = PyramidROIAlign([pool_size, pool_size, pool_size], 
                               name="roi_align_classifier")(...)
mask_aligned = PyramidROIAlign([mask_pool_size, mask_pool_size, mask_pool_size],
                               name="roi_align_mask")(...)
```

This ensures weights trained in `training_head_e2e` mode load correctly during inference.

### 2. Improved Loss Functions

**Class Loss (Focal Loss):**
- Universal fg_prob calculation (works for binary and multi-class)
- Per-class accuracy metrics
- Adaptive alpha/gamma balancing

**BBox Loss (Huber Loss):**
- Soft clipping: `3.0 * tanh(pred / 3.0)` instead of hard clip
- More robust to outliers than Smooth L1
- Detailed diagnostics: mean_err, max_err, percentage of large errors

**Mask Loss (Combined BCE + Dice):**
- `0.2 * BCE + 0.8 * Dice` (focused on shape accuracy)
- Center-of-mass error tracking
- Empty mask filtering

### 3. Recommended Hyperparameters

For best results in `training_head_e2e` mode:

```json
{
  "RPN_POSITIVE_IOU": 0.5,        // Higher threshold for quality proposals
  "RPN_NEGATIVE_IOU": 0.3,        // Wider margin
  "ROI_POSITIVE_RATIO": 0.5,      // Balanced sampling
  "TRAIN_ROIS_PER_IMAGE": 128,
  
  "LOSS_WEIGHTS": {
    "mrcnn_class_loss": 1.0,
    "mrcnn_bbox_loss": 0.5,       // Reduced weight for stability
    "mrcnn_mask_loss": 1.5         // Increased for better segmentation
  },
  
  "OPTIMIZER": {
    "name": "ADAM",
    "parameters": {
      "lr": 0.0001,
      "beta_1": 0.9,
      "beta_2": 0.999,
      "clipnorm": 5.0               // Gradient clipping
    }
  }
}
```

## Training Pipeline

### Traditional 2-Stage Pipeline:
```bash
# Stage 1: Train RPN
docker run ... --task RPN_TRAINING --config_path configs/rpn/config.json

# Stage 2: Generate targets
docker run ... --task TARGET_GENERATION --config_path configs/targeting/config.json

# Stage 3: Train HEAD (MODE: "training")
docker run ... --task HEAD_TRAINING --config_path configs/heads/config.json
```

### New End-to-End Pipeline:
```bash
# Stage 1: Train RPN
docker run ... --task RPN_TRAINING --config_path configs/rpn/config.json

# Stage 2: Train HEAD end-to-end (MODE: "training_head_e2e")
docker run ... --task HEAD_TRAINING --config_path configs/heads/config_e2e.json
```

## Docker Commands

### RPN Training
```bash
docker run -it --rm --name rpn_train --gpus device=0 \
  -v "$PWD":/workspace \
  -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets \
  -w /workspace gdavid57/3d-mask-r-cnn \
  python -W "ignore::UserWarning" -m main \
  --task RPN_TRAINING \
  --config_path configs/rpn/scp_rpn_rats.json \
  --summary
```

### Target Generation (Traditional Pipeline Only)
```bash
docker run -it --rm --name target --gpus device=0 \
  -v "$PWD":/workspace \
  -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets \
  -w /workspace gdavid57/3d-mask-r-cnn \
  python -W "ignore::UserWarning" -m main \
  --task TARGET_GENERATION \
  --config_path configs/targeting/scp_target_rat.json \
  --summary
```

### HEAD Training (Both Traditional and E2E)
```bash
docker run -it --rm --name head_train --gpus device=0 \
  -v "$PWD":/workspace \
  -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets \
  -w /workspace gdavid57/3d-mask-r-cnn \
  python -W "ignore::UserWarning" -m main \
  --task HEAD_TRAINING \
  --config_path configs/heads/scp_heads_rats.json \
  --summary
```

### MRCNN Evaluation
```bash
docker run -it --rm --name mrcnn_eval --gpus device=0 \
  -v "$PWD":/workspace \
  -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets \
  -w /workspace gdavid57/3d-mask-r-cnn \
  python -W "ignore::UserWarning" -m main \
  --task MRCNN_EVALUATION \
  --config_path configs/mrcnn/scp_mrcnn_rats.json \
  --summary
```

## Expected Training Metrics

During `training_head_e2e` HEAD training, you should see:

```
CLASS_LOSS:
  pos_acc: 0.90-1.0   (classification accuracy on positives)
  bg_acc:  0.95-0.99  (background classification accuracy)
  fg_prob: 0.70-0.90  (confidence on true class)
  loss:    0.01-0.05

BBOX_LOSS:
  mean_err: 0.3-0.6   (mean coordinate error in normalized coords)
  max_err:  2.0-4.0   (max error)
  pct_large: <0.05    (percentage of large errors >2.0)
  loss:     0.15-0.35

MASK_LOSS:
  dice:     0.80-0.90 (Dice coefficient)
  com_err:  0.05-0.10 (center-of-mass error, normalized)
  loss:     0.12-0.25

TOTAL LOSS: 0.30-0.60
```

## Configuration Examples

### Traditional HEAD Training Config
```json
{
  "MODE": "training",
  "DATA_DIR": "/path/to/dataset/",
  "NUM_CLASSES": 2,
  "RPN_WEIGHTS": "weights/rpn/best.h5",
  "HEAD_WEIGHTS": null,
  ...
}
```

### End-to-End HEAD Training Config
```json
{
  "MODE": "training_head_e2e",
  "DATA_DIR": "/path/to/dataset/",
  "NUM_CLASSES": 2,
  "RPN_WEIGHTS": "weights/rpn/best.h5",
  "HEAD_WEIGHTS": null,
  "FROM_EPOCH": 0,
  "EPOCHS": 100,
  
  "RPN_POSITIVE_IOU": 0.5,
  "RPN_NEGATIVE_IOU": 0.3,
  "ROI_POSITIVE_RATIO": 0.5,
  "TRAIN_ROIS_PER_IMAGE": 128,
  
  "LOSS_WEIGHTS": {
    "mrcnn_class_loss": 1.0,
    "mrcnn_bbox_loss": 0.5,
    "mrcnn_mask_loss": 1.2
  },
  
  "OPTIMIZER": {
    "name": "ADAM",
    "parameters": {
      "lr": 0.0001,
      "beta_1": 0.9,
      "beta_2": 0.999,
      "clipnorm": 5.0
    }
  }
}
```

## Code Changes Summary

### 1. HEAD Class (`core/models.py`)
- Added `_build_e2e_model()` - full end-to-end architecture with frozen RPN
- Added `_train_e2e()` - training loop using `RPNGenerator` in targeting mode
- Added `_freeze_rpn_layers()` - selective layer freezing
- Improved weight loading: auto-loads `best.h5` when `FROM_EPOCH > 0`

### 2. MASKRCNN Class (`core/models.py`)
- Modified `build()` inference mode to use explicit `PyramidROIAlign` layers
- Uses `fpn_classifier_graph()` and `build_fpn_mask_graph()` (without `_with_RoiAlign`)
- Added dynamic reshape support for variable number of ROIs

### 3. Loss Functions (`core/losses.py`)
- `mrcnn_class_loss_graph`: Focal loss with universal fg_prob
- `mrcnn_bbox_loss_graph`: Huber loss with soft clipping
- `mrcnn_mask_loss_graph`: Combined BCE+Dice with COM error tracking

### 4. Graph Functions
- `fpn_classifier_graph()`: Dynamic reshape via Lambda for variable ROIs
- `build_fpn_mask_graph()`: Consistent with training architecture

## Troubleshooting

### Large BBox Errors During Training
**Symptoms:** `mean_err > 1.0`, `max_err > 8.0`, high `pct_large`

**Solution:**
1. Increase `RPN_POSITIVE_IOU` to 0.5 (ensures better quality proposals)
2. Reduce `mrcnn_bbox_loss` weight to 0.5
3. Verify RPN is properly trained before HEAD training

### Weight Loading Fails
**Symptoms:** Weights don't load or layers missing during inference

**Solution:**
1. Verify layer names match between training and inference
2. Check `HEAD_CONV_CHANNEL` and `FPN_CLASSIF_FC_LAYERS_SIZE` consistency
3. Use `by_name=True, skip_mismatch=True` for partial loading

### "None values not supported" Error
**Symptoms:** Error during model build with static reshape

**Solution:**
- Ensure `fpn_classifier_graph()` uses dynamic reshape via `Lambda`
- Check all ROI Align layers have explicit `name` parameter

## Performance Tips

1. **Start with well-trained RPN** - critical for HEAD success
2. **Use `training_head_e2e` mode** - better generalization than pre-generated targets
3. **Monitor Dice score** - best indicator of segmentation quality
4. **Adjust loss weights** based on your metrics (typical: class=1.0, bbox=0.5, mask=1.2)
5. **Use gradient clipping** - prevents exploding gradients (clipnorm=5.0)
6. **Higher IoU threshold** - `RPN_POSITIVE_IOU=0.5` gives more stable training

## Datasets

This implementation has been tested on:
- **Rats Neurons** - 3D microscopy of neuron cells
- **Hela Kyoto Cells** - 3D cellular imaging dataset

Preprocessing scripts are included in the repository.

## Citation

Based on the [2D implementation](https://github.com/matterport/Mask_RCNN) by Matterport, Inc, [this update](https://github.com/ahmedfgad/Mask-RCNN-TF2) and [this fork](https://github.com/matterport/Mask_RCNN/pull/1611/files).

This 3D implementation was written by Gabriel David (LIRMM, Montpellier, France). Most of the code inherits from the MIT Licence edicted by Matterport, Inc (see core/LICENCE).

End-to-end training mode and loss function improvements by Miroslav Maiurov.

**Related paper:**

**G. David and E. Faure, End-to-end 3D instance segmentation of synthetic data and embryo microscopy images with a 3D Mask R-CNN, Front. Bioinform., 27 January 2025, Volume 4 - 2024 | [DOI link](https://doi.org/10.3389/fbinf.2024.1497539)**

