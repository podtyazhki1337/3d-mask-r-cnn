<h1 align="center"> 3D Mask R-CNN </h1>
Modified version of the Mask R-CNN for 3D image data (volumes). Now it is supports different image depths, gives telemetry during training and evaluation of RPN, and has a more flexible configuration system.
Also added support for chunked data loading and training for head training and evaluation.
Also contains preprocessing script for the datasets used for my trainings.

Commands for using the Docker image on the datasets (Rats-neurons and Hela Kyoto cells):
    
    docker run -it --rm --gpus device=0 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task RPN_TRAINING --config_path configs/rpn/scp_rpn_rats.json --summary
    docker run -it --rm --gpus device=1 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task RPN_TRAINING --config_path configs/rpn/scp_rpn_hela.json --summary

    docker run -it --rm --gpus device=0 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task TARGET_GENERATION --config_path configs/targeting/scp_target_rat.json --summary
    docker run -it --rm --gpus device=1 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task TARGET_GENERATION --config_path configs/targeting/scp_target_hela.json --summary


    docker run -it --rm --gpus device=0 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task HEAD_TRAINING --config_path configs/heads/scp_heads_rats.json --summary
    docker run -it --rm --gpus device=1 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task HEAD_TRAINING --config_path configs/heads/scp_heads_hela.json --summary

    docker run -it --rm --gpus device=0 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task MRCNN_EVALUATION --config_path configs/mrcnn/scp_mrcnn_rats.json --summary
    docker run -it --rm --gpus device=1 -v "$PWD":/workspace -v /NAS/mmaiurov/Datasets:/NAS/mmaiurov/Datasets -w /workspace gdavid57/3d-mask-r-cnn python -W "ignore::UserWarning" -m main --task MRCNN_EVALUATION --config_path configs/mrcnn/scp_mrcnn_hela.json --summary


Based on the [2D implementation](https://github.com/matterport/Mask_RCNN) by Matterport, Inc, [this update](https://github.com/ahmedfgad/Mask-RCNN-TF2) and [this fork](https://github.com/matterport/Mask_RCNN/pull/1611/files).

This 3D implementation was written by Gabriel David (LIRMM, Montpellier, France). Most of the code inherits from the MIT Licence edicted by Matterport, Inc (see core/LICENCE).

This repository is linked to the paper:

**G. David and E. Faure, End-to-end 3D instance segmentation of synthetic data and embryo microscopy images with a 3D Mask R-CNN, Front. Bioinform., 27 January 2025, Volume 4 - 2024 | [DOI link](https://doi.org/10.3389/fbinf.2024.1497539)**

