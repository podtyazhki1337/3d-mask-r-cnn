import os
import argparse

from core.config import load_config
from core.models import MaskRCNN
from core.data_generators import ToyDataset

# Comment the following line to debug TF or libcuda issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/mrcnn/scp_mrcnn_config.json')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--summary', type=bool, default=True)
    args = parser.parse_args()

    # Load training config
    toy_config = load_config(args.config_path)

    # Assign gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Create Datasets
    test_dataset = ToyDataset()
    test_dataset.load_dataset(data_dir=toy_config.DATA_DIR, is_train=False)
    test_dataset.prepare()

    # Initiate model
    weight_dir = f"./weights/{toy_config.NAME}/"
    os.makedirs(weight_dir, exist_ok=True)
    mrcnn = MaskRCNN(toy_config, weight_dir, None, test_dataset)

    # Display model summary
    if args.summary:
        mrcnn.keras_model.summary(line_length=140)
    
    # Training loop (one epoch -> save weights -> one epoch -> ...)
    mrcnn.train()
