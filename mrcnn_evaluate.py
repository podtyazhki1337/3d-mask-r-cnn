import os
import argparse

from core.config import load_config
from core.models import MaskRCNN

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

    # Initiate model
    mrcnn = MaskRCNN(toy_config, show_summary=args.summary)

    # Inference and evaluation over test dataset
    mrcnn.evaluate()
