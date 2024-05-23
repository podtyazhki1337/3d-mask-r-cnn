import os
import argparse

from core.config import load_config
from core.models import RPN, HEAD, MaskRCNN

# Comment the following line to debug TF or libcuda issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', 
                        type=str,
                        required=True,
                        choices=['RPN_TRAINING', 'RPN_EVALUATION', 'TARGET_GENERATION' 'HEAD_TRAINING', 'MRCNN_EVALUATION'],
                        help='Task to operate: "RPN_TRAINING", "RPN_EVALUATION", "TARGET_GENERATION", "HEAD_TRAINING" ou "MRCNN_EVALUATION"')
    
    parser.add_argument('--config_path', 
                        type=str, 
                        required=True,
                        help="Path to config file.")
    
    parser.add_argument('--gpus', 
                        type=str, 
                        default='0',
                        help="Choice of GPU. Default is '0'.")
    
    parser.add_argument('--summary', 
                        action='store_true')
    
    args = parser.parse_args()

    # Load training config
    toy_config = load_config(args.config_path)

    # Assign gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    
    if args.task == "RPN_TRAINING":

        # Initiate model
        rpn = RPN(toy_config, show_summary=args.summary)
    
        # Training loop (one epoch -> save weights -> quick train monitoring -> one epoch -> ...)
        rpn.train()
    
    elif args.task == "RPN_EVALUATE":

        # Initiate model
        rpn = RPN(toy_config, show_summary=args.summary)

        # Next line allows to check predicted boxes (so called rpn_rois) and ground truth boxes
        # Note: must add RPN_WEIGHTS in scp_rpn_config.json
        rpn.evaluate()
    
    elif args.task == "TARGET_GENERATION":

        # Initiate model
        rpn = RPN(toy_config, show_summary=args.summary)

        # Head target generation
        rpn.head_target_generation()

    elif args.task == "HEAD_TRAINING":

        # Initiate model
        head = HEAD(toy_config, show_summary=args.summary)
        
        # Training loop (one epoch -> save weights -> quick train monitoring -> one epoch -> ...)
        head.train()
    
    elif args.task == "MRCNN_EVALUATE":

        # Initiate model
        mrcnn = MaskRCNN(toy_config, show_summary=args.summary)

        # Inference and evaluation over test dataset
        mrcnn.evaluate()