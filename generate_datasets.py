import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data_dir, test_size):

    cab_dir = f"{data_dir}classes_and_boxes/"
    seg_dir = f"{data_dir}seg/"
    images_dir = f"{data_dir}images/"
    masks_dir = f"{data_dir}masks/"

    list_segs = [f"{seg_dir}{x}" for x in os.listdir(seg_dir)]
    list_images = [x.replace(seg_dir, images_dir) for x in list_segs]
    list_masks = [x.replace(seg_dir, masks_dir).replace(".tiff", ".pickle") for x in list_segs]
    list_cabs = [x.replace(seg_dir, cab_dir).replace(".tiff", ".dat") for x in list_segs]
    list_names = [x.split("/")[-1].split(".")[0] for x in list_segs]

    df = pd.DataFrame(
        {
            "names": list_names,
            "images": list_images,
            "segs": list_segs,
            "cabs": list_cabs,
            "masks": list_masks,
        }
    )

    df_train, df_test = train_test_split(df, test_size=test_size)
    os.makedirs(f"{data_dir}datasets/", exist_ok=True)
    df_train.to_csv(f"{data_dir}datasets/train.csv", index=None)
    df_test.to_csv(f"{data_dir}datasets/test.csv", index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--test_size', type=float, default=0.05)
    args = parser.parse_args()
    split_data(args.data_dir, args.test_size)