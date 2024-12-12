import os
import numpy as np
import random
from skimage import io
import bz2
import _pickle as cPickle
import threading
from tqdm import tqdm
import argparse
from skimage.measure import regionprops
from scipy.ndimage import rotate
import pandas as pd


base = 15
range_random = 2.0
num_max_objects = 20

def apply_noise(img):

    image_poisson = np.random.poisson(img * 10) / 10.0

    gaussian_noise = np.random.normal(0, 0.05, img.shape)

    image_noisy = image_poisson + gaussian_noise

    background_noise = np.random.uniform(0, 0.01, img.shape)
    image_noisy += background_noise

    return image_noisy


def apply_random_rotation(obj):

    padded_obj = np.pad(obj, pad_width=1, mode='constant', constant_values=0)

    angle_x = random.uniform(0, 360)
    angle_y = random.uniform(0, 360)
    angle_z = random.uniform(0, 360)
    
    rotated_obj = rotate(padded_obj, angle_x, axes=(1, 2), reshape=True, mode='nearest')
    rotated_obj = rotate(rotated_obj, angle_y, axes=(0, 2), reshape=True, mode='nearest')
    rotated_obj = rotate(rotated_obj, angle_z, axes=(0, 1), reshape=True, mode='nearest')
    
    return rotated_obj



def get_aspect_ratios(delta_y, delta_x, delta_z):

    ryx = np.minimum(delta_y, delta_x) / np.maximum(delta_y, delta_x)
    ryz = np.minimum(delta_y, delta_z) / np.maximum(delta_y, delta_z)
    rxz = np.minimum(delta_x, delta_z) / np.maximum(delta_x, delta_z)

    return ryx, ryz, rxz

def create_data(inputs):

    image_shape, out_dir, six = inputs

    name = str(six+1).zfill(6)

    img_df = pd.DataFrame(
        {
            "image": [],
            "label": [],
            "class": [],
            "noise": [],
            "y1": [],
            "x1": [],
            "z1": [],
            "y2": [],
            "x2": [],
            "z2": [],
            "ryx": [],
            "ryz": [],
            "rxz": [],
        }
    )
    img = np.zeros(image_shape)
    seg = np.zeros(image_shape).astype('uint8')
    nbCenter = random.randint(3, num_max_objects)
    f = open(out_dir+'classes_and_boxes/'+name+'.dat', 'w')
    masks = np.zeros((*image_shape, nbCenter))
    n = 0
    trial = 0
    while n < nbCenter:
        object_choice = random.choice([[getEllipsoid, 1], [getCuboid, 2], [getPyramid, 3]])

        getObject = object_choice[0]

        obj = getObject(base, range_random)

        delta_y, delta_x, delta_z = obj.shape

        delta_y = int(0.5 * delta_y)
        delta_x = int(0.5 * delta_x)
        delta_z = int(0.5 * delta_z)

        x = random.randint(delta_x, img.shape[0] - delta_x - 1)
        y = random.randint(delta_y, img.shape[1] - delta_y - 1)
        z = random.randint(delta_z, img.shape[2] - delta_z - 1)
        
        coords = np.array(np.where(obj))

        coords[0] += y - delta_y
        coords[1] += x - delta_x
        coords[2] += z - delta_z

        isOther = np.unique(seg[coords[0], coords[1], coords[2]])
        if len(isOther) == 1 and isOther[0] == 0:  # Evertything is OK
            seg[coords[0], coords[1], coords[2]] = n+1
            object_noise = random.uniform(0.02, 0.10)
            img[coords[0], coords[1], coords[2]] += object_noise
            masks[coords[0], coords[1], coords[2], n] = np.uint8(1)
            horizontal_indicies = np.where(np.any(np.any(masks[..., n], axis=0), axis=1))[0]
            vertical_indicies = np.where(np.any(np.any(masks[..., n], axis=1), axis=1))[0]
            profound_indicies = np.where(np.any(np.any(masks[..., n], axis=0), axis=0))[0]
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = profound_indicies[[0, -1]]
            n += 1
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(object_choice[1], y1, x1, z1, y2+1, x2+1, z2+1))
            ryx, ryz, rxz = get_aspect_ratios(delta_y, delta_x, delta_z)
            img_df.loc[len(img_df.index)] = [name, n+1, object_choice[1], object_noise, y1, x1, z1, y2+1, x2+1, z2+1, ryx, ryz, rxz]
        else:
            trial += 1

        if trial > 100:
            masks = masks[..., :n]
            break
        
    f.close()

    with bz2.BZ2File(out_dir + 'masks/' + name + '.pickle', 'w') as f:
        cPickle.dump(masks, f)
    
    io.imsave(out_dir +'seg/'+name+'.tiff', seg, check_contrast=False)

    img = apply_noise(img)
    img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    img8bit = img.astype(np.uint8)
    io.imsave(out_dir+ 'images/' +name+'.tiff', img8bit, check_contrast=False)

    img_df.to_csv(out_dir+ 'csvs/' +name+'.csv')

def getEllipsoid(base, random_range):
    rx = int(base * random.uniform(1 / random_range, random_range))
    ry = int(base * random.uniform(1 / random_range, random_range))
    rz = int(base * random.uniform(1 / random_range, random_range))
    max_dim = 2 * max(rx, ry, rz)
    ellipsoid = np.zeros((max_dim, max_dim, max_dim)).astype(np.uint8)
    center = max_dim // 2
    for z in range(max_dim):
        for y in range(max_dim):
            for x in range(max_dim):
                if ((x - center) / rx) ** 2 + ((y - center) / ry) ** 2 + ((z - center) / rz) ** 2 <= 1:
                    ellipsoid[y, x, z] = 1
    ellipsoid = apply_random_rotation(ellipsoid)
    bbox = regionprops(ellipsoid)[0].bbox
    y1, x1, z1, y2, x2, z2 = bbox
    ellipsoid = ellipsoid[y1:y2, x1:x2, z1:z2]
    return ellipsoid

def getCuboid(base, random_range):
    lx = 2*int(base * random.uniform(1 / random_range, random_range))
    ly = 2*int(base * random.uniform(1 / random_range, random_range))
    lz = 2*int(base * random.uniform(1 / random_range, random_range))
    cuboid = np.ones((int(lx), int(ly), int(lz))).astype(np.uint8)
    cuboid = apply_random_rotation(cuboid)
    bbox = regionprops(cuboid)[0].bbox
    y1, x1, z1, y2, x2, z2 = bbox
    cuboid = cuboid[y1:y2, x1:x2, z1:z2]
    return cuboid

def getPyramid(base, random_range):
    lx = 2*int(base * random.uniform(1 / random_range, random_range))
    ly = 2*int(base * random.uniform(1 / random_range, random_range))
    lz = 2*int(base * random.uniform(1 / random_range, random_range))

    pyramid = np.zeros((ly, lx, lz)).astype(np.uint8)

    for z in range(lz):
        x_size = int((1 - z / lz) * lx)
        y_size = int((1 - z / lz) * ly)
        for y in range(y_size):
            for x in range(x_size):
                pyramid[y, x, z] = 1
    pyramid = apply_random_rotation(pyramid)
    bbox = regionprops(pyramid)[0].bbox
    y1, x1, z1, y2, x2, z2 = bbox
    pyramid = pyramid[y1:y2, x1:x2, z1:z2]
    return pyramid

def generate_data(info, start, end):
    for i in tqdm(range(start, end)):
        create_data(info[i])


def generate_experiment(train_image_nb, image_size, train_dir, thread_nb):

    os.makedirs(f"{train_dir}", exist_ok=True)
    os.makedirs(f"{train_dir}classes_and_boxes/", exist_ok=True)
    os.makedirs(f"{train_dir}seg/", exist_ok=True)
    os.makedirs(f"{train_dir}masks/", exist_ok=True)
    os.makedirs(f"{train_dir}images/", exist_ok=True)
    os.makedirs(f"{train_dir}csvs/", exist_ok=True)

    image_shape = (image_size, image_size, image_size)

    info = []
    info += [[image_shape, train_dir, six] for six in range(train_image_nb)]

    batch_size = int(train_image_nb / thread_nb)
    threads = list()
    print('starting creating {} images'.format(len(info)))
    for i in range(thread_nb):
        x = threading.Thread(target=generate_data, args=(info, i * batch_size, (i +1) * batch_size))
        threads.append(x)
        x.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/')
    parser.add_argument('--thread_nb', type=int, default=1)
    parser.add_argument('--train_image_nb', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()

    generate_experiment(args.train_image_nb, args.image_size, args.train_dir, args.thread_nb)
