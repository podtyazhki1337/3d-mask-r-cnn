import os
import numpy as np
import random
from skimage import io
import bz2
import _pickle as cPickle
import threading
from tqdm import tqdm
import argparse

num_classes = 15
num_max_objects = 20

def apply_random_transform(matrix):
    dice = random.randint(0, 7)
    if dice == 0:
        return matrix
    if dice == 1:
        return np.rot90(matrix, k=1)
    if dice == 2:
        return np.rot90(matrix, k=2)
    if dice == 3:
        return np.rot90(matrix, k=3)
    matrix = np.flip(matrix, axis=2)
    if dice == 4:
        return matrix
    if dice == 5:
        return np.rot90(matrix, k=1)
    if dice == 6:
        return np.rot90(matrix, k=2)
    if dice == 7:
        return np.rot90(matrix, k=3)

def create_data(inputs):

    image_shape, out_dir, six = inputs

    img = np.random.rand(*image_shape)*0.1
    seg = np.zeros(image_shape).astype('uint8')
    nbCenter = random.randint(3, num_max_objects)
    # print('processing {} {} with {} cells '.format(out_dir, six, nbCenter))
    f = open(out_dir+'classes_and_boxes/'+str(six+1).zfill(6)+'.dat', 'w')
    masks = np.zeros((*image_shape, nbCenter))
    n = 0
    trial = 0
    while n < nbCenter:
        object_choice = random.choice([[getSphere, spheres, 1], [getCube, cubes, 2], [getPyramid, pyramids, 3]])

        getObject = object_choice[0]
        objects = object_choice[1]

        cid = random.randint(0, num_classes - 1)
        r = 10 + cid

        x = random.randint(r, img.shape[0] - r - 1)
        y = random.randint(r, img.shape[1] - r - 1)
        z = random.randint(r, img.shape[2] - r - 1)

        if r not in objects:
            objects[r] = getObject(r)

        obj = objects[r]
        if object_choice[2] == 3:
            obj = apply_random_transform(obj)
        
        coords = np.array(np.where(obj))
        coords[0] += x - r
        coords[1] += y - r
        coords[2] += z - r

        isOther = np.unique(seg[coords[0], coords[1], coords[2]])
        if len(isOther) == 1 and isOther[0] == 0:  # Evertything is OK
            seg[coords[0], coords[1], coords[2]] = n+1
            img[coords[0], coords[1], coords[2]] += 0.15
            masks[coords[0], coords[1], coords[2], n] = np.uint8(1)
            horizontal_indicies = np.where(np.any(np.any(masks[..., n], axis=0), axis=1))[0]
            vertical_indicies = np.where(np.any(np.any(masks[..., n], axis=1), axis=1))[0]
            profound_indicies = np.where(np.any(np.any(masks[..., n], axis=0), axis=0))[0]
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = profound_indicies[[0, -1]]
            n += 1
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(object_choice[2],y1, x1, z1, y2+1, x2+1, z2+1))
        else:
            trial += 1

        if trial > 100:
            masks = masks[..., :n]
            break
        
    f.close()
    with bz2.BZ2File(out_dir + 'masks/' + str(six+1).zfill(6) + '.pickle', 'w') as f:
        cPickle.dump(masks, f)
    io.imsave(out_dir +'seg/'+str(six+1).zfill(6)+'.tiff', seg, check_contrast=False)
    img = img - np.min(img)
    maxi = np.mean(img) + 5 * np.std(img)
    img = (np.where(img > maxi, maxi, img)) * 255 / maxi
    img8bit = img.astype(np.uint8)
    io.imsave(out_dir+ 'images/' +str(six+1).zfill(6)+'.tiff', img8bit, check_contrast=False)

spheres = {}
def getSphere(rayon):
    # print("Create Sphere "+str(rayon))
    r2 = rayon ** 2
    diametre = rayon * 2
    sphere = np.zeros((diametre, diametre, diametre)).astype('int32')
    for z in range(diametre):
        for y in range(diametre):
            for x in range(diametre):
                r = ((x - rayon) ** 2 + (z - rayon) ** 2 + (y - rayon) ** 2)
                if (r - r2) < 0:
                    sphere[y][x][z] = rayon - np.int32(np.sqrt(r))
    return sphere


cubes = {}
def getCube(rayon):
    # print("Create Sphere "+str(rayon))
    side = 2*rayon
    cube = rayon*np.ones((side, side, side)).astype('int32')
    return cube


pyramids = {}
def getPyramid(rayon):
    side = 2*rayon
    pyramid = np.zeros((side, side, side)).astype('int32')
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if j <= i-k:
                    pyramid[i, j ,k] = 1
    return pyramid

def generate_data(info, start, end):
    for i in tqdm(range(start, end)):
        create_data(info[i])


def generate_experiment(nb_train_images, image_size, train_dir, nb_thread):

    os.makedirs(f"{train_dir}", exist_ok=True)
    os.makedirs(f"{train_dir}classes_and_boxes/", exist_ok=True)
    os.makedirs(f"{train_dir}seg/", exist_ok=True)
    os.makedirs(f"{train_dir}masks/", exist_ok=True)
    os.makedirs(f"{train_dir}images", exist_ok=True)

    image_shape = (image_size, image_size, image_size)

    info = []
    info += [[image_shape, train_dir, six] for six in range(nb_train_images)]

    batch_size = int(nb_train_images / nb_thread)
    threads = list()
    print('starting creating {} images'.format(len(info)))
    for i in range(nb_thread):
        x = threading.Thread(target=generate_data, args=(info, i * batch_size, (i +1) * batch_size))
        threads.append(x)
        x.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/')
    parser.add_argument('--nb_thread', type=int, default=1)
    parser.add_argument('--nb_train_images', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()

    generate_experiment(args.nb_train_images, args.image_size, args.train_dir, args.nb_thread)
