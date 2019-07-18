"""
 This is the very first file that you should run to extract training data
 This file extract the skeleton data and the depth and rgb data
 
 author: Di Wu : stevenwudi@gmail.com
 2015/05/15
 """

import cv2
import os
import sys
import shutil
import errno
import gzip
from itertools import tee, islice
from pickle import dump
from glob import glob
from random import shuffle
import numpy as np
from numpy import *
from numpy import linalg
from numpy.random import RandomState
import h5py
from multiprocessing import Pool, cpu_count



#timing
import time

from ChalearnLAPSample_wudi import GestureSample
from functions.preproc_functions import *

#data path and store path definition

data=os.getcwd()

# global variable definition
store_result = True
bg_remove = False
norm_gray = True

vid_res = (480, 640)  # 640 x 480 video resolution
vid_shape_hand = (128, 128)
vid_shape_body = (128, 128)

batch_size = 20  # number of gesture instance
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft', 'HandLeft',
               'ElbowRight', 'WristRight', 'ShoulderRight', 'HandRight',
               'Head', 'Spine', 'HipCenter']
#globals
offset = vid_shape_hand[1] / 2
v, s, l = [], [], []
batch_idx = 0
count = 1

# Then we  choose 8 frame before and after the ground true data:
# in effect it only generate 4 frames because acceleration requires 5 frames
NEUTRUAL_SEG_LENGTH = 4
# number of hidden states for each gesture class
STATE_NO = 6


def main():
    samples = glob("./Train/*.zip")  # because zipped all the files already!
    print(len(samples), "samples found")
    #start preprocessing
    preprocess(samples, "training")

    samples = glob("./Test/*.zip")  # because zipped all the files already!
    preprocess(samples, "validation")


def preprocess(samples, set_label="training"):
    #calculate size of needed arrays
    frame_count = 0
    gesture_count = 0
    for file in sort(samples):
        sample = GestureSample(os.path.join(data, file))
        gestures = sample.getGestures()
        gesture_count += len(gestures)
        # Iterate for each action in this sample
        frame_count += sample.getNumFrames()

    print(gesture_count, frame_count)

    with h5py.File("./dataset.hdf5", "a") as f:
        grp = f.create_group(set_label)
        dst_range = grp.create_dataset("range", (gesture_count,2), compression="szip", dtype="i", chunks=True, shuffle=True, compression_opts=('nn', 20))
        dst_video = grp.create_dataset("video", (frame_count, 480, 640, 3), compression="szip", chunks=True, shuffle=True, compression_opts=('nn', 20))
        dst_skeleton = grp.create_dataset("skeleton", (frame_count, 20, 9), compression="szip", chunks=True, shuffle=True, compression_opts=('nn', 20))
        dst_skeleton_feature = grp.create_dataset("skeleton_feature", (frame_count, 891), compression="szip", chunks=True, shuffle=True, compression_opts=('nn', 20))
        dst_label = grp.create_dataset("label", (frame_count,), compression="szip", chunks=True, shuffle=True,compression_opts=('nn', 20))

        frame_count = 0
        gesture_count = 0
        print(dst_video)
        for file_count, file in enumerate(sort(samples)):
            dest =r"/home/tim/Documents/RISE19/data/Data_processed/Train"
                    
            print(("\t Processing file " + file))
            start_time = time.time()

            # Create the object to access the sample
            sample = GestureSample(os.path.join(data, file))
            gestures = sample.getGestures()


            # Iterate for each action in this sample
            with Pool(cpu_count()) as p:
                results = [x for x in p.starmap(computeData, ((gesture, os.path.join(data, file)) for gesture in gestures)) if x is not None]
            range, skeleton, skelet_feature, video, labels = [np.concatenate(x) for x in zip(*results)]
            print(labels)
            write_to_dataset(dst_range, range+frame_count, gesture_count)
            write_to_dataset(dst_skeleton, skeleton, frame_count)
            write_to_dataset(dst_skeleton_feature, skelet_feature, frame_count)
            write_to_dataset(dst_video, video, frame_count)
            write_to_dataset(dst_label, labels, frame_count)
            gesture_count += range.shape[0]
            frame_count += skeleton.shape[0]
            
            end_time = time.time()





def computeData(gesture, path):
    sample = GestureSample(path)
    skelet, depth, gray, user, c = sample.get_data_wudi(gesture, vid_res, NEUTRUAL_SEG_LENGTH)
    if c: print('corrupt'); return None
    
    # preprocess
    # skelet_feature: frames * num_features? here gestures because we need netural frames
    skelet_feature, Targets, c = proc_skelet_wudi(sample, used_joints, gesture, STATE_NO,
                                                  NEUTRUAL_SEG_LENGTH)
    if c: print('corrupt'); return None
    
    user_o = user.copy()
    user = proc_user(user)
    skelet_proc, c = proc_skelet(skelet)
    user_new, depth, c = proc_depth_wudi(depth, user, user_o, skelet_proc, NEUTRUAL_SEG_LENGTH)
    if c: print('corrupt'); return None
    gray, c = proc_gray_wudi(gray, user, skelet_proc, NEUTRUAL_SEG_LENGTH)
    if c: print('corrupt'); return None

    traj2D, traj3D, ori, pheight, hand, center = skelet_proc
    skelet_proc = traj3D, ori, pheight
    

    assert user.dtype == gray.dtype == depth.dtype == traj3D.dtype == ori.dtype == "uint8"
    assert gray.shape == depth.shape
    if not gray.shape[1] == skelet_feature.shape[0] == Targets.shape[0]:
        print("too early movement or too late,skip one");
        return None

    # we don't need user info. anyway
    video = empty((2,) + gray.shape, dtype="uint8")
    video[0], video[1] = gray, depth

    skeleton = np.array([[np.concatenate(x.getAllData()[joint]) for joint in x.getAllData().keys()] for x in [sample.getSkeleton(i) for i in range(gesture[1], gesture[2])]])
    video = np.array([sample.getRGB(i) for i in range(gesture[1], gesture[2])])

    return np.array([[gesture[1], gesture[2]]]), skeleton, skelet_feature[:skeleton.shape[0]], video, np.array(Targets.argmax(axis=1))[:skeleton.shape[0]]
                

def write_to_dataset(dataset, data, pos, axis=0):
    print(pos, pos + data.shape[axis])
    assert np.count_nonzero(dataset[tuple(slice(None) if not i == axis else slice(pos, pos + data.shape[axis]) for i,dim in enumerate(dataset.shape))]) == 0
    if data.ndim == 1:
        np.insert(dataset, data, pos)
    else:
        dataset[tuple(slice(None) if not i == axis else slice(pos, pos + data.shape[axis]) for i,dim in enumerate(dataset.shape))] = data


if __name__ == '__main__':
    main()
