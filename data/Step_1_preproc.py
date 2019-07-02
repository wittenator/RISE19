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


#timing
import time

from ChalearnLAPSample_wudi import GestureSample
from functions.preproc_functions import *

#data path and store path definition

data="/home/tim/Documents/RISE19/data/Train"

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
NEUTRUAL_SEG_LENGTH = 8
# number of hidden states for each gesture class
STATE_NO = 5


def main():
    os.chdir(data)
    samples = glob("*.zip")  # because zipped all the files already!
    
    #samples.sort()
    print(len(samples), "samples found")
    #start preprocessing
    preprocess(samples, "training")
#    preprocess(samples, "Validation")


def preprocess(samples, set_label="training"):
    #calculate size of needed arrays
    frame_count = 0
    gesture_count = 0
    for file in sort(samples):
        sample = GestureSample(os.path.join(data, file))
        gestures = sample.getGestures()
        gesture_count += len(gestures)
        # Iterate for each action in this sample
        for gesture in gestures:
            frame_count += len(gesture)

    with h5py.File("./dataset.hdf5", "a") as f:
        grp = f.create_group(set_label)
        dst_range = grp.create_dataset("range", (gesture_count,2), compression="lzf", dtype="i")
        dst_video = grp.create_dataset("video", (2, 2, frame_count, 5, 128, 128), compression="lzf")
        dst_skeleton = grp.create_dataset("skeleton", (frame_count, 11, 9), compression="lzf")
        dst_skeleton_feature = grp.create_dataset("skeleton_feature", (frame_count, 891), compression="lzf")
        dst_label = grp.create_dataset("label", (gesture_count,), compression="lzf")

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
            for gesture in gestures:
                skelet, depth, gray, user, c = sample.get_data_wudi(gesture, vid_res, NEUTRUAL_SEG_LENGTH)
                
                
                skeleton = np.array([[np.concatenate(x.getAllData()[joint]) for joint in used_joints] for x in skelet])
                if c: print('corrupt'); continue
                
                # preprocess
                # skelet_feature: frames * num_features? here gestures because we need netural frames
                skelet_feature, Targets, c = proc_skelet_wudi(sample, used_joints, gesture, STATE_NO,
                                                              NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue
                
                user_o = user.copy()
                user = proc_user(user)
                skelet_proc, c = proc_skelet(skelet)
                user_new, depth, c = proc_depth_wudi(depth, user, user_o, skelet_proc, NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue
                gray, c = proc_gray_wudi(gray, user, skelet_proc, NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue

                traj2D, traj3D, ori, pheight, hand, center = skelet_proc
                skelet_proc = traj3D, ori, pheight
                

                assert user.dtype == gray.dtype == depth.dtype == traj3D.dtype == ori.dtype == "uint8"
                assert gray.shape == depth.shape
                if not gray.shape[1] == skelet_feature.shape[0] == Targets.shape[0]:
                    print("too early movement or too late,skip one");
                    continue

                # we don't need user info. anyway
                video = empty((2,) + gray.shape, dtype="uint8")
                video[0], video[1] = gray, depth
                write_to_dataset(dst_range, np.array([[dst_range.shape[0], dst_range.shape[0]+1]]), gesture_count)
                write_to_dataset(dst_skeleton, skeleton, frame_count)
                write_to_dataset(dst_skeleton_feature, skelet_feature, frame_count)
                write_to_dataset(dst_video, video, frame_count, axis=2)
                write_to_dataset(dst_label, np.array(Targets.argmax(axis=1)), gesture_count)
                print(dst_video)
                gesture_count += 1
                frame_count += len(gesture)
                
            end_time = time.time()
            print(dst_skeleton_feature)







def write_to_dataset(dataset, data, pos, axis=0):
    dataset[tuple(slice(None) if not i == axis else slice(data.shape[axis]) for i,dim in enumerate(dataset.shape))] = data


if __name__ == '__main__':
    main()
