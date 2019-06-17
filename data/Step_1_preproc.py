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
import numpy
from numpy import *
from numpy import linalg
from numpy.random import RandomState


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

show_gray = False
show_depth = False
show_user = False

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
    for file_count, file in enumerate(sort(samples)):
        if (set_label == "training"):
            condition = (file_count < 650)
            dest =r"/home/tim/Documents/RISE19/data/Data_processed/Train"


        else:
            condition = (file_count >= 650)
            dest=r"/home/tim/Documents/RISE19/data/Data_processed/valid"
                

        #set == "training" ? (condition = (file_count<650)) : (condition = (file_count>=650))
        if condition:  #wudi only used first 650 for validation !!! Lio be careful!
            print(("\t Processing file " + file))
            start_time = time.time()
            # Create the object to access the sample
            sample = GestureSample(os.path.join(data, file))
            # ###############################################
            # USE Ground Truth information to learn the model
            # ###############################################
            # Get the list of actions for this frame
            gestures = sample.getGestures()
            # Iterate for each action in this sample
            for gesture in gestures:
                skelet, depth, gray, user, c = sample.get_data_wudi(gesture, vid_res, NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue

                # preprocess
                # skelet_feature: frames * num_features? here gestures because we need netural frames
                skelet_feature, Targets, c = proc_skelet_wudi(sample, used_joints, gesture, STATE_NO,
                                                              NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue
                user_o = user.copy()
                user = proc_user(user)
                skelet, c = proc_skelet(skelet)
                # depth: 2(h&b) * frames * 5 (stacked frames) * vid_shape_hand[0] *vid_shape_hand[1]
                user_new, depth, c = proc_depth_wudi(depth, user, user_o, skelet, NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue
                # gray:  2(h&b) * frames * 5 (stacked frames) * vid_shape_hand[0] *vid_shape_hand[1]
                gray, c = proc_gray_wudi(gray, user, skelet, NEUTRUAL_SEG_LENGTH)
                if c: print('corrupt'); continue

                if show_depth: play_vid_wudi(depth, Targets, wait=1000 / 10, norm=False)
                if show_gray: play_vid_wudi(gray, Targets, wait=1000 / 10, norm=False)
                if show_user: play_vid_wudi(user_new, Targets, wait=1000 / 10, norm=False)
                # user_new = user_new.astype("bool")
                traj2D, traj3D, ori, pheight, hand, center = skelet
                skelet = traj3D, ori, pheight

                assert user.dtype == gray.dtype == depth.dtype == traj3D.dtype == ori.dtype == "uint8"
                assert gray.shape == depth.shape
                if not gray.shape[1] == skelet_feature.shape[0] == Targets.shape[0]:
                    print("too early movement or too late,skip one");
                    continue

                # we don't need user info. anyway
                video = empty((2,) + gray.shape, dtype="uint8")
                video[0], video[1] = gray, depth
                store_preproc_wudi(video, skelet_feature, Targets.argmax(axis=1), skelet, dest)


            end_time = time.time()


            print("Processing one batch requires: %d second\n"% ( end_time - start_time))         
            if condition and file_count==(len(samples)-1):
                dump_last_data(video,skelet_feature, Targets.argmax(axis=1), skelet, dest)

            # we should add the traning data as well
            if not condition and file_count == 650-1:
                dump_last_data(video,skelet_feature, Targets.argmax(axis=1), skelet, dest)







def store_preproc_wudi(video, skelet, label, skelet_info, dest):
    """
    Wudi modified how to- store the result
    original code is a bit hard to understand
    """
    global v, s, l, sk, count, batch_idx
    if len(v) == 0:
        v = video
        s = skelet
        l = label
        sk = []
        sk.append(skelet_info)
    else:
        v = numpy.concatenate((v, video), axis=2)
        s = numpy.concatenate((s, skelet), axis=0)
        l = numpy.concatenate((l, label))
        sk.append(skelet_info)

    if len(l) > 1000:
        make_sure_path_exists(dest)
        os.chdir(dest)
        file_name = "batch_" + "_" + str(batch_idx) + "_" + str(len(l)) + ".zip"
        if store_result:
            file = gzip.GzipFile(file_name, 'wb')
            dump((v, s, l, sk), file, -1)
            file.close()

        print(file_name)
        batch_idx += 1
        count = 1
        v, s, l, sk = [], [], [], []

    count += 1


def dump_last_data(video, skelet, label, skelet_info, dest):
    global v, s, l, sk, count, batch_idx
    v = numpy.concatenate((v, video), axis=2)
    s = numpy.concatenate((s, skelet), axis=0)
    l = numpy.concatenate((l, label))
    sk.append(skelet_info)
    os.chdir(dest)
    file_name = "batch_" + "_" + str(batch_idx) + "_" + str(len(l)) + ".zip"
    if store_result:
        file = gzip.GzipFile(file_name, 'wb')
        dump((v, s, l, sk), file, -1)
        file.close()

    print(file_name)


if __name__ == '__main__':
    main()
