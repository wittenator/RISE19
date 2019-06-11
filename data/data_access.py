# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:27:08 2019

@author: cmp3tahera
"""

#from classes import GestureSample
from ChalearnLAPSample import GestureSample
gestureSample = GestureSample("Sample0002.zip")
#fps=gestureSample.getFPS()
 #Finally, we can access to an object that encodes the skeleton information in the same way:

skeleton=gestureSample.getSkeleton(10)
'''
SampleXXXX_skeleton.mp4: CSV with the skeleton information for each frame of the viedos. Each line corresponds to one frame. 
Skeletons are encoded as a sequence of joins, providing 9 values per join [Wx, Wy, Wz, Rx, Ry, Rz, Rw, Px, Py]
(W are world coordinats, R rotation values and P the pixel coordinats). The order of the joins in the sequence is:
1.HipCenter, 2.Spine, 3.ShoulderCenter, 4.Head,5.ShoulderLeft, 6.ElbowLeft,7.WristLeft, 8.HandLeft, 9.ShoulderRight, 
10.ElbowRight, 11.WristRight, 12.HandRight, 13.HipLeft, 14.KneeLeft, 15.AnkleLeft, 16.FootLeft, 17.HipRight, 
18.KneeRight, 19.AnkleRight, and 20.FootRight.
'''
'''
To get the skeleton information, we have some provided functionalities. For each join 
the [Wx, Wy, Wz, Rx, Ry, Rz, Rw, Px, Py] description array is stored in a dictionary as three independent vectors. 
You can access each value for each join (eg. the head) as follows:
'''
[Wx, Wy, Wz]=skeleton.getAllData()['Head'][0]

[Rx, Ry, Rz, Rw]=skeleton.getAllData()['Head'][1]

[Px, Py]=skeleton.getAllData()['Head'][2]
'''
Additionally, some visualization functionalities are provided. You can get an image representation of the skeleton or a 
composition of all the information for a frame.

skelImg=gestureSample.getSkeletonImage(10)
frameData=gestureSample.getComposedFrame(10)
'''
#To visualize all the information of a sample, you can use this code:
def showSkeleton(path_to_data):
    
    import cv2
    from ChalearnLAPSample import GestureSample
    
    gestureSample = GestureSample(path_to_data)
    cv2.namedWindow(path_to_data,cv2.WINDOW_NORMAL) 
    for x in range(1, gestureSample.getNumFrames()):
    #    img=gestureSample.getComposedFrame(x)
        img=gestureSample.getSkeletonImage(x)
        cv2.imshow(path_to_data,img)
        cv2.waitKey(1)
    del gestureSample
    cv2.destroyAllWindows()

def showComposed(path_to_data):
    
    import cv2
    from ChalearnLAPSample import GestureSample
    
    gestureSample = GestureSample(path_to_data)
    cv2.namedWindow(path_to_data,cv2.WINDOW_NORMAL) 
    for x in range(1, gestureSample.getNumFrames()):
        img=gestureSample.getComposedFrame(x)
#        img=gestureSample.getSkeletonImage(x)
        cv2.imshow(path_to_data,img)
        cv2.waitKey(1)
    del gestureSample
    cv2.destroyAllWindows()    
    
path_to_data = "Sample0001.zip" 
showComposed(path_to_data)
#showSkeleton(path_to_data)
