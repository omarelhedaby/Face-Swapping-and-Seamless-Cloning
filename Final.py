from commonfunctions import *
from PIL import Image, ImageDraw
import cv2
from skimage.filters import median
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
import face_recognition
import imutils
from landmark_detection import *
from functions import *

drawpoint = False
drawtri = False
from PIL import Image
import cv2

face2 = io.imread("images/sanad.jpg")  # host el soora ely ha5odha menko ya shabab
show_images([face2])

face1 = io.imread("images/atwa.jpeg")  # to be swapped

#emotion=hf.getemotion(face2,path1,path2)

emotion = "Happy"
if emotion=="Happy":
    face1 = io.imread("images/atwa.jpeg")  # to be swapped
elif emotion=="Angry":
    face1 = io.imread("images/angry.jpg")  # to be swapped
elif emotion=="Sad":
    face1 = io.imread("images/sad.jpeg")  # to be swapped
elif emotion=="Surprised":
    face1 = io.imread("images/surprised.jpeg")  # to be swapped
elif emotion=="Neutral":
    face1 = io.imread("images/sayed.jpg")
elif emotion=="Other":
    face1 = io.imread("images/atwa.jpeg")  # to be swapped


# Face to swap
points1 = detect_landmarks(face1)[0]  # points of face to swap
points2 = detect_landmarks(face2)
for i in range(len(points2)):
    drawPoints(points1, face1, drawpoint)
    hullPointsList1, hullPointsList2, hullIndex = convexHull(face1, points1, points2[i], False)
    triangleList1 = getDelaunayTriangulation(face1.shape, hullPointsList1)
    triangleList2 = getOtherDelaunayTriangulation(triangleList1, hullPointsList1, hullIndex, points2[i])
    drawDelaunayTriangulation(face1, triangleList1, drawtri)
    drawOther(face2, triangleList2, drawtri)
    facecpy = np.copy(face2)
    morphedface = face2
    triangleList1 = triListforMorph(triangleList1)
    for i in range(len(triangleList1)):
        morphing(face1, triangleList1[i].astype(int), morphedface, triangleList2[i].astype(int))
    # show_images([morphedface])
    output = applymask(facecpy, hullPointsList2, morphedface)
    show_images([output])
    face2 = output
show_images([face2])