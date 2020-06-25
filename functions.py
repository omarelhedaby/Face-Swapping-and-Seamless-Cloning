from commonfunctions import *
from PIL import Image, ImageDraw
import pyamg
import cv2
from skimage.filters import median
import numpy as np
import skimage.io as io
from io import BytesIO
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
import face_recognition
#! /usr/bin/env python
import scipy.sparse
from scipy.sparse.linalg import spsolve
import base64
import cv2
import numpy as np


def blend(img_target, img_source, img_mask,offset,center):
    # compute regions to be blended
    # clip and normalize mask image
    # create coefficient matrix
    if(img_target.shape!=img_source.shape):
        img_source=cv2.resize(img_source,(img_target.shape[1],img_target.shape[0]))
        img_mask=cv2.resize(img_mask,(img_target.shape[1],img_target.shape[0]))
    img_source = np.roll(img_source, offset[1], axis=0)
    img_source = np.roll(img_source, offset[0], axis=1)
    img_mask = np.roll(img_mask, offset[1], axis=0)
    img_mask = np.roll(img_mask, offset[0], axis=1)
    img_mask=rgb2gray(img_mask)
    zeros=np.nonzero(img_mask)
    sizeofzero=zeros[0].shape[0]/img_mask.shape[0] #ratio of white to black features
    erosion=int(sizeofzero*(80/120))
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion+50,erosion+50))
    if(erosion-40<0):
        SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion + 35, erosion + 35))
        SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion+9, erosion+9))
    else:
        SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion-30, erosion - 5))
    #img_mask[0:int(center[0]),::]=binary_erosion(img_mask[0:int(center[0]),::],SE)
    #img_mask[int(center[0]):img_mask.shape[0],::]=binary_erosion( img_mask[int(center[0]):img_mask.shape[0],::],SE2)
    img_mask=binary_erosion(img_mask,SE)
    img_mask=binary_dilation(img_mask,SE2)
    show_images([img_mask])
    #img_mask=binary_dilation(img_mask,SE1)
    #SE=np.ones(())
    A = scipy.sparse.identity(img_mask.shape[0]*img_mask.shape[1], format='lil')
    for j in range(img_mask.shape[0]):
        for i in range(img_mask.shape[1]):
            if img_mask[j][i]!=0 :
                index = i+j*img_mask.shape[1]
                A[index, index] = 4
                if index+1 < (img_mask.shape[0]*img_mask.shape[1]) :
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+img_mask.shape[1] < img_mask.shape[0]*img_mask.shape[1]:
                    A[index, index+img_mask.shape[1]] = -1
                if index-img_mask.shape[1] >= 0:
                    A[index, index-img_mask.shape[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)


    # for each layer (ex. RGB)
    for RGB in range(3):
        # get subimages
        t = img_target[:,:,RGB]
        s = img_source[:,:,RGB]
        t = t.flatten()
        s = s.flatten()
        # create b
        b = P * s
        laplace = np.abs(np.reshape(b, img_mask.shape))
        F2 = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        filter1 = np.abs(convolve2d(in1=img_source[:,:,RGB], in2=F2))
        show_images([laplace,filter1],["poisson","LOG"])

        for y in range(img_mask.shape[0]):
            for x in range(img_mask.shape[1]):
                if img_mask[y][x]==0:
                    index = x+y*img_mask.shape[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)
        # assign x to target image
        x = np.reshape(x, img_mask.shape)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[:,:,RGB] = x
    return img_target
def applymask(face2,hullPointsList2,morphedface):

    mask = np.zeros((face2.shape[0], face2.shape[1]), dtype=face2.dtype)
    cv2.fillConvexPoly(mask, np.array(hullPointsList2), color=255)
    #show_images([mask])
    #mask=io.imread("test1_mask.png")
    show_images([mask])
    output=blend(face2,morphedface,mask,(0,-4),getCenter(hullPointsList2))
    #output = cv2.seamlessClone(morphedface, face2, mask, getCenter(hullPointsList2), cv2.NORMAL_CLONE)
    #output=process(morphedface, face2, mask)
    return output
def getCenter(hullPoints):
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([hullPoints]))
    center = ((x1 + int(w1 / 2), y1 + int(h1 / 2)))
    #bounding_rectangle = cv2.rectangle(face2.copy(), (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    return center

def triListforMorph(trilist):
    list = []
    for tri in trilist:
        point1 = (tri[0], tri[1])
        point2 = (tri[2], tri[3])
        point3 = (tri[4], tri[5])
        list.append([point1, point2, point3])
    list = np.array(list).astype(int)
    return list
#def morphTriangle(img1,img, t1, t) :


def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst
def morphing(image1,tri1,image2,tri2):
    x1,y1,w1,h1 = cv2.boundingRect(np.float32([tri1]))
    x2,y2,w2,h2 = cv2.boundingRect(np.float32([tri2]))
    for i in range(3):
        tri1[i][0]=tri1[i][0]-x1
        tri1[i][1]=tri1[i][1]-y1
        tri2[i][0] = tri2[i][0] - x2
        tri2[i][1] = tri2[i][1] - y2
    wrapimage=np.copy(image1[y1:y1+h1,x1:x1+w1])
    mask=np.zeros((h2,w2))
    cv2.fillConvexPoly(mask,tri2,color=(255))
    wrap=applyAffineTransform(wrapimage,tri1,tri2,[w2,h2])
    for i in range(image2[y2:y2 + h2, x2:x2 + w2].shape[0]):
        for j in range(image2[y2:y2 + h2, x2:x2 + w2].shape[1]):
            if mask[i][j]==255:
                image2[y2:y2+h2,x2:x2+w2][i][j]=wrap[i][j]
def convexHull(face,points,points2,draw=False):
    pil_image = Image.fromarray(face)
    d = ImageDraw.Draw(pil_image)
    # Print the location of each facial feature in this image
    # Let's trace out each facial feature in the image with a line!
    points = np.array(points)
    hullIndex = cv2.convexHull(points, returnPoints=False)
    hullPoints1 = points[hullIndex]
    hullPoints2=points2[hullIndex]
    hullPointsList1 = []
    hullPointsList2=[]
    for idx,i in enumerate(hullPoints1):
        if (i[0][0] < face.shape[1] and i[0][1] < face.shape[0]):
            hullPointsList1.append((i[0][0], i[0][1]))
            hullPointsList2.append((hullPoints2[idx][0][0],hullPoints2[idx][0][1]))
    d.line(hullPointsList1, width=5)
    if(draw==True):
       pil_image.show()
    return  hullPointsList1,hullPointsList2,hullIndex
def drawDelaunayTriangulation(image,trilist,draw):
    if draw==True:
        for tri in trilist:
            point1 = (tri[0], tri[1])
            point2 = (tri[2], tri[3])
            point3 = (tri[4], tri[5])
            cv2.line(image, point1, point2, color=(255, 0, 0))
            cv2.line(image, point2, point3, color=(255, 0, 0))
            cv2.line(image, point3, point1, color=(255, 0, 0))
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def drawOther(image,trilist,draw):
    if draw==True:
        for tri in trilist:
            point1 = (tri[0])
            point2 = (tri[1])
            point3 = (tri[2])
            cv2.line(image, tuple(point1), tuple(point2), color=(255, 0, 0))
            cv2.line(image, tuple(point2), tuple(point3), color=(255, 0, 0))
            cv2.line(image, tuple(point3), tuple(point1), color=(255, 0, 0))
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def getDelaunayTriangulation(shape,hullpoints):
    subdiv = cv2.Subdiv2D((0, 0, shape[1], shape[0]))
    for point in hullpoints:
        subdiv.insert((point[0],point[1]))
    triList=subdiv.getTriangleList()
    return triList
def getOtherDelaunayTriangulation(delaunay,hullpoints,hullindex,points2):
    triList2=[]
    for tri in delaunay:
        point1 = (tri[0], tri[1])
        point1_other=points2[int(hullindex[hullpoints.index(point1)])]
        point2 = (tri[2], tri[3])
        point2_other = points2[int(hullindex[hullpoints.index(point2)])]
        point3 = (tri[4], tri[5])
        point3_other = points2[int(hullindex[hullpoints.index(point3)])]
        triList2.append([point1_other, point2_other, point3_other])
    triList2 = np.array(triList2).astype(int)
    return triList2

def drawPoints(points,image,draw):
    if draw==True:
        for i in points:
            cv2.circle(image, (i[0], i[1]), 1, (0, 0, 255), -1)
            # d.line(face_landmarks[facial_feature], width=5)
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def facedetection(image):
    face_locations = face_recognition.face_locations(image)
    # img=io.imread("obama.jpg")
    top, right, bottom, left = face_locations[0]
    img = image[top:bottom, left:right]
    face_landmarks_list = face_recognition.face_landmarks(img)
    #chin = face_landmarks_list[0]["chin"]
    #chin = np.array(chin)
   # min_chin = chin[chin[:, 0] == min(chin[:, 0])].astype(int)
    #max_chin = chin[chin[:, 0] == max(chin[:, 0])].astype(int)
    return img,(top,left,bottom,right)
def stringToRGB(base64_string):
    im = Image.open(BytesIO(base64.b64decode(str(base64_string))))
    im.save('image.png', 'PNG')
    return (io.imread('image.png'))
# Load the jpg file into a numpy array
