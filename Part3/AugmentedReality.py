# Part 3 : Align Image 1 in a part of Videos

################### Libraries ###################
from copy import deepcopy
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from sklearn.linear_model import ridge_regression

np.set_printoptions(suppress=True)
global Point_coordinates
Point_coordinates = []

def get_points(Image):
    cv2.imshow('frame 1', Image)
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('frame 1', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
    
    # Show selected points
    show_Point_coordinates(Image)
    

def click_event(event, x, y, flags, params):

        if(event == cv2.EVENT_LBUTTONDOWN):
            Point_coordinates.append([x,y])
            print(x, ' ', y)

def show_Point_coordinates(Image):
    Image2 = deepcopy(Image)
    
    for point in Point_coordinates:
        Image2 = cv2.circle(Image2, point, 3, (255,0,0), -1)
    
    plt.figure(figsize=(16,9))
    plt.imshow(Image2)
    plt.show()

def read_frames(video_path,frames):
    cap = cv2.VideoCapture(video_path)

    print('Start reading ...')

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frames.append(frame)

        else:
            break
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('End reading!')

    frames = np.array(frames)
    return frames

def Augmented_reality(video_path, image_path, feature_type):
    global Point_coordinates
    frames = []
    dst_frames = []
    # read video and save it in frames
    frames = read_frames(video_path, frames)

    align_image = cv2.imread(image_path)
    align_image = cv2.cvtColor(align_image, cv2.COLOR_BGR2RGB)

    # show first frame to user to select points
    get_points(frames[0])

    for i in range(len(frames) - 1):

        if(feature_type == 'Sift'):
            Key_Points_prev = Sift_detector(frames[i])

        elif(feature_type == 'Harris'):
            Key_Points_prev = Harris_detector(frames[i], 0.2)

        # cast type to float
        Key_Points_prev = np.float32(Key_Points_prev)

        # pass frames and previous key points to calc points in next frame
        Point_coordinates_next = KLT_Track(frames[i], Point_coordinates, frames[i+1])

        # Alignment
        frame_output = align(frames[i], Point_coordinates, align_image)
        dst_frames.append(frame_output)

        # set points for next frame
        Point_coordinates = Point_coordinates_next.tolist()
    
    dst_frames = np.array(dst_frames)
    write_video(dst_frames)
    


def write_video(dst_frames):

    size = (720, 1280)
    fps = 30

    video = cv2.VideoWriter('./Results/output.avi', cv2.VideoWriter_fourcc(*'MP42'), fps, (size[1], size[0]))

    for frame in dst_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame)
    video.release()

        
def align(Image1, Image1_Points, Image2):

    # Image 1 = Chess (background) - frames[i]
    # Image 2 = building (foreground) - align_image
    # change x,y -> y,x
    for point in Image1_Points:
        point[0], point[1] = point[1], point[0]

    Image2_Points = np.array([
        [Image2.shape[0], 0],
        [0, 0],
        [Image2.shape[0], Image2.shape[1]],
        [0, Image2.shape[1]]
    ], dtype= np.float32)

    Image1_Points = np.roll(Image1_Points, 0, axis = 1)
    Image2_Points = np.roll(Image2_Points, 1, axis = 1)

    Image1_Points = Image1_Points.reshape((-1, 1, 2))
    Image2_Points = Image2_Points.reshape((-1, 1, 2))

    # H_Matrix
    H_Matrix = homography(Image2_Points, Image1_Points)

    image2_transformed = cv2.warpPerspective(Image2, H_Matrix, (Image1.shape[0], Image1.shape[1]))
    
    # set alignment
    # gray scale image for Transpose
    image2_transformed_gray = cv2.cvtColor(image2_transformed, cv2.COLOR_RGB2GRAY)

    # create final image using Image1
    visual_result = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
    # set 0 to parts of visual result which is belong to image2
    visual_result[image2_transformed_gray.T.nonzero()] = 0

    # Transpose of an 3d array 
    image2_transformed = np.transpose(image2_transformed, (1, 0, 2))

    # combine 2 images
    final_result = image2_transformed + visual_result

    # Debug
    #plt.figure()
    #plt.imshow(final_result)
    #plt.show()
    return final_result


def homography(Image1, Image2, sample_no = 4):

    # Point indices
    #Point_indices = random.sample(list(range(Image1.shape[0])), sample_no)
    Point_indices = np.arange(sample_no)

    # Cast types 
    Image1 = np.float32(Image1)

    # Matrix : A
    A_matrix = np.zeros((sample_no * 2, 9))
    index = 0
    

    for i, j in zip(Point_indices, Point_indices):
        A_matrix[index, :] = np.array([
            -Image1[i, 0, 0],
            -Image1[i, 0, 1],
            -1,
            0 ,
            0 ,
            0 ,
            Image2[j, 0, 0] * Image1[i, 0, 0], 
            Image2[j, 0, 0] * Image1[i, 0, 1],
            Image2[j, 0, 0]
        ])

        index += 1

        A_matrix[index, :] = np.array([
            0,
            0,
            0,
            -Image1[i, 0, 0],
            -Image1[i, 0, 1],
            -1,
            Image2[j, 0, 1] * Image1[i, 0, 0],
            Image2[j, 0, 1] * Image1[i, 0, 1],
            Image2[j, 0, 1]
        ])
            
        index += 1
        
    u, s, v = np.linalg.svd(A_matrix)
    H_UnNormal = v[8].reshape(3, 3)
        
    Homography_Matrix = (1 / H_UnNormal.flatten()[8]) * H_UnNormal
    return Homography_Matrix


    
def KLT_Track(prev_image, prev_points, next_image):
    prev_pts = deepcopy(prev_points)
    prev_pts = np.array(prev_pts).astype(np.float32)

    LK_Params = dict(
            winSize = (19, 19),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    next_points, Err, St = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY),
            prev_pts,
            None,
            **LK_Params
        )
    
    # Debug
    #print(prev_points[10])
    #print(next_points[10])
    #plt.figure()
    #plt.imshow(prev_image, cmap = 'gray')
    #plt.figure()
    #plt.imshow(next_image, cmap = 'gray')
    #plt.show()

    return next_points


def Harris_detector(Image, Threshold):

    Gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Harris_image = cv2.cornerHarris(Gray_image, 7, 3, 0.03)

    Harris_Detected_Points = Harris_image > Threshold * Harris_image.max()
    Harris_Detected_Points = (Harris_Detected_Points * 1).nonzero()
    
    # Debug
    #Image2 = deepcopy(Image)
    #for point in range(len(Harris_Detected_Points[0])):
        #Image2 = cv2.circle(img= Image2, center= (Harris_Detected_Points[1][point], Harris_Detected_Points[0][point]), radius= 2, color= [255, 0, 30], thickness= -1)
    #plt.figure(figsize=(20,15))
    #plt.imshow(Image2)
    #plt.show()

    return np.array(Harris_Detected_Points).T

def Sift_detector(Image):
    Gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures= 200)
    SIFT_KP = sift.detect(Gray_image, None)

    Sift_Detected_Points = []
    for KP in SIFT_KP:
        TMP = [int(KP.pt[0]), int(KP.pt[1])]
        Sift_Detected_Points.append(TMP)

    # Debug
    #Image2 = deepcopy(Image)
    #for point in Sift_Detected_Points:
        #Image2 = cv2.circle(img= Image2, center= (point[0], point[1]), radius= 2, color= [255, 0, 30], thickness= -1)
    #plt.figure(figsize=(20,15))
    #plt.imshow(Image2)
    #plt.show()


    return np.array(Sift_Detected_Points)

Augmented_reality('../Res/Videos/input2.avi','../Res/images/building1.jpg', 'Harris')