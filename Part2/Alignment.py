# Part 2 : Align Image 1 in a part of Image 2
# Link 1 : https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
# Link 2 : https://stackoverflow.com/questions/33548150/how-to-delete-drawn-lines-on-image-in-python
# Link 3 : https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/

################### Libraries ###################
from copy import deepcopy
import sys
import cv2
import random
from cv2 import HOGDescriptor
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
################### GLOBAL variables ###################
cl_points = []
################### Class : Alignment ###################
class Alignment:
    def __init__(self):
        pass

    def align(self, Image1, Image1_Points, Image2):

        # Image 1 = Background
        # Image 2 = Foreground
        # Change x,y -> y,x
        for point in Image1_Points:
            point[0], point[1] = point[1], point[0]
        # print(Image1_Points)
        # Image1_Points = np.array([[385, 538],[300, 538],[388, 689],[294, 689]], dtype=np.float32)

        # print(Image1_Points)
        Image2_Points = np.array([
            [Image2.shape[0], 0],
            [0, 0],
            [Image2.shape[0], Image2.shape[1]],
            [0, Image2.shape[1]]
            ], dtype=np.float32)

        Image1_Points = np.roll(Image1_Points, 0, axis = 1)
        Image2_Points = np.roll(Image2_Points, 1, axis = 1)

        Image1_Points = Image1_Points.reshape((-1, 1, 2))
        Image2_Points = Image2_Points.reshape((-1, 1, 2))

        # H martix
        H_Matrix = self.homography(Image2_Points, Image1_Points)
        image2_transformed = cv2.warpPerspective(Image2, H_Matrix, (Image1.shape[0],Image1.shape[1]))

        # Conver Colors
        image2_transformed_gray = cv2.cvtColor(image2_transformed, cv2.COLOR_RGB2GRAY)

        plt.figure(figsize=(10,10))
        visual_result = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)

        visual_result[image2_transformed_gray.T.nonzero()] = 0
        
        # Transpose of an 3d array 
        image2_transformed = np.transpose(image2_transformed, (1, 0, 2))

        
            # combine 2 images
        final_result = image2_transformed + visual_result
        plt.imshow(final_result)
        plt.imsave('alignment.jpg',final_result,dpi=400)
        plt.show()



    def get_point_positions(self, Image1):
        global cl_points   

        # displaying the image
        Image1 = cv2.cvtColor(Image1, cv2.COLOR_RGB2BGR)
        cv2.imshow('image1', Image1)
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image1', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cl_points

    def click_event(self, event, x, y, flags, params):
        global cl_points

        if(event == cv2.EVENT_LBUTTONDOWN):
            cl_points.append([x,y])
            print(x, ' ', y)

    def Draw_points(self, Image1, Points):
        if ((len(Points) != 4) or (len(Points[0]) != 2) or 
            (len(Points[1]) != 2) or (len(Points[2]) != 2) or
            (len(Points[3]) != 2)):

            raise Exception("Point Shape is incorrect")
        
        Image2 = deepcopy(Image1)

        for center in clicked_points:
            Image2 = cv2.circle(Image2, center, 3, (255, 0, 0), -1)
        
        plt.figure(figsize= (16, 9))
        plt.imshow(Image2, cmap= 'gray')
        plt.show()

    def homography(self, Image1, Image2):
        # Point indices
        Point_indices = np.arange(4)

        # Matrix : A
        A_matrix = np.zeros((4 * 2, 9))
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



if __name__ == '__main__':

    Image1_path = sys.argv[1]
    Image2_path = sys.argv[2]

    Image1 = cv2.imread(Image1_path)
    Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
    Image2 = cv2.imread(Image2_path)
    Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)


    Align_obj = Alignment()
    # Allow user to click on image and choose points
    clicked_points = Align_obj.get_point_positions(Image1)
    Align_obj.Draw_points(Image1, clicked_points)
    Align_obj.align(Image1, np.array(clicked_points, dtype=np.float32), Image2)
    
    
    