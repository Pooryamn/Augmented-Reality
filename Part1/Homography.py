# Part 1.C : Get Homography Matrix
# Link 1 : https://github.com/hughesj919/HomographyEstimation
# Link 2 : https://math.stackexchange.com/questions/3509039/calculate-homography-with-and-without-svd

################### Libraries ###################
import sys
import numpy as np
import random
import cv2
from Lucas_Kanade import Lucas_Kanade

################### Class : Homography ###################
class Homography:
    def __init__(self, Num_Points, Min_num_Points):
        if(Num_Points < Min_num_Points):
            raise Exception("Number of points must be bigger than Minimum number of Points!")

        self.Num_Points = Num_Points
        self.Min_num_Points = Min_num_Points

    def Find_H_Matrix(self, Image1, Image2):

        # Build LK object and calculate Good points
        LK_Tracker = Lucas_Kanade(Threshold= 5)
        Image1_Good_Points, Image2_Good_Points = LK_Tracker.Lucas_Kanade_Tracker(Image1, Image2)

        # Point indices
        Point_indices = random.sample(list(range(len(Image1_Good_Points))), self.Min_num_Points)

        # Matrix : A
        A_matrix = np.zeros((self.Num_Points * 2, 9))
        index = 0

        for i, j in zip(Point_indices, Point_indices):
            A_matrix[index, :] = np.array([
                -Image1_Good_Points[i, 0, 0],
                -Image1_Good_Points[i, 0, 1],
                -1,
                0 ,
                0 ,
                0 ,
                Image2_Good_Points[j, 0, 0] * Image1_Good_Points[i, 0, 0], 
                Image2_Good_Points[j, 0, 0] * Image1_Good_Points[i, 0, 1],
                Image2_Good_Points[j, 0, 0]
            ])

            index += 1

            A_matrix[index, :] = np.array([
                0,
                0,
                0,
                -Image1_Good_Points[i, 0, 0],
                -Image1_Good_Points[i, 0, 1],
                -1,
                Image2_Good_Points[j, 0, 1] * Image1_Good_Points[i, 0, 0],
                Image2_Good_Points[j, 0, 1] * Image1_Good_Points[i, 0, 1],
                Image2_Good_Points[j, 0, 1]
            ])
            
            index += 1
        
        u, s, v = np.linalg.svd(A_matrix)

        H_UnNormal = v[8].reshape(3, 3)

        Homography_Matrix = (1 / H_UnNormal.flatten()[8]) * H_UnNormal

        return Homography_Matrix

    def Test(self, Image1_path, Image2_path):
        # read images
        image1 = cv2.imread(Image1_path)
        image2 = cv2.imread(Image2_path)

        Homography_Matrix = self.Find_H_Matrix(image1, image2)

        print(f"Homography Matrix: \n{Homography_Matrix}")

if __name__ == '__main__':
    Image1_path = sys.argv[1]
    Image2_path = sys.argv[2]

    Homography_finder = Homography(4, 4)
    Homography_finder.Test(Image1_path, Image2_path)