# Part 1.D : Remove outliers with RANSAC
# link 1 : https://github.com/anubhavparas/ransac-implementation
# link 2 : https://omyllymaki.medium.com/algorithms-from-scratch-ransac-f5a03bed2fde

################### Libraries ###################
import sys
import numpy as np
import cv2
from Homography import Homography
from Lucas_Kanade import Lucas_Kanade
################### Class : Ransac ###################
class Ransac:
    def __init__(self, Min_num_Points, N_iter, Threshold):
        self.Min_num_Points = Min_num_Points
        self.N_iter = N_iter
        self.Threshold = Threshold

    def __error(self, match, H_matrix):
        p1 = match['p1']
        p2 = match['p2']

        # create a homogeneous points
        point1_homo = np.array([p1[0], p1[1], 1])
        point2_homo = np.array([p2[0], p2[1], 1])

        # prediction of point2 with homography matrix and point 1
        point2_predicted = np.dot(H_matrix, point1_homo.T)

        # cast homogeneous point2 to cartesian (devide by last element)
        point2_predicted = point2_predicted / point2_predicted[-1]

        # calculate error
        error = np.sqrt(np.sum((point2_homo - point2_predicted) ** 2))
        return error


    def Ransac_func(self, Image1, Image2):
        
        # Parameters
        Max_inlier = 0
        Best_H_Matrix = np.zeros((3,3))

        # Homography Object
        Homography_finder = Homography(self.Min_num_Points, self.Min_num_Points)

        # Build LK object and calculate Good points
        LK_Tracker = Lucas_Kanade(Threshold= 1.0)
        Image1_Good_Points, Image2_Good_Points = LK_Tracker.Lucas_Kanade_Tracker(Image1, Image2)

        for i in range(self.N_iter):
            print(f"Iteration : {i}")
            # Calculate H_Matrix with different Point indices (in "Homography_finder" function)
            H_matrix = Homography_finder.Find_H_Matrix(Image1, Image2)

            inlier = 0

            for j in range(len(Image1_Good_Points)):
                # create a match of points
                match = {
                    'p1' : Image1_Good_Points[j, 0],
                    'p2' : Image2_Good_Points[j, 0]
                }

                # calculate error of paired points
                err = self.__error(match, H_matrix)

                # error less than Threshold is inlier
                if (err <= self.Threshold):
                    inlier += 1
            
            if(inlier > Max_inlier):
                Max_inlier = inlier
                Best_H_Matrix = H_matrix

        return Best_H_Matrix, Max_inlier 

    def Test(self, Image1_path, Image2_path):
        # read images
        image1 = cv2.imread(Image1_path)
        image2 = cv2.imread(Image2_path)   

        Best_H_Matrix, Max_inlier = self.Ransac_func(image1, image2) 

        # print results with normal notationsnot scientific
        np.set_printoptions(suppress=True)
        print(f"Best Homography Matrix: \n{Best_H_Matrix}\nNumber of inliers: {Max_inlier}")  

        # Compare scratch function performance with cv2 implemented results
        LK_Tracker = Lucas_Kanade(Threshold= 5.0)
        Image1_Good_Points, Image2_Good_Points = LK_Tracker.Lucas_Kanade_Tracker(image1, image2)

        H, status = cv2.findHomography(Image1_Good_Points, Image2_Good_Points, cv2.RANSAC, self.Threshold)

        print('CV2 Library')
        print(f"Homography Matrix:\n{H}")


if __name__ == '__main__':
    Image1_path = sys.argv[1]
    Image2_path = sys.argv[2]

    ransac = Ransac(4, 50, 10.0)
    ransac.Test(Image1_path, Image2_path)  

