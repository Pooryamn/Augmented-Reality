# Part 1.A : Extract Harris Points 
# Link 1 : https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
# Link 2 : https://www.geeksforgeeks.org/python-corner-detection-with-harris-corner-detection-method-using-opencv/

################### Libraries ###################
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt 

################### Class : Harris ###################
class Harris:
    def __init__(self, Threshold):
        self.Threshold = Threshold
        pass

    def Harris_Points(self, Image):
        # cast pixels to float for processing in Harris
        Gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        Harris_image = cv2.cornerHarris(Gray_image, 7, 3, 0.03)
        # Not all of detected Points are needed. and some of them must delete using Threshold Parameter.
        Harris_Detected_Points = Harris_image > self.Threshold * Harris_image.max()
        # Convert T/F array to an image (non Harris points = 0)
        Harris_Detected_Points = (Harris_Detected_Points * 1).nonzero()

        return Harris_Detected_Points

    def To_Array(self, Harris_Detected_Points):
        # Input  : Tuple of coordinates (list[x], list[y])
        # Output : A 3D array of coordinates (type : np.float32)
        Array_Points = np.zeros((len(Harris_Detected_Points[0]), 1, 2), dtype= np.float32)
        Array_Points[:, :, 0] = np.array(Harris_Detected_Points[1].reshape(-1, 1))
        Array_Points[:, :, 1] = np.array(Harris_Detected_Points[0].reshape(-1, 1))

        return Array_Points

    def To_Tuple(self, Harris_Detected_Points):
        # Input  : A 3D array of coordinates (type : np.float32)
        # Output : Tuple of coordinates (list[x], list[y])
        Tuple_Points_1 = []
        Tuple_Points_2 = []
        for i in Harris_Detected_Points:
            Tuple_Points_1.append(i[0, 0].astype(np.int32))
            Tuple_Points_2.append(i[0, 1].astype(np.int32))
        Tuple_Points = (Tuple_Points_2, Tuple_Points_1)

        return Tuple_Points

    def Test(self, Image_Path, Output_path):
        Image = cv2.imread(Image_Path)
        Harris_Detected_Points = self.Harris_Points(Image= Image)
        # Visualize Output
        # circle around points
        plt.figure(figsize=(16,9))
    
        # show points in Image
        for point in range(len(Harris_Detected_Points[0])):
            Image = cv2.circle(img= Image, center= (Harris_Detected_Points[1][point], Harris_Detected_Points[0][point]), radius= 1, color= [252, 132, 3], thickness= 1)

        plt.imshow(Image)
        plt.imsave(Output_path, Image, dpi = 300)
        plt.show()


        print(f"Number of Detected Points = {len(Harris_Detected_Points[0])}")

        return Harris_Detected_Points
        

if __name__ == '__main__':
    Image_path = sys.argv[1]
    Output_path = sys.argv[2]
    Harris_Detector = Harris(Threshold= 0.1)
    Harris_Detected_Points = Harris_Detector.Test(Image_path, Output_path)
    print(f"type of Point1 : {type(Harris_Detected_Points)}")
    Array_Points = Harris_Detector.To_Array(Harris_Detected_Points)
    print(f"type of Point2 : {Array_Points[0].shape}")
    print(f"type of Point2 : {Array_Points[0]}")
    print(f"type of Point2 : {Array_Points[0][0]}")
    Tuple_Points = Harris_Detector.To_Tuple(Array_Points)
    print(f"type of Point3 : {type(Tuple_Points)}")