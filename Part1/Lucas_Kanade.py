# Part 1.B : Lucas-Kanade Optical Flow
# Link 1 : https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# Link 2 : https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

################### Libraries ###################
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Harris import Harris

################### Class : Lucas_Kanade ###################
class Lucas_Kanade:
    def __init__(self, Threshold):
        self.Threshold = Threshold

    def Lucas_Kanade_Tracker(self, Image1, Image2):
        LK_Params = dict(
            winSize = (19, 19),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # USE Harris for finding points and casting them to Array
        Harris_Detector = Harris(Threshold= 0.05)

        Image1_Harris_Points = Harris_Detector.Harris_Points(Image= Image1)
        Image1_Array_Points  = Harris_Detector.To_Array(Image1_Harris_Points)

        Image2_Harris_Points = Harris_Detector.Harris_Points(Image= Image2)
        Image2_Array_Points  = Harris_Detector.To_Array(Image2_Harris_Points)

        # Image 2 Lucas-Kanade Points
        Image2_Point_LK, Err, St = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY),
            Image1_Array_Points,
            None,
            **LK_Params
        )

        # Image 1 Lucas-Kanade Points (recalc)
        Image1_Point_LK_Recalc, Err, St = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY),
            Image2_Point_LK,
            None,
            **LK_Params
        )

        # Calc maximum Distance between real point and recalculated Points
        Distance = abs(Image1_Array_Points - Image1_Point_LK_Recalc).reshape(-1, 2).max(-1)

        Good_Points_Flag = Distance < self.Threshold

        # Keep Good Points
        Image1_Good_Points = Image1_Array_Points[Good_Points_Flag == 1]
        Image2_Good_Points = Image2_Point_LK[Good_Points_Flag == 1]

        return Image1_Good_Points, Image2_Good_Points

    def Test(self, Image1_path, Image2_path,Output_path):
        # read images
        # pass images to Lucas_Kanade_Tracker
        Image1 = cv2.imread(Image1_path)
        Image2 = cv2.imread(Image2_path)

        Image1_Good_Points, Image2_Good_Points = self.Lucas_Kanade_Tracker(Image1, Image2)

        # Visualization
        # create random color
        color = np.random.randint(0, 255, (len(Image2_Good_Points), 3))

        mask = np.zeros_like(Image1)
        frame = Image2.copy()

        for i,(i2, i1) in enumerate(zip(Image2_Good_Points, Image1_Good_Points)):
            # x, y of end point
            a, b = i2.ravel()
            a = int(a)
            b = int(b)
            # x, y of start point
            c, d = i1.ravel()
            c = int(c)
            d = int(d)
            # draw a line from end to start point with some random color
            mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)

            # draw a circle around end point of line
            #frame = cv2.circle(frame, (a,b), 1, color[i].tolist(), -1)
        
        # add mask to frame
        visualization_figure = cv2.add(frame, mask)

        plt.figure(figsize=(16,9))
        plt.imshow(visualization_figure)
        plt.imsave(Output_path, visualization_figure, dpi = 300)
        plt.show()

if __name__ == '__main__':
    Image1_path = sys.argv[1]
    Image2_path = sys.argv[2]
    output_path = sys.argv[3]

    # create object
    LK = Lucas_Kanade(Threshold= 1.0)

    # Test performance
    LK.Test(Image1_path, Image2_path, output_path)

