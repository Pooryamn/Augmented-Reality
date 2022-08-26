# TEST PART 1 visualy
import sys
import cv2
import matplotlib.pyplot as plt
from Ransac import Ransac
from Harris import Harris
from Lucas_Kanade import Lucas_Kanade

if __name__ == '__main__':
    Image1_path = sys.argv[1]
    Image2_path = sys.argv[2]
    image1 = cv2.imread(Image1_path)
    image2 = cv2.imread(Image2_path)

    # Test Harris
    # HR = Harris(0.05)
    # HR.Test(Image1_path, 'outs/building1_harris.jpg')
    # HR.Test(Image2_path, 'outs/building2_harris.jpg')

    # Test LK
    # LK = Lucas_Kanade(2.0)
    # LK.Test(Image1_path, Image2_path, 'outs/LK.jpg')

    # Test Ransac and homography
    Ransac_obj = Ransac(4, 50, 50.0)
    Best_H_Matrix, Max_inlier = Ransac_obj.Ransac_func(image1, image2)


    height, width = image2.shape[:2]

    transformed_image = cv2.warpPerspective(image1, Best_H_Matrix, (width,height))
    
    # plt.figure(figsize=(16, 9))
    # plt.imshow(transformed_image)
    # plt.show()

    final_result = cv2.addWeighted(image2, 0.5, transformed_image, 0.5, 0)
    plt.figure(figsize=(16, 9))
    plt.imshow(final_result)
    plt.show()
    plt.imsave('outs/Homography_result.jpg', final_result)
