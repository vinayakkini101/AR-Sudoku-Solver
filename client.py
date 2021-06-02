import cv2
from imutils import contours
import numpy as np
# from VideoStreamWidget import VideoStreamWidget


stream_link='http://192.168.0.103:8080/video'
cap = cv2.VideoCapture(stream_link)

# def nothing(x):
#         pass

# cv2.namedWindow('slider')
# cv2.createTrackbar('blocksize', 'slider', 3, 19, nothing)
# cv2.createTrackbar('C', 'slider', 0, 20, nothing)

# trackbarBlocksize = cv2.getTrackbarPos('blocksize', 'slider')
# trackbarC = cv2.getTrackbarPos('C', 'slider')


while(True):
    ret, frame = cap.read()
    originalFrame = cv2.resize(frame, (600, 325), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('original frame',originalFrame)

    grayFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayscale',grayFrame)

    blurredFrame =  cv2.GaussianBlur(grayFrame,(5,5),cv2.BORDER_DEFAULT)
    # cv2.imshow('blurred frame',blurredFrame)

    # trackbarBlocksize = cv2.getTrackbarPos('blocksize', 'slider')
    # if trackbarBlocksize % 2 == 0 :
    #     trackbarBlocksize += 1
    # trackbarC = cv2.getTrackbarPos('C', 'slider')

    thresFrame = cv2.adaptiveThreshold(blurredFrame, 255, cv2.ADAPTIVE_THRESH_MEAN_C ,\
            cv2.THRESH_BINARY, 5, 2)
    cv2.imshow('adaptive thresholding',thresFrame)

    inverted = cv2.bitwise_not(thresFrame)
    cv2.imshow('inverted', inverted)

    # kernel = np.ones((0,1),np.uint8)
    # openedFrame = cv2.morphologyEx(thresFrame, cv2.MORPH_DILATE, kernel)
    # openedFrame = cv2.morphologyEx(thresFrame, cv2.MORPH_ERODE, kernel)
    # openedFrame = cv2.dilate(thresFrame, kernel, iterations=1)
    # cv2.imshow('dl',openedFrame)
    
    cnts = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    maxAreaContour = [[]]
    maxArea = 0
    totalGridArea = 0
    for c in cnts:
        totalGridArea = cv2.contourArea(c)
        if totalGridArea > 100 :
            epsilon = 0.01*cv2.arcLength(c, True)
            approxRect = cv2.approxPolyDP(c, epsilon, True)
            if totalGridArea > maxArea :
        # and len(approxRect) == 4:
                maxArea = totalGridArea
                maxAreaContour = c

    cv2.drawContours(originalFrame, [maxAreaContour], -1, (0, 255, 255), 2)
    cv2.imshow('max area contour', originalFrame)

    epsilon = 0.01*cv2.arcLength(maxAreaContour, True)
    approxRect = cv2.approxPolyDP(maxAreaContour, epsilon, True)
    cv2.drawContours(originalFrame, [approxRect], 0, (0), 3)

    x,y = approxRect[0][0]
    if len(approxRect) == 4:
    #     cv2.putText(originalFrame, "Readjust camera", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
    #     cv2.imshow('Frame', originalFrame)
    #     continue
        cv2.putText(originalFrame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        cv2.imshow('polygon', originalFrame)

        pt_A = approxRect[0][0]
        pt_B = approxRect[1][0]
        pt_C = approxRect[2][0]
        pt_D = approxRect[3][0]

        # print(pt_A[0])
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))
    
        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
        
        matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
        warpedFrame = cv2.warpPerspective(thresFrame ,matrix,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
        cv2.imshow('warped perspective', warpedFrame)

        


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


    # def order_points(pts):
    #     # Order along X axis
    #     Xorder = pts[np.argsort(pts[:, 0]), :]

    #     left = Xorder[:2, :]
    #     right = Xorder[2:, :]

    #     # Order along Y axis
    #     left = left[np.argsort(left[:, 1]), :]
    #     (tl, bl) = left

    #     # use distance to get bottom right
    #     D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
    #     (br, tr) = right[np.argsort(D)[::-1], :]

    #     return np.array([tl, tr, br, bl]) 