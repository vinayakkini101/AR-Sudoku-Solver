import cv2
from imutils import contours
import numpy as np
# from VideoStreamWidget import VideoStreamWidget


stream_link='http://192.168.0.101:8080/video'
cap = cv2.VideoCapture(stream_link)

while(True):
    ret, frame = cap.read()
    originalFrame = cv2.resize(frame, (600, 325), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('original frame',originalFrame)

    grayFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayscale',grayFrame)

    blurredFrame =  cv2.GaussianBlur(grayFrame,(5,5),cv2.BORDER_DEFAULT)
    # cv2.imshow('blurred frame',blurredFrame)

    thresFrame = cv2.adaptiveThreshold(blurredFrame, 255, cv2.ADAPTIVE_THRESH_MEAN_C ,\
            cv2.THRESH_BINARY, 5, 2)
    # cv2.imshow('adaptive gaussian thresholding',thresFrame)

    inverted = cv2.bitwise_not(thresFrame)
    cv2.imshow('inverted', inverted)

    # kernel = np.ones((1,1),np.uint8)
    # openedFrame = cv2.morphologyEx(thresFrame, cv2.MORPH_CLOSE, kernel)
    # openedFrame = cv2.morphologyEx(thresFrame, cv2.MORPH_OPEN, kernel)
    # openedFrame = cv2.dilate(thresFrame, kernel, iterations=1)
    # cv2.imshow('dl',openedFrame)

    cnts = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    maxAreaContour = [[]]
    maxArea = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > maxArea :
            maxArea = area
            maxAreaContour = c
    
    # cv2.drawContours(originalFrame, [maxAreaContour], -1, (255,0,0), -1)
    # cv2.imshow('contours',originalFrame)
    """
    # lines = cv2.HoughLinesP(inverted, rho=1, theta=np.pi/180, threshold=100,\
    #         minLineLength=1, maxLineGap=0)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv2.line(originalFrame,(x1,y1),(x2,y2),(0,255,0),2)
    # cv2.imshow('probab hough transform',originalFrame)
    """

    # extLeft = tuple(maxAreaContour[maxAreaContour[:, :, 0].argmin()][0])
    # extRight = tuple(maxAreaContour[maxAreaContour[:, :, 0].argmax()][0])
    # extTop = tuple(maxAreaContour[maxAreaContour[:, :, 1].argmin()][0])
    # extBot = tuple(maxAreaContour[maxAreaContour[:, :, 1].argmax()][0])
    # print(extLeft, extRight, extBot, extTop)

    cv2.drawContours(originalFrame, [maxAreaContour], -1, (0, 255, 255), 2)
    # cv2.circle(originalFrame, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(originalFrame, extRight, 8, (0, 255, 0), -1)
    # cv2.circle(originalFrame, extTop, 8, (255, 0, 0), -1)
    # cv2.circle(originalFrame, extBot, 8, (255, 255, 0), -1)
    cv2.imshow('max area contour', originalFrame)

    epsilon = 0.01*cv2.arcLength(maxAreaContour, True)
    approxRect = cv2.approxPolyDP(maxAreaContour, epsilon, True)
    cv2.drawContours(originalFrame, [approxRect], 0, (0), 3)

    # print("area of grid :",cv2.contourArea(approxRect))
    x,y = approxRect[0][0]
    if len(approxRect) != 4:
        cv2.putText(originalFrame, "Readjust camera", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        cv2.imshow('Frame', originalFrame)
        continue
    # cv2.putText(originalFrame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
    # cv2.imshow('polygon', originalFrame)

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
    warpedFrame = cv2.warpPerspective(inverted ,matrix,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    cv2.imshow('warped perspective', warpedFrame)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    warpedFrame = cv2.morphologyEx(warpedFrame, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    warpedFrame = cv2.morphologyEx(warpedFrame, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    grids = cv2.findContours(warpedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grids = grids[0] if len(grids) == 2 else grids[1]
    (grids, _) = contours.sort_contours(grids, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for (i, c) in enumerate(grids, 1):
        area = cv2.contourArea(c)
        if area < 500000:
            row.append(c)
            if i % 9 == 0:  
                (grids, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(grids)
                row = []

    # Iterate through each box
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(originalFrame.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            result = cv2.bitwise_and(originalFrame, mask)
            result[mask==0] = 255
            cv2.imshow('result', result)
            cv2.waitKey(175)

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