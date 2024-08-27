import sys
import numpy as np
import cv2
import math

#mask Gray           H, S, V
lower_mask = np.array([82,0,0])
Upper_mask = np.array([180,40,180])

P_infinity=99999999999999

Frame_Rotate_180= False

def nothing(x):
    pass

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def getContours(imgin,img):
    contours,hierarchy =cv2.findContours(imgin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(255,0,0),3)
    for cnt in contours:
        area =cv2.contourArea(cnt)
        if area>1000:
            peri = cv2.arcLength(cnt,True)
            approx =cv2.approxPolyDP(cnt,0.02*peri,True)
            #objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"Target",(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,69,255),2)

def getTableMask(frame):
    if frame is None:
        raise ValueError("Image not loaded correctly")
    # frame = np.squeeze(frame)
    if(Frame_Rotate_180):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # Print the shape of the image
    print("Image shape:", frame.shape)
    # Reorder dimensions if necessary
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # Print the new shape of the image
    print("Reordered image shape:", frame.shape)

    # Verify the number of channels
    if len(frame.shape) < 3 or frame.shape[2] != 3:
        raise ValueError("Invalid number of channels in input image")
    
    dimensions = frame.shape

    #convert BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # cv2.imshow('', hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

    ###Blur
    blur=cv2.GaussianBlur(hsv,(15,15),cv2.BORDER_DEFAULT)
    blur_display= cv2.cvtColor(blur, cv2.COLOR_HSV2BGR)  
    
    ###Filter Color
    mask_frame = cv2.inRange(blur,lower_mask,Upper_mask)
    mask_res_frame = mask_frame
 
    ###Filter Dilate & Eroder
    kernel =np.ones((20,20),np.uint8)
    mask_res_frame = cv2.erode(mask_res_frame,kernel,iterations=1)
    kernel =np.ones((15,15),np.uint8)
    mask_res_frame = cv2.dilate(mask_res_frame,kernel,iterations=1)
    #print("Number of contours detected:",mask_res_frame[12][12])
    ###Filter Edge
    #edge_frame=cv2.Canny(mask_res_frame,30,50)
    
    ###Find Most Corner Point
    most_TL = np.array([0,0])
    most_TR = np.array([0,0])
    most_BL = np.array([0,0])
    most_BR = np.array([0,0])
    
    most_TL_temp_v = np.array([0,0,100000])
    most_TR_temp_v = np.array([0,0,100000])
    most_TL_temp_h = np.array([0,0,100000])
    most_TR_temp_h = np.array([0,0,100000])
    
    most_BL_temp_v = np.array([0,0,100000])
    most_BR_temp_v = np.array([0,0,100000])
    most_BL_temp_h = np.array([0,0,100000])
    most_BR_temp_h = np.array([0,0,100000])
    
    #              m,c
    L1 = np.float32([0,0])
    L2 = np.float32([0,0])
    L3 = np.float32([0,0])
    L4 = np.float32([0,0])
    
    # Find the biggest contour by area
    contours,hierarchy =cv2.findContours(mask_res_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the original image
    
    cv2.drawContours(mask_res_frame, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('', mask_res_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    biggest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask_res_frame, biggest_contour, -1, (0, 255, 0), 2)
    # cv2.imshow('biggest', mask_res_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    peri = cv2.arcLength(biggest_contour,True)
    approx =cv2.approxPolyDP(biggest_contour,0.02*peri,True)
    #objCor = len(approx)
    x,y,w,h = cv2.boundingRect(approx)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #cv2.putText(frame,"Target",(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,69,255),2)    
    # Get the bounding box coordinates of the biggest contour
    x_bc, y_bc, w_bc, h_bc = cv2.boundingRect(biggest_contour) 
    
    for y in range(y_bc, y_bc + h_bc): # 
            for x in range(x_bc, x_bc + w_bc):
                if mask_res_frame[y, x] > 0:
                    r_TL=math.sqrt( 10*math.pow(x, 2)+math.pow(y, 2))
                    r_TR=math.sqrt( 10*math.pow(x-dimensions[1], 2)+math.pow(y, 2))
                    r_TL1=math.sqrt( math.pow(x, 2)+10*math.pow(y, 2))
                    r_TR1=math.sqrt( math.pow(x-dimensions[1], 2)+10*math.pow(y, 2))
                    
                    r_BL=math.sqrt( 10*math.pow(x, 2)+math.pow(y-dimensions[0], 2))
                    r_BR=math.sqrt( 10*math.pow(x-dimensions[1], 2)+math.pow(y-dimensions[0], 2))
                    r_BL1=math.sqrt( math.pow(x, 2)+10*math.pow(y-dimensions[0], 2))
                    r_BR1=math.sqrt( math.pow(x-dimensions[1], 2)+10*math.pow(y-dimensions[0], 2))
                    
                    if r_TL<=most_TL_temp_v[2]:
                        most_TL_temp_v=([x, y, r_TL])
                    elif r_TR<=most_TR_temp_v[2]:
                        most_TR_temp_v=([x, y, r_TR])
                    if r_TL1<=most_TL_temp_h[2]:
                        most_TL_temp_h=([x, y, r_TL1])
                    elif r_TR1<=most_TR_temp_h[2]:
                        most_TR_temp_h=([x, y, r_TR1])    
                        
                    elif r_BL<=most_BL_temp_v[2] :
                        most_BL_temp_v=([x, y, r_BL])
                    elif r_BR<=most_BR_temp_v[2] :
                        most_BR_temp_v=([x, y,r_BR])
                    elif r_BL1<=most_BL_temp_h[2] :
                        most_BL_temp_h=([x, y, r_BL1])
                    elif r_BR1<=most_BR_temp_h[2] :
                        most_BR_temp_h=([x, y,r_BR1])
                            
    
    if (most_TL_temp_v[0]-most_BL_temp_v[0])!=0:                        
        L1[0] = (most_TL_temp_v[1]-most_BL_temp_v[1])/(most_TL_temp_v[0]-most_BL_temp_v[0])
    else:
        L1[0] = P_infinity
    L1[1] = most_TL_temp_v[1]-(L1[0]*most_TL_temp_v[0])
    
    if (most_BL_temp_h[0]-most_BR_temp_h[0])!=0:                        
        L2[0] = (most_BL_temp_h[1]-most_BR_temp_h[1])/(most_BL_temp_h[0]-most_BR_temp_h[0])
    else:
        L2[0] = P_infinity
    L2[1] = most_BL_temp_h[1]-(L2[0]*most_BL_temp_h[0])
        
    if (most_TR_temp_v[0]-most_BR_temp_v[0])!=0:                        
        L3[0] = (most_TR_temp_v[1]-most_BR_temp_v[1])/(most_TR_temp_v[0]-most_BR_temp_v[0])
    else:
        L3[0] = P_infinity
    L3[1] = most_TR_temp_v[1]-(L3[0]*most_TR_temp_v[0])
    
    if (most_TL_temp_h[0]-most_TR_temp_h[0])!=0:                        
        L4[0] = (most_TL_temp_h[1]-most_TR_temp_h[1])/(most_TL_temp_h[0]-most_TR_temp_h[0])
    else:
        L4[0] = P_infinity
    L4[1] = most_TL_temp_h[1]-(L4[0]*most_TL_temp_h[0])
    
    print(L4,L2)
    print("p",most_TL_temp_h)
    #Line equation
    temp_x=  (L1[1]-L4[1])/(L4[0]-L1[0])
    temp_x= int(temp_x)  
    temp_y = L1[0]*temp_x+L1[1]
    temp_y = int(temp_y)               
    most_TL = np.array([temp_x,temp_y,0])
    
    temp_x= (L3[1]-L4[1])/(L4[0]-L3[0]) 
    temp_x= int(temp_x)  
    temp_y = L4[0]*temp_x+L4[1]
    temp_y = int(temp_y) 
    most_TR = np.array([temp_x,temp_y,0])
    
    temp_x=  (L1[1]-L2[1])/(L2[0]-L1[0])
    temp_x= int(temp_x)  
    temp_y = L2[0]*temp_x+L2[1]
    temp_y = int(temp_y)               
    most_BL = np.array([temp_x,temp_y,0])
    
    temp_x= (L3[1]-L2[1])/(L2[0]-L3[0]) 
    temp_x= int(temp_x)  
    temp_y = L2[0]*temp_x+L2[1]
    temp_y = int(temp_y) 
    most_BR = np.array([temp_x,temp_y,0])
    
    # pts = np.float32(([most_TL[0],most_TL[1]],[most_TR[0],most_TR[1]],[most_BL[0],most_BL[1]],[most_BR[0],most_BR[1]]))
    # print(pts)
    # warp_frame = four_point_transform(frame, pts)
    # cv2.imshow('', warp_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result = cv2.bitwise_and(frame, frame, mask=mask_res_frame)
    result = cv2.circle(result, (most_TL[0],most_TL[1]), radius=3, color=(180, 0, 255), thickness=3)
    result = cv2.circle(result, (most_TR[0],most_TR[1]), radius=3, color=(180, 0, 255), thickness=3)
    result = cv2.circle(result, (most_BL[0],most_BL[1]), radius=3, color=(180, 255, 255), thickness=3)
    result = cv2.circle(result, (most_BR[0],most_BR[1]), radius=3, color=(180, 0, 255), thickness=3)

    # Define the coordinates of the polygon (a rectangle in this case)
    points = np.array([ [most_TR[0],most_TR[1]],[most_TL[0],most_TL[1]], [most_BL[0],most_BL[1]], [most_BR[0],most_BR[1]]], np.int32)
    cv2.imshow('',result)
    cv2.imwrite("mask_corners.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Reshape the points to the shape required by cv2.polylines
    points_reshaped = points.reshape((-1, 1, 2))
    result = np.squeeze(result)
    result = np.transpose(result, (-1,0,1))
    return  points


