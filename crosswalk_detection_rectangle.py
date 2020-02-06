import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob, math
import imutils


left=[0,600] #left bottom most point of trapezium
right=[1280,600] #right bottom most point of trapezium
apex_left=[300,450] # left top most point of trapezium
apex_right=[980,450] # right top most point of trapezium
    

src=np.float32([left,apex_left,apex_right,right]) # Source Points for Image Warp
dst= np.float32([[200 ,720], [200  ,0], [980 ,0], [980 ,720]]) # Destination Points for Image Warp

def ROI(originalImage):
    return cv2.polylines(originalImage,np.int32(np.array([[left,apex_left,apex_right,right]])),True,(0,0,255),10)

def WarpPerspective(image):
    y=image.shape[0]
    x=image.shape[1]
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (x,y), flags=cv2.INTER_LINEAR)

# will return the contour of black/white threshold of the bird view image 
def process_img(img_path="./DSC_0155.JPG"):
    orig = cv2.imread(img_path)
    # resize
    orig = cv2.resize(orig, (1280, 720))
    # convert the resized image to grayscale
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # convert to bird view 
    warped = WarpPerspective(img)
    # add gaussian blur 
    blurred = cv2.GaussianBlur(warped, (5, 5), 0)
    # threshold black and white 
    highThresh, thresh_im = cv2.threshold(warped, 133, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*highThresh
    # erode the image     
    kernel = np.ones((2,2),np.uint8)
    eroded_img = cv2.erode(thresh_im, kernel,iterations = 1)
    
    cv2.imshow("Image", eroded_img)
    cv2.waitKey(0)

    # find contours 
    cnts = cv2.findContours(thresh_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return orig, cnts

def is_crosswalk(w, h, box): # w, h of the rect, plus the box
    # check if a minAreaRect box is a cross walk 
    # area shouldn't be too small
    if w*h < 4800 or w*h > 30000: # 40 * 120 || 100 * 300
        return False
    # shouldn't be too thin otherwise is a roadline 
    if h/w > 4:
        return False
    else:
        return True

def process_cnts(cnts):
    img = np.zeros((720,1280,3), np.uint8)
    # filter out the contours that are more likely a cross walk 
    filtered_cnts = []
    crosswalk_boxes = []

    for c in cnts:
        # if contours points too few, drop 
        if c.shape[0] < 50:
            continue
        else: 
            ((cx, cy), (w, h), theta) = rect = cv2.minAreaRect(c) #((cx, cy), (w, h), theta)
            # convert to box2d 
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            print(box)
            print(rect)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            #t = math.sqrt(w**2 + h**2)/2
            ## TODO: find upper left using min(x+y) function 
            #x1, x2, y1, y2 = cx - math.sin(theta)*t, cx + math.sin(theta)*t, cy - math.cos(theta)*t, cy + math.cos(theta)*t

            #cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
            if is_crosswalk(w, h, box):
                filtered_cnts.append(c)
                crosswalk_boxes.append(box)
                
            #cv2.drawContours(img, c, -1, (0, 255, 0), 3)
    cv2.drawContours(img, filtered_cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

    return img, np.array(crosswalk_boxes)

def draw_stopline(orig, new_img, boxes):
    if len(boxes)== 0:
        return None

    print(boxes)
    
    xs = boxes[:,0,:][:,0]
    ys = boxes[:,0,:][:,1]
    
    x1_ind, x2_ind = np.argmin(xs), np.argmax(xs)
    cv2.line(new_img, (int(xs[x1_ind]), int(ys[x1_ind])), (int(xs[x2_ind]), int(ys[x2_ind])), (0,255,0), 3)
    cv2.imshow("Image", new_img)
    cv2.waitKey(0)

    Minv = cv2.getPerspectiveTransform(dst, src)

    newwarp = cv2.warpPerspective(new_img, Minv, (1280, 720))
    result = cv2.addWeighted(orig, 1, newwarp, 0.5, 0)

    cv2.imshow("Image", result)
    cv2.waitKey(0)


orig_img, cnts = process_img() #"./DSC_0153.JPG"
new_img, boxes = process_cnts(cnts)
draw_stopline(orig_img, new_img, boxes)

## TODO: transform back 
## TODO: need to add another sanity check: left gradient and right gradient should be similar 
'''

# find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


class ShapeDetector:
    def __init__(self):
        pass
 
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
 
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape



# import the necessary packages
#from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#    help="path to the input image")
#args = vars(ap.parse_args())

def pipeline():
    cnts, ratio, img = get_img_cnts()
    #highThresh, thresh_im = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #lowThresh = 0.5*highThresh
    
    # Edge detection
    #dst = cv2.Canny(warped, 50, 150, apertureSize = 3)
    # Copy edges to the images that will display the results in BGR
    #cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)

    #linesP = cv2.HoughLinesP(dst, rho = 1,theta = 1*np.pi/180,threshold = 50, minLineLength = 50,maxLineGap = 20)

    sd = ShapeDetector()
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        print(M)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
     
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
     
        # show the output image
        cv2.imshow("Image", img)
        cv2.waitKey(0)


pipeline()
'''
