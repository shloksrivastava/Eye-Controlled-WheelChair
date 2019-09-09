import numpy as np
import cv2
import serial

ser = serial.Serial('COM8', 9600)
cap = cv2.VideoCapture(0) #640,480
w = 640
h = 480
#take input from camera

while 1:
    ret, frame = cap.read() #"frame" will get the next frame in the camera (via "cap"). "ret" will obtain return value from getting the camera frame, either true of false.
    if ret==True:
#downsample
#frameD = cv2.pyrDown(cv2.pyrDown(frame))
#frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)

        #detect face
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #convert from colored to gray
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        #faces = cv2.CascadeClassifier('haarcascade_eye.xml')
        #detected = faces.detectMultiScale(frame, 1.3, 5) 
        detected = eye_cascade.detectMultiScale(frame, 1.3,5)#cv2.CascadeClassifier.detectMultiScale(image, scaleFactor, minimumNeighbors)

        #faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
        
        MainImage = frame
        pupilO = frame
        
        windowClose = np.ones((5,5),np.uint8) #Taking a matrix of size 5 as the kernel
        windowOpen = np.ones((2,2),np.uint8) #taking a matrix of size 2 as the kernel
        windowErode = np.ones((2,2),np.uint8) 

        #draw square
        #for x,y,w,h in eyes
        for (x,y,w,h) in detected:
                cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1) #a rectangle is drawn (start point, end point,color code,width of line)
                cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)
                cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
                MainImage = cv2.equalizeHist(frame[int(y+(h*.25)):(y+h), x:(x+w)]) #improving contrast of image by spreading out the pixels to all intensities
                pupilO = MainImage
                ret, MainImage = cv2.threshold(MainImage,55,255,cv2.THRESH_BINARY)            
                
                #If pixel value > threshold value, it is assigned one value say white. If pixelValue < thresholdValue, it is assiged another value.
                #parameters: (src img, thresholdValue,MaxValue to be given,type of thresholding)
                #after converting it to a binary image, we will now use morphological transformation.
                
                MainImage = cv2.morphologyEx(MainImage, cv2.MORPH_CLOSE, windowClose)
                #Closing: To remove false negatives. This is where we have detected shape,but we still have some unwanted pixels within the object. Closing clears them.
                MainImage = cv2.morphologyEx(MainImage, cv2.MORPH_ERODE, windowErode)
                #Erodes the edges.A slider (windowErode here) size 2 x 2 pixels is slid around, and if all of the pixels are white,we get white, otherwise black. This may help eliminate some white noise.
                #A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel i.e windowErode is 1, otherwise it is eroded (made to zero).
                MainImage = cv2.morphologyEx(MainImage, cv2.MORPH_OPEN, windowOpen)
                
                #The goal of opening is to remove "false positives". Sometimes, in the background, you get some pixels here and there of "noise.". So to remove them.

                #so above we did image processing to get the pupil..
                #now we find the biggest blob and get the centriod
                threshold = MainImage
                #threshold = cv2.inRange(MainImage,250,255)             #get the blobs
                _, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                #Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
                #print(contours)
                #arguments: cv2.findContours(sourceimage, contour retreival mode, contour approximation method )
                #https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
                #cnt = contours[4]
                #cv2.drawContours(threshold, [cnt], 0, (0,255,0), 3)
                
                #if there are 3 or more blobs, delete the biggest and delete the left most for the right eye
                #if there are 2 blob, take the second largest
                #if there are 1 or less blobs, do nothing
                
                if len(contours) >= 2:
                        #find biggest blob
                        maxArea = 0
                        MAindex = 0                     #to get the unwanted frame 
                        distanceX = []          #delete the left most (for right eye)
                        currentIndex = 0
                        for cnt in contours:
                                area = cv2.contourArea(cnt) #calculates area of that particular contour
                                center = cv2.moments(cnt) #to calculate some features like COM, centroid etc.
                                if float(center['m00']) != 0:
                                    cx,cy = int(float(center['m10'])/float(center['m00'])), int(float(center['m01'])/float(center['m00'])) #centroid coordinates
                                    distanceX.append(cx) #append the value of the centroid loc in x axis to distanceX  
                                    if area > maxArea:
                                            maxArea = area
                                            MAindex = currentIndex
                                    currentIndex = currentIndex + 1
        
                        del contours[MAindex]           #remove the picture frame contour
                        del distanceX[MAindex]
                
                eye = 'right'

                if len(contours) >= 2:          #delete the left most blob for right eye
                        if eye == 'right':
                                edgeOfEye = distanceX.index(min(distanceX))
                        else:
                                edgeOfEye = distanceX.index(max(distanceX))     
                        del contours[edgeOfEye]
                        del distanceX[edgeOfEye]

                if len(contours) >= 1:          #get largest blob
                        maxArea = 0
                        for cnt in contours:
                                area = cv2.contourArea(cnt)
                                if area > maxArea:
                                        maxArea = area
                                        largeBlob = cnt
                                
                if len(largeBlob) > 0:  
                        center = cv2.moments(largeBlob)
                        cx,cy = int(float(center['m10'])/float(center['m00'])), int(float(center['m01'])/float(center['m00']))
                        #print(cx,cy)
                        cv2.circle(pupilO,(cx,cy),5,255,-1)
                        #if ((cx >=35 and cx <= 45) and cy <= 17):
                            #print('top')
                        #if (cx <= 40 and cy <= 25):
                            #print('right')
                        if (cx >= 50 and cx <= 69): #and (cy >= 25 and cy <= 55)):
                            print('straight')
                            ser.write("F".encode())
                        if (cx >= 29 and cx <= 49): #and (cy >= 25 and cy <= 35)):
                            print('right')
                            ser.write("R".encode())
                        if (cx >=70 and cx <100): #and (cy >= 25 and cy <= 55)):
                            print('left')
                            ser.write("L".encode())
                            
                        #if ((cx >= 35 and cx <= 45) and cy >= 17):
                            #print('bottom')


        #show picture
        cv2.imshow('frame',pupilO)
        #height = frame.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #width = frame.get(cv2.CAP_PROP_FRAME_WIDTH)
        #print(height)
        #print(width)
        #cv2.imshow('frame2',MainImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#else:
        #break
