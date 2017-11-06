#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:46:43 2017

@author: danbergeland
"""

import numpy as np
import cv2

CAPTURE_BOX_TOP = 200
CAPTURE_BOX_LEFT = 515
CAPTURE_BOX_SIDES = 350

def run_cam():
    cap = cv2.VideoCapture(0)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        #Grab the middle of the frame
        pred_img = get_midframe(gray)
        #Draw a white rect, so user knows where the pred_box is
        cv2.rectangle(gray,(CAPTURE_BOX_LEFT,CAPTURE_BOX_TOP),
                      (CAPTURE_BOX_LEFT+CAPTURE_BOX_SIDES,CAPTURE_BOX_TOP+CAPTURE_BOX_SIDES),
                      (255),4)
        
        #Resize to match trained network input
        pred_img = cv2.resize(pred_img, (28,28))
        
        #Write the prediction on the screen
        pred = 'Prediction: ' + str(1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray,pred,(400,100), font, 2,(255),2,cv2.LINE_AA)
    
        # Display the resulting frame
        cv2.imshow('Real-time MNIST Classifier',gray)
        #cv2.imshow('subset',pred_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
def get_midframe(img):
    l = CAPTURE_BOX_LEFT
    r = CAPTURE_BOX_LEFT+CAPTURE_BOX_SIDES
    t = CAPTURE_BOX_TOP
    b = CAPTURE_BOX_TOP+CAPTURE_BOX_SIDES
    return img[t:b, l:r]
    
    
if __name__ == "__main__":
    run_cam()