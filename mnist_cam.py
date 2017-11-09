#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:46:43 2017

@author: danbergeland
"""
import cv2
import tensorflow as tf
import os
import numpy as np
import time

CAPTURE_BOX_TOP = 200
CAPTURE_BOX_LEFT = 515
CAPTURE_BOX_SIDES = 350

export_dir = './mnist_model'

def run_cam():

    cap = cv2.VideoCapture(0)
    #Use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING], export_dir)
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
            
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
        
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            #Grab the middle of the frame
            pred_img = get_midframe(gray)
            #Draw a white rect, so user knows where the pred_box is
            cv2.rectangle(frame,(CAPTURE_BOX_LEFT,CAPTURE_BOX_TOP),
                          (CAPTURE_BOX_LEFT+CAPTURE_BOX_SIDES,CAPTURE_BOX_TOP+CAPTURE_BOX_SIDES),
                          (0,255,0),4)
            
            #Resize to match trained network input
            pred_img_raw = cv2.resize(pred_img, (28,28))
            #invert & threshold
            pred_img_raw = 255-pred_img_raw
            thresh = 180
            pred_img_raw[pred_img_raw<thresh] = 0
            #pred_img = np.ravel(pred_img_raw)
            display_pred_img = pred_img_raw
            pred_img_raw = np.reshape(pred_img_raw,(1,784))
            pred_img_raw = pred_img_raw/255.
    
            #run the net
            softmax_tensor = sess.graph.get_tensor_by_name('softout:0')
            predictions = sess.run(softmax_tensor, {'in:0': pred_img_raw,'dropout/Placeholder:0':1.0})
            prediction = predictions[0]    
            # Get the highest confidence category.
            prediction = prediction.tolist()
            max_value = max(prediction)
            predicted_label = prediction.index(max_value) 
            print(prediction)
            
            frame = insert_midframe(frame, display_pred_img)
            frame = cv2.flip(frame,1)
            
            #Write the prediction on the screen
            pred = 'Prediction: ' + str(predicted_label) + ' @ ' + str(max_value) + ' conf.'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,pred,(400,600), font, 2,(255,255,255),2,cv2.LINE_AA)
            # Display the resulting frame
            cv2.imshow('Real-time MNIST Classifier',frame)
            
            #limit frame rate
            time.sleep(.05)
            #cv2.imshow('subset',cv2.resize(pred_img_raw,(600,600)))
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

def insert_midframe(img,insert):
    l = CAPTURE_BOX_LEFT
    r = CAPTURE_BOX_LEFT+CAPTURE_BOX_SIDES
    t = CAPTURE_BOX_TOP
    b = CAPTURE_BOX_TOP+CAPTURE_BOX_SIDES
    insert = cv2.flip(insert,1)
    img[t:b, l:r] = cv2.cvtColor(cv2.resize(insert,(CAPTURE_BOX_SIDES,CAPTURE_BOX_SIDES)), cv2.COLOR_GRAY2BGR)
    return img
    
    
if __name__ == "__main__":
    run_cam()