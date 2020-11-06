#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import numpy as np
import re
from threading import Thread
from queue import Queue


# In[2]:


from multiprocessing import Process


# In[3]:


import multiprocessing


# In[4]:


def det_hands(outq,outrq,outlq):
    global handstate
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        img_cpy=image.copy()
        roimg=image.copy()
        img_cpy=cv2.flip(img_cpy, 1)
        roimg=cv2.flip(roimg, 1)
        limg_cpy=img_cpy.copy()
        height, width, channels = image.shape

        if not success:
            break

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

      # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lxcords=[] #left hand coordinates
        lycords=[]
        xcords=[] #right hand coordinates
        ycords=[]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i,v in enumerate(results.multi_hand_landmarks):


                handed=2
          #left hand is 0, right hand is 1
                handed_str=str(results.multi_handedness[i])
                handed_str=handed_str.split("\n")[3].split(":")[1]
                if (re.search("Right",handed_str)):
                    handed=1
                    xcords=[] #right hand coordinates
                    ycords=[]
            #print("right hand")
                else:
                    handed=0
                    lxcords=[] #left hand coordinates
                    lycords=[]



                hand_coords=str(v)
                hand_coords=hand_coords.split('landmark')
                hand_coords=hand_coords[1:]
                for lm in hand_coords:
                    b=lm.split('\n')
                    x=float(b[1].split(' ')[3])
                    y=float(b[2].split(' ')[3])
                    z=float(b[3].split(' ')[3])
                    if handed==1:
                        xcords.append(x)
                        ycords.append(y)
                        img_cpy = cv2.circle(img_cpy,(int(x*width),int(y*height)), 2, (255, 0, 0), 2)
                    elif handed ==0:
                        lxcords.append(x)
                        lycords.append(y)
                        lmg_cpy = cv2.circle(limg_cpy,(int(x*width),int(y*height)), 2, (0, 0, 255), 2)

        #cv2.putText(img_cpy,str(len(xcords)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #cv2.putText(img_cpy,str(len(lxcords)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        righthandcentre=(None,None)
        lefthandcentre=(None,None)
        x_max=None
        x_min=None
        y_max=None
        y_min=None
        lx_max=None
        lx_min=None
        ly_max=None
        ly_min=None



        if len(xcords)!=0  and len(ycords)!=0:
            x_max=max(xcords)
            x_min=min(xcords)
            y_max=max(ycords)
            y_min=min(ycords)
            img_cpy=cv2.rectangle(img_cpy, (int(x_min*width),int(y_min*height)), (int(x_max*width),int(y_max*height)), (255,0,0), 2)
            righthandcentre=((x_max+x_min)/2, (y_max+y_min)/2)
            img_cpy = cv2.circle(img_cpy,(int(righthandcentre[0]*width),int(righthandcentre[1]*height)), 2, (0, 255, 0), 2)


        if len(lxcords)!=0  and len(lycords)!=0:
            lx_max=max(lxcords)
            lx_min=min(lxcords)
            ly_max=max(lycords)
            ly_min=min(lycords)
            limg_cpy=cv2.rectangle(limg_cpy, (int(lx_min*width),int(ly_min*height)), (int(lx_max*width),int(ly_max*height)), (255,0,0), 2)
            lefthandcentre=((lx_max+lx_min)/2, (ly_max+ly_min)/2)
            limg_cpy = cv2.circle(limg_cpy,(int(lefthandcentre[0]*width),int(lefthandcentre[1]*height)), 2, (0, 255, 0), 2)

        thumbIsOpen = False
        firstFingerIsOpen = False
        secondFingerIsOpen = False
        thirdFingerIsOpen = False
        fourthFingerIsOpen = False

        LeftthumbIsOpen = False
        LeftfirstFingerIsOpen = False
        LeftsecondFingerIsOpen = False
        LeftthirdFingerIsOpen = False
        LeftfourthFingerIsOpen = False

        Lefthandfacing=False
        Righthandfacing=False





        #print (len(xcords))

        if len(xcords)>=21:
            if (xcords[20]>xcords[2] ):
                Righthandfacing=True
               # print("right hand facing")
                cv2.putText(img_cpy,"Right hand facing", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))


        if len(lxcords)>=21:
            if (lxcords[20]<lxcords[2] ):
                Lefthandfacing=True 
               # print("left hand facing")

                cv2.putText(limg_cpy,"Left hand facing", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))




        #RIGHT HAND       
        if len(xcords)>=5:
            KeyPoint = xcords[2]
            if (xcords[3] < KeyPoint and xcords[4] < KeyPoint):
                if Righthandfacing:
                    thumbIsOpen = True
                cv2.putText(img_cpy,"thumb is open", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if len(ycords)>=9:   
            KeyPoint = ycords[6]
            if (ycords[7] < KeyPoint and ycords[8] < KeyPoint):
                firstFingerIsOpen = True
                cv2.putText(img_cpy,"firstfinger is open", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if len(ycords)>=13:   
            KeyPoint = ycords[10]
            if (ycords[11] < KeyPoint and ycords[12] < KeyPoint):
                secondFingerIsOpen  = True
                cv2.putText(img_cpy,"secondfinger is open", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if len(ycords)>=17:   
            KeyPoint = ycords[14]
            if (ycords[15] < KeyPoint and ycords[16] < KeyPoint):
                thirdFingerIsOpen  = True
                cv2.putText(img_cpy,"thirdfinger is open", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if len(ycords)>=21:   
            KeyPoint = ycords[18]
            if (ycords[19] < KeyPoint and ycords[20] < KeyPoint):
                fourthFingerIsOpen  = True
                cv2.putText(img_cpy,"fourthfinger is open", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        #LEFT HAND 
        if len(lxcords)>=5:
            KeyPoint = lxcords[2]
            if (lxcords[3] > KeyPoint and lxcords[4] > KeyPoint):
                if Lefthandfacing:
                    LeftthumbIsOpen = True
                cv2.putText(limg_cpy,"Leftthumb is open", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))

        if len(lycords)>=9:   
            KeyPoint = lycords[6]
            if (lycords[7] < KeyPoint and lycords[8] < KeyPoint):
                LeftfirstFingerIsOpen = True
                cv2.putText(limg_cpy,"Leftfirstfinger is open", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255))

        if len(lycords)>=13:   
            KeyPoint = lycords[10]
            if (lycords[11] < KeyPoint and lycords[12] < KeyPoint):
                LeftsecondFingerIsOpen  = True
                cv2.putText(limg_cpy,"Leftsecondfinger is open", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))

        if len(lycords)>=17:   
            KeyPoint = lycords[14]
            if (lycords[15] < KeyPoint and lycords[16] < KeyPoint):
                LeftthirdFingerIsOpen  = True
                cv2.putText(limg_cpy,"Leftthirdfinger is open", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255))

        if len(lycords)>=21:   
            KeyPoint = lycords[18]
            if (lycords[19] < KeyPoint and lycords[20] < KeyPoint):
                LeftfourthFingerIsOpen  = True
                cv2.putText(limg_cpy,"Leftfourthfinger is open", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))




        handstate={
        "thumbIsOpen": thumbIsOpen,
        "firstFingerIsOpen" : firstFingerIsOpen,
        "secondFingerIsOpen" : secondFingerIsOpen,
        "thirdFingerIsOpen" : thirdFingerIsOpen,
        "fourthFingerIsOpen" : fourthFingerIsOpen,

        "LeftthumbIsOpen" : LeftthumbIsOpen,
        "LeftfirstFingerIsOpen": LeftfirstFingerIsOpen,
        "LeftsecondFingerIsOpen" : LeftsecondFingerIsOpen,
        "LeftthirdFingerIsOpen" : LeftthirdFingerIsOpen,
        "LeftfourthFingerIsOpen" : LeftfourthFingerIsOpen,

        "Lefthandfacing" : Lefthandfacing,
        "Righthandfacing" :Righthandfacing,

        "xcords" : xcords,
        "ycords" : ycords,

        "lxcords" : lxcords,
        "lycords" : lycords,

        "righthandcentre" :righthandcentre,
        "lefthandcentre" :lefthandcentre,

        "rboundingbox" : (x_min,y_min,x_max,y_max),
        "lboundingbox" : (lx_min,ly_min,lx_max,ly_max),

        }
        outq.put(handstate)
        outrq.put(img_cpy)
        outlq.put(limg_cpy)

        #cv2.imshow('MediaPipe Hands', img_cpy)
        #cv2.imshow('lMediaPipe Hands', limg_cpy)

        #if cv2.waitKey(5) & 0xFF == 27:
            #break
    #cap.release()
    #cv2.destroyAllWindows()


# In[ ]:




