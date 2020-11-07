# Gesture
Hand tracking and keypoint detection using python and MediaPipe

Dependencies:
opencv
mediapipe
numpy 

detecthands_multiprocess.det_hands(outq,outrq,outlq) accepts and writes to three queues:
outrq- video stream with right hand keypoint mask 
outlq- video stream with left hand keypoint mask 
outq - Python dictionary containing:
  
        thumbIsOpen : boolean. Defaults to False. True if Right thumb open
        firstFingerIsOpen : boolean. Defaults to False. True if  Right first finger open
        secondFingerIsOpen : boolean. Defaults to False. True if Right second finger open
        thirdFingerIsOpen :  boolean. Defaults to False. True if Right third finger open
        fourthFingerIsOpen :  boolean. Defaults to False. True if Right fourth finger open

        LeftthumbIsOpen : boolean. Defaults to False. True if Left thumb finger open
        LeftfirstFingerIsOpen: boolean. Defaults to False. True if Left first finger open
        LeftsecondFingerIsOpen : boolean. Defaults to False. True if Right second finger open
        LeftthirdFingerIsOpen : boolean. Defaults to False. True if Right third finger open
        LeftfourthFingerIsOpen : boolean. Defaults to False. True if Right fourth finger open

        Lefthandfacing : boolean. Defaults to False .True if left hand facing camera
        Righthandfacing : boolean. Defaults to False .True if right hand facing camera

        xcords : list 21 x coordinates of right hand,
        ycords :list 21 y coordinates of right hand,
        lxcords : list 21 x coordinates of left hand,
        lycords : list 21 y coordinates of left hand,

        righthandcentre :(x,y) coordinates of right hand centre
        lefthandcentre :(x,y) coordinates of left hand centre,

        rboundingbox : In the form (x_min,y_min,x_max,y_max). Bounding box coordinates for right hand
        lboundingbox: In the form (x_min,y_min,x_max,y_max). Bounding box coordinates for right hand

       
