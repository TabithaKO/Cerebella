# Cerebella
Gesture controlled robotic arm

This project uses computer vision to detect certain human joints in a video frame. The base code for the keypoint detection model is OpenPose model. 
OpenPose was modified by [Gunjan Sethi] (https://medium.com/@gsethi2409) for TensorFlow 2.x and the result parsing code was based off Marcelo Rovai's [article](https://towardsdatascience.com/realtime-multiple-person-2d-pose-estimation-using-tensorflow2-x-93e4c156d45f)

I wrote code that calculates the change in position/shape of the arm from frame to frame and transfers those changes to electric motors (Servos) that control the robotic arm.

## Files
### OpenPose Cerebella
This notebook contains the code that parses OpenPose inferences and calculates the change in position of the right arm

### OpenPose Cerebella 3d
This notebook contains the code that translates OpenPose inferences into 3d keypoints. The functions are also suited to calculate the changes in angle from frame to frame

### Proto1.py
This python file contains the code that I used to run my first arm demo. I collected angle changes from 20 select frames and wrote directly to the servos controlling the arm

### livestream.py
This file contains the code that passes an image through the OpenPose model, parses through the inferences, calculates the angle changes, and writes to the servos.


## Results:
Combining the functions from OpenPose Cerebella and Proto1 one should be able to control a robot arm using movement from a pre-recorded video.

## Demos:
OpenPose [Right Arm](https://youtu.be/x71DUxIfWlQ)

Proto1 [Robot Arm](https://youtu.be/A33udcIALBo)

## To Do:
1. Revise the code from proto1.py to allow reading of angles from a .csv file
2. Revise the code from OpenPose Cerebella to function as a readily implementable body of code
3. There's more I can do now that I have 3d keypoint values, but I need to work out how to calculate the coordinate changes with 3 axes instead of 2
