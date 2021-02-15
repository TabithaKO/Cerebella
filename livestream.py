import argparse
import logging
import sys
import time
import os
import random
import string
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import math

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# This code allows me to control a servo using
# the RPi. I'll use the code provided by (Christopher Barnatt - Explaining Computers) 
# as a template which I will modify

import RPi.GPIO as GPIO
import time

#set GPIO numbering mode
GPIO.setmode(GPIO.BOARD)

#Set pn 11 as output and set servo 1 as pin 11 as PWM
GPIO.setup(11,GPIO.OUT)
wrist = GPIO.PWM(11,50) #11 is the pin and 50 = 50Hz pulse
#Set servo 2 on pin 13
GPIO.setup(13, GPIO.OUT)
elbow = GPIO.PWM(13,50)
#Set servo 3 to pin 50
GPIO.setup(15, GPIO.OUT)
shoulder = GPIO.PWM(15,50)

#start PWM running, but with value of 7 (pulse off)
#I'd like for all of my servos to start at 90
#because some of my diplacement values are negative

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



def wristAngle(angle, wrist_current):
	print("wrist current",wrist_current)
	displacement = round(angle/18,2)
	duty = displacement + wrist_current
	if duty > 0:
		print("wrist duty is:",duty) 
		wrist.ChangeDutyCycle(duty)
		wrist_current = duty
		
	else:	
		print("wrist duty remained constant:",wrist_current) 
		wrist.ChangeDutyCycle(wrist_current)
	
	
	return wrist_current
	
def elbowAngle(angle, elbow_current):
	print("elbow current",elbow_current)
	displacement = round(angle/18,2)
	duty = displacement + elbow_current 
	if duty > 0:
		print("elbow duty is:",duty) 
		elbow.ChangeDutyCycle(duty)
		elbow_current = duty
		
	else:	
		print("elbow duty remained constant:", elbow_current) 
		elbow.ChangeDutyCycle(elbow_current)
	
	return elbow_current
	
def shoulderAngle(angle, shoulder_current):
	print("shoulder current",shoulder_current)
	displacement = round(angle/18,2)
	duty = displacement + shoulder_current 
	if duty > 0:
		print("shoulder duty is:",duty) 
		shoulder.ChangeDutyCycle(duty)
		shoulder_current = duty
	else:
		print("shoulder duty remained constant", shoulder_current)	
		shoulder.ChangeDutyCycle(shoulder_current)
		
	return shoulder_current


def kpts():
	plot.figure(figsize=(30,30))
	plt.axis([0, image.shape[1],0,image.shape[0]])
	plt.scatter(*zip(*keypts_array),s=200, color='pink', alpha=0.6)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(img,alpha =0.7)
	ax = plt.gca()
	ax.set_ylim(ax.get_ylim()[::-1])
	ax.xaxis.tick_top()
	plt.grid()
	
	return plt
	
def magnitude(vec1, vec2):
	diff = vec1 - vec2
	squared =  diff ** 2
	summed = sum(squared)
	mag = math.sqrt(summed)
	return(mag)

# arguments: edges --> (s,e), (e,w), (s,w)
def angle_T1(se, ew, sw):
  # cosine of the shoulder vertex
	cosS = ((sw**2)+(se**2)-(ew**2))/(2*sw*se)
	beta = round(math.acos(cosS)*(180/(math.pi)),1)
	cosE = ((ew**2)+(se**2)-(sw**2))/(2*ew*se)
	alpha = round(math.acos(cosE)*(180/(math.pi)),1)
	cosW = ((sw**2)+(ew**2)-(se**2))/(2*sw*ew)
	gamma = round(math.acos(cosW)*(180/(math.pi)),1)

	result = np.array([beta,alpha,gamma])

	return result

# arguments: edges --> (s,e), (h,e), (s,h)
def angle_T2(se, sh, he):
  # cosine of the shoulder vertex
  cosS = ((sh**2)+(se**2)-(he**2))/(2*sh*se)
  theta = round(math.acos(cosS)*(180/(math.pi)),1)
  cosE = ((se**2)+(he**2)-(sh**2))/(2*se*he)
  omega = round(math.acos(cosE)*(180/(math.pi)),1)
  cosH = ((sh**2)+(he**2)-(se**2))/(2*sh*he)
  phi = round(math.acos(cosH)*(180/(math.pi)))

  result = np.array([theta, omega, phi])

  return result
	
def main():
	windowName = "Live Window Feed"
	print(windowName)
	cv2.namedWindow(windowName)
	cap = cv2.VideoCapture(0)
	
	if cap.isOpened():
		ret, frame = cap.read()
	else:
		ret = False
		
	# set the servos to 90 to begin
	wrist_current = 7
	elbow_current = 7
	shoulder_current = 7

		
	while ret:
		
		ret, frame = cap.read()
		
		#analyze the image 
		w, h = model_wh('432x368')
		if w == 0 or h == 0:
			e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
		else:
			e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
			
		# make an estimation on the frame
		# estimate human poses from a single image 
		image = common.read_imgfile(frame, None, None)
		if image is None:
			logger.error('Image can not be read')
			sys.exit(-1)

		t = time.time()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		elapsed = time.time() - t

		logger.info('inference image in %.4f seconds.' % (elapsed))

		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

			
		# collect the relevant keypoints
		
		# get the keypoints
		if len(humans) == 1:
			keypoints = str(str(str(humans[0]).split('BodyPart:')[1:]).split('-')).split(' score=')
			# parse through the results to get the specific keypoints
			keypoints_list=[]
			for k in range (len(keypoints)-1): 
				pnt = keypoints[k][-11:-1]
				pnt = tuple(map(float, pnt.split(', ')))
				keypoints_list.append(pnt)

			# create a keypoints array
			keypts_array = np.array(keypoints_list)
			keypts_array = keypts_array*(image.shape[1],image.shape[0])
			keypts_array = keypts_array.astype(int) 
    
			# collect the necessary inferences
			if len(keypts_array) > 11:
				shoulder = keypts_array[2]
				elbow = keypts_array[3]
				wrist = keypts_array[4]
				hip = keypts_array[8]
				holder = [shoulder, elbow, wrist, hip]
				kp_series.append(holder)
				scores_series.append([scores_list[2], scores_list[3],scores_list[4],scores_list[8]])
		
		
		# T1
		orig_beta = 0
		orig_alpha = 0
		orig_gamma = 0
		
		# T2
		orig_theta = 0
		orig_omega = 0
		orig_phi = 0
		
		
		# get the vectors
		shoulder_vec = kp_series[i][0]
		elbow_vec = kp_series[i][1]
		wrist_vec = kp_series[i][2]
		hip_vec = kp_series[i][3]

		# calculate the magnitudes
		# T1: s,e,w
		se_mag = magnitude(shoulder_vec, elbow_vec)
		sw_mag = magnitude(shoulder_vec, wrist_vec)
		ew_mag = magnitude(elbow_vec, wrist_vec)

		# T2: s,e,h
		# I've aready calculated se in T1
		sh_mag = magnitude(shoulder_vec, hip_vec)
		he_mag = magnitude(hip_vec, elbow_vec)

		# calculate the angles
		results_T1 = angle_T1(se_mag,ew_mag,sw_mag)
		angles_T1.append(results_T1.tolist())

		results_T2 = angle_T2(se_mag,sh_mag,he_mag)
		angles_T2.append(results_T2.tolist())

		# calculate the delta T1 & T2
		original_angles_T1 = np.array([orig_beta, orig_alpha, orig_gamma]) 
		T1_frame_delta = results_T1 - original_angles_T1
		T1_delta_rounded = []
		# for some reason some values weren't rounding
		# properly so I'll take the long route out for now
		for unit in T1_frame_delta:
			val = round(unit,1)
			T1_delta_rounded.append(val)
		

		original_angles_T2 = np.array([orig_theta, orig_phi, orig_omega])
		T2_frame_delta = results_T2 - original_angles_T2
		T2_delta_rounded = []
		for unit in T2_frame_delta:
			val = round(unit,1)
			T2_delta_rounded.append(val)
		


		# replace the original angles with the current angles 
		orig_beta, orig_aplha, orig_gamma = results_T1
		orig_theta, orig_phi, orig_omega =  results_T2
		
		
		# calculate the angle changes and write them to the servo
		
		wrist_angle = results_T1[2] 
		w_c = wristAngle(wrist_angle, wrist_current)
		wrist_current = w_c
	
		elbow_angle = results_T1[1]
		e_c = elbowAngle(elbow_angle, elbow_current)
		elbow_current = e_c
	
		shoulder_angle = results_T2[0]
		s_c = shoulderAngle(shoulder_angle, shoulder_current)
		shoulder_current = s_c
	
		

		
		
		cv2.imshow(windowName, frame)
		if cv2.waitKey(1) == 27:
			break
			
	cv2.destroyWindow(windowName)
	# Clean things up at the end
	wrist.stop()
	elbow.stop()
	shoulder.stop()
	GPIO.cleanup()
	
	cap.release()

if __name__=="__main__":
	main()
