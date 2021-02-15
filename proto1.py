

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

time.sleep(7)
wrist.start(7)
elbow.start(7)
shoulder.start(7)

global wrist_current 
global elbow_current 
global shoulder_current 



wrist_current = 7
elbow_current = 7
shoulder_current = 7


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



# For this first experiment I will manually 
# fill in the delta array

# I took out 20
delta = [[-37,-11.7,-6.1],[24.2, 0.4, -0.3],[-9,3, -2, -0.7],[-1.4, 0.4, -0.1],
[-19.8, 5.5, 2.6],[-52.8, 5.8,2.1],[0.4, 18.8,13],[-3.7, 9.4, 2.7],[4,0.8, -0.7],[-18.2, -3.2, -4],[21.1, -10.6, -4.7],
[23.1, -7.2, -5.1],[21.5, -15.8, -4.2], [37.1, 16.1, 10], [-27.6, -17.3, 9.9], [-1.9, 0.4, 0.6], [-38.8, 25, 4.8], [-71.4, 50.4, 19.7],
[2.3, -6.6, -1],[24.7, -27.9, -4.3], [72.1, -43.5, -21.3], [4.4, -0.7, -0.3], [0.2, 0.8, 0.8], [23.6, 0.6, 0.1], [-68.9, 19.8, 11.7],
[-11.4, -7.7, -6.9],[28.5, 12.7, 8.3]]




	
print("beginning the experiment")
time.sleep(5)

# exploring with the first servo
# loop for duty values from 2 to 12 (0 to 180 degrees)

counter = 1
for i in delta:
	print(counter)
	wrist_angle = i[2] 
	w_c = wristAngle(wrist_angle, wrist_current)
	wrist_current = w_c
	
	elbow_angle = i[1]
	e_c = elbowAngle(elbow_angle, elbow_current)
	elbow_current = e_c
	
	shoulder_angle = i[0]
	s_c = shoulderAngle(shoulder_angle, shoulder_current)
	shoulder_current = s_c
	
	time.sleep(2)
	
	print("-"*20)
	counter += 1


# Clean things up at the end
wrist.stop()
elbow.stop()
shoulder.stop()
GPIO.cleanup()
print("Goodbye")
