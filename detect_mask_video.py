# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pyttsx3
from picamera import PiCamera
from smbus2 import SMBus
from mlx90614 import MLX90614
from time import sleep
import subprocess
#import _thread
#import threading
import pyrebase

firebaseConfig = {
  'apiKey': "AIzaSyB6nQO6Zhiu5ImedEiJ9i2URtSBA64kX2o",
  'authDomain': "covid-19-project-fyp.firebaseapp.com",
  'databaseURL': "https://covid-19-project-fyp-default-rtdb.asia-southeast1.firebasedatabase.app",
  'projectId': "covid-19-project-fyp",
  'storageBucket': "covid-19-project-fyp.appspot.com",
  'messagingSenderId': "701975883354",
  'appId': "1:701975883354:web:b5bcb74e65b2276c183263",
  'measurementId': "G-VRQDT03C8T"
};

firebase = pyrebase.initialize_app(firebaseConfig)

storage = firebase.storage()

database = firebase.database()

# initialisation
engine = pyttsx3.init()

#initialize temperature sensor bus and gpio
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5a)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
     
def getTempData():
    temp = sensor.get_object_1()
    return temp

# load our serialized face detector model from disk
prototxtPath = "/home/pi/Desktop/Face-Mask-Detection-master/face_detector/deploy.prototxt"
weightsPath = "/home/pi/Desktop/Face-Mask-Detection-master/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

img_counter = 0
temp_img_counter = 0

#Apply Algorithm
def applyLogic(label):
    print("Algorithm Activated")
    path_to_cloud_mask = "no_mask_images"
    path_to_cloud_temp = "high_temp_images"
    temp = getTempData()
    global img_counter
    text = "Warning! No Mask Detected!"
    text2 = "Scanning Temperature Now"
    text3 = "Mask Detected!"
    text4 = "Warning! High Temperature Detected!"
    text5 = "Normal Temperature"
    
    if (label=="No Mask"):
        print("Algorithm Detected")
        engine.say(text)
        engine.runAndWait()
        global vs
        vs.stream.release()
        sleep(1)
        camera = PiCamera()
        img_name = "no_mask_frame_{}.png".format(img_counter)
        camera.capture(img_name)
        print("{} written!".format(img_name))
        storage.child(path_to_cloud_mask).child(img_name).put(img_name)
        print("Image Uploaded")
        no_mask_data = {"Label":label}
        database.child("Users").child("Sensor").child("No Mask").child("Results").push(no_mask_data)
        os.remove(img_name)
        print("Image Removed")
        print("Closing Camera Service")
        camera.close()
        sleep(1)
        print("Starting Streaming Service")
        vs = VideoStream(src=0, framerate=30).start()
        sleep(2)
        
    elif (label=="Mask"):
        print("Temp Algorithm Detected")
        engine.say(text3)
        engine.say(text2)
        engine.runAndWait()
        sleep(2)
        if temp >= 30:
            print("High Temperature Detected")
            engine.say(text4)
            engine.runAndWait()
            vs.stream.release()
            sleep(1)
            camera = PiCamera()
            img_name2 = "high_temp_frame_{}.png".format(temp_img_counter)
            camera.capture(img_name2)
            print("{} written!".format(img_name2))
            storage.child(path_to_cloud_temp).child(img_name2).put(img_name2)
            print("Image Uploaded")
            high_temp_data = {"Temperature": temp, "Label": label}
            database.child("Users").child("Sensor").child("Mask").child("Results").push(high_temp_data)
            print("Temperature Uploaded")
            os.remove(img_name2)
            print("Image Removed")
            print("Closing Camera Service")
            camera.close()
            sleep(1)
            print("Starting Streaming Service")
            vs = VideoStream(src=0, framerate=30).start()
            sleep(2)
        
        else:
            engine.say(text5)
            engine.runAndWait()

        

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label_out = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
		#temperature sensor data
		temp = getTempData()
		#temp = sensor.get_object_1()
		person_temp = "Temp: {:.1f}".format(temp)
        
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label_out, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
		print(label)
		print(temp)
		sleep(2)
		img_counter += 1
		temp_img_counter += 1
		applyLogic(label)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break 

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

