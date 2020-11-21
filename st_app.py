import streamlit as st
from tensorflow import keras
import cv2
import numpy as np
import dlib
from playsound import playsound

list1 = ['Adjusting hair or makeup', 'Drinking a beverage', 'Operating the radio', 'Reaching behind', 'Safe driving', 'Talking on the phone(left)', 'Talking on the phone(right)', 'Talking to a passenger', 'Texting using phone(left)', 'Texting using phone(right)']

def load_model(path):
	model = keras.models.load_model(path)
	return model

def realTime(model):
	capture = cv2.VideoCapture(0)
	real_time = []

	placeHolder = st.empty()

	while True:
		ret, frame = capture.read()
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224,224))/255

		real_time.append(img)
		x = real_time

		for i in range(len(x)):
			# x[i] = x[i].expand_dims(axis = 0)
			# x[i] = x[i].reshape(1, 224, 224, 3)
			mm = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
			mm[0] = x[i]

			pred = model.predict(mm)
			kk = np.argmax(pred)

		cv2.putText(frame, list1[kk], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
		# cv2.imshow('frame', frame)
		placeHolder.image(frame, use_column_width=True, channels='RGB')
		# figure out how to have cv2 close with esc key

		key = cv2.waitKey(1)

		# cv2.waitKey not working
		if key == 27:
			break

	# webcam not stopping even after capture.release()
	# maybe try st.stop()
	capture.release()
	cv2.destroyAllWindows()
	return real_time

def distractionDet(model):
	html="""
	<style>

	</style>
	"""

	st.title("Distraction Detector")
	st.markdown("Detector for distractedness")

	st.markdown(html, unsafe_allow_html=True)

	if st.button("Launch Application"):
		realTime(model)

# Drowsiness
def EAR(landmarks, frame):
	left = []
	right = []
	alarm = 0

	# left eye
	for i in range(36,42):
		x = landmarks.part(i).x
		y = landmarks.part(i).y
		left.append((x,y))

		nextpt1 = i + 1
		if i == 41:
			nextpt1 = 36

		x1 = landmarks.part(nextpt1).x
		y1 = landmarks.part(nextpt1).y

	# right eye
	for i in range(42,48):
		x = landmarks.part(i).x
		y = landmarks.part(i).y
		right.append((x,y))

		nextpt2 = i + 1
		if i == 47:
			nextpt2 = 42

		x1 = landmarks.part(nextpt2).x
		y1 = landmarks.part(nextpt2).y

	# calculate ratio
	left_EAR = aspect_ratio(left)
	right_EAR = aspect_ratio(right)
	ratio = ((left_EAR + right_EAR)/2)
	return ratio

def aspect_ratio(eye):
	eye = np.squeeze(eye)
	a = np.linalg.norm(eye[1] - eye[5])
	b = np.linalg.norm(eye[2] - eye[4])
	c = np.linalg.norm(eye[0] - eye[3])

	return (a + b)/(2.0 * c)

def yawn(landmarks, frame):
	top, bottom = [], []

	for i in range(51,54): # should rather be 51, 54
		x = landmarks.part(i).x

	for i in range(62,65): # should rather be 62, 65
		y = landmarks.part(i).y

	top.append((x,y))

	for i in range(65,68): # should rather be 66, 69
		x1 = landmarks.part(i).x

	for i in range(56,59): # should rather be 57, 60
		y1 = landmarks.part(i).y

	bottom.append((x1,y1))

	top_mean = np.mean(top, axis=0)
	bottom_mean = np.mean(bottom, axis=0)

	dist = abs(top_mean[1] - bottom_mean[1])
	return dist

def detect():
	capture = cv2.VideoCapture(0)

	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(r"file dependencies/shape_predictor_68_face_landmarks.dat")
	c = 0
	alarm = 0
	placeHolder2 = st.empty()

	while True:
		ret, frame = capture.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = detect(gray)

		for face in faces:
			p = face.left()
			q = face.top()
			r = face.right()
			s = face.bottom()
			landmarks = predict(gray, face)

			for i in range(68):
				x = landmarks.part(i).x
				y = landmarks.part(i).y

			yawn_threshold = 22
			eye_closed = 0.26
			eye_threshold = 47
			eye_min = 30

			x = EAR(landmarks, frame)
			if x < eye_closed:
				c += 1
				if c >= eye_threshold:
					playsound(r'file dependencies/alarm.mp3')
					c = 0

			y = yawn(landmarks, frame)
			if y > yawn_threshold:
				cv2.putText(frame,"YOU SEEM DROWSY!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

			placeHolder2.image(frame, caption="Driver Live Video", use_column_width=True)

		key = cv2.waitKey(1)
		if key == 27:
			break

	capture.release()
	cv2.destroyAllWindows()

def drowsinessDet():
	st.title("Drowsiness Detector")

	if st.button("Launch Application"):
		detect()

def main():
	link = 'https://images.unsplash.com/photo-1519865241348-e0d7cd33a287?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80'
	st.image(link, use_column_width=True)

	menu = ['Welcome', 'Drowsiness Detector', 'Distraction Detector']
	option = st.sidebar.selectbox('Choose of any of our 2 web-apps', menu)

	st.title("SteerSecure ~ Driver Security")
	st.header("SteerSecure is an ML web-app which has been trained to ensure the safety of drivers by intelligently monitoring their levels of distraction, and fatigue.")

	st.markdown("")
	st.markdown("")

	if option == 'Distraction Detector':
		model = load_model('models/vgg_model.h5')
		distractionDet(model)

	elif option == 'Drowsiness Detector':
		drowsinessDet()

if __name__ == "__main__":
	main()
