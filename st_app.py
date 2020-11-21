# General Imports
import streamlit as st
import numpy as np
import keras
import numpy as np
import cv2

# Drowsiness Imports
from pydub import AudioSegment
from pydub.playback import play
import dlib

# Chatbot Imports
import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import random
import pickle

# Distraction
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

		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224,224))

		img = keras.preprocessing.image.img_to_array(img)
		img = img / 255.0

		real_time.append(img)

		for i in range(len(real_time)):
			mm = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
			mm[0] = real_time[i]

			pred = model.predict(mm)
			# kk = np.argmax(pred)

		print("Prediction Array: {}".format(pred))
		print("amax: {}".format(np.amax(pred)))

		if np.amax(pred) > 0:
			kk = np.argmax(pred)
			print("kk: {}".format(kk))
			cv2.putText(frame, list1[kk], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

		placeHolder.image(frame, use_column_width=True, channels='BGR')

		key = cv2.waitKey(1)

		if key == 27:
			break

	st.stop()
	capture.release()
	cv2.destroyAllWindows()

def distractionDet(model):
	html="""
	<style>
    .element-container:nth-child(9) {
      left: 240px;
      top: 0px;
    }
	</style>
	"""

	st.title("Distraction Detector")
	st.markdown("Our distraction app takes in live video of the driver and processes the data in real-time to detect any distracted activities. When our machine-learning model finds the user to be in a distracted state such as drinking a beverage, talking to a passenger and so on, the activity done is shown immediately to the driver on their dashboard screen. In such a way, a live video feed of them driving along with a log of their distractedness will encourage alertness and discourage irresponsible behavior.")

	st.subheader("The app expects the camera to be placed above the passenger's window tilted slightly downwards, on the driver's right. Our application is optimized for such an angle and this setup would be recommended.")
	st.subheader("Press the button below to get started. When you're done, click the Stop button on the top right. Your webcam should turn off in a few moments.")

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

	for i in range(50, 53):
		x = landmarks.part(i).x

	for i in range(61, 64):
		y = landmarks.part(i).y

	top.append((x,y))

	for i in range(56, 59):
		x1 = landmarks.part(i).x

	for i in range(65, 68):
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

			yawn_threshold = 20
			eye_closed = 0.26
			eye_threshold = 30

			x = EAR(landmarks, frame)
			if x < eye_closed:
				c += 1
				if c >= eye_threshold:
					song = AudioSegment.from_wav('file dependencies/alarm.wav')
					play(song)
					c = 0

			y = yawn(landmarks, frame)
			if y > yawn_threshold:
				cv2.putText(frame,"YOU SEEM DROWSY!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

			placeHolder2.image(frame, caption="Driver Live Video", use_column_width=True,channels='BGR')

		key = cv2.waitKey(1)
		if key == 27:
			break

	capture.release()
	cv2.destroyAllWindows()

# Chatbot
def load_chatbot(path):
	model = keras.models.load_model(path)
	return model

def input_bag(sen, words):
	lemm = WordNetLemmatizer()

	bag = [0] * len(words)
	wrds = nltk.word_tokenize(sen)

	wrds = [lemm.lemmatize(w.lower()) for w in wrds]

	for s in wrds:
		for i, j in enumerate(words):
			if j == s:
				bag[i] = 1

	return np.array(bag)

def navChatbot(model, textInput):
	with open('file dependencies/chatbot_intents.json') as file:
		data = json.load(file,strict=False)

	words = pickle.load(open('file dependencies/words.pkl','rb'))
	labels = pickle.load(open('file dependencies/labels.pkl','rb'))

	while True:
		if textInput.lower() == 'exit':
			return "😴"

		bag = input_bag(textInput, words)
		result = model.predict(np.array([bag]))[0]
		pred_index = np.argmax(result)

		tag = labels[pred_index]

		for val in data['intents']:
			if val['tag'] == tag:
				resp = val['responses']
				break

		return random.choice(resp)

def drowsinessDet():

	html="""
	<style>
    .element-container:nth-child(9) {
      left: 240px;
    }
	</style>
	"""

	st.title("Drowsiness Detector")
	st.markdown("Our drowsiness detection app cleverly keeps track of the driver's fatigue by monitoring the eyes as well as any yawns. If the eyes of the driver are being closed for more than a few milliseconds (something called microsleep in sleep science), an alarm is sounded immediately to alert the driver. This element has been added because microsleep can be incredibly dangerous and is often responsible for a majority of the accidents happening due to long-haul rides. In addition, if the app sees the driver yawning, a warning message is displayed indicating the driver's fatigue.")

	st.markdown(html, unsafe_allow_html=True)

	st.subheader("The app expects the camera to be placed above the steering wheel on top of the speedometer, directly facing the driver. Our application is optimized for such an angle and this setup would be recommended.")
	st.subheader("Press the button below to get started. When you're done, click the Stop button on the top right. Your webcam should turn off in a few moments.")

	if st.button("Launch Application"):
		detect()

# MAIN Function
def main():

	html = """
	<style>
	.sidebar .sidebar-content {
		background-image: linear-gradient(#36cf9c, #27aedb);
		color: white;
	}
	</style>
	"""

	st.markdown(html, unsafe_allow_html=True)

	menu = ['Welcome', 'Drowsiness Detector', 'Distraction Detector', 'Nav ~ the Chatbot']
	with st.sidebar.beta_expander("Menu", expanded=False):
		option = st.selectbox('Select from any of our 3 applications', menu)
		st.subheader("Made with ❤️ by Team Armada")

	if option == 'Welcome':
		html = """
		<style>
		.element-container:nth-child(4)
		{
			color: #40E0D0;
		}
		</style>
		"""

		st.markdown(html, unsafe_allow_html=True)

		link = 'https://images.unsplash.com/photo-1519865241348-e0d7cd33a287?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80'
		st.image(link, use_column_width=True)

		st.title("SteerSecure 🚗 ~ Driver Safety")
		st.header("SteerSecure is an ML web-app which has been trained to ensure the safety of drivers by intelligently monitoring their levels of distraction, and fatigue.")

		st.markdown("Our goal is very simple: to ensure that everybody gets home safely. More than 44% of traffic accidents occur not due to technical malfunctions or driving errors, but simply due to fatigue. Combined often with sleep deprivation, the incredibly dangerous concept of microsleep occurs.")
		st.markdown("[Microsleep](https://en.wikipedia.org/wiki/Microsleep) is a momentary episode of half-sleep where a person engaged in some activity, simultaneously deprived of sleep, is just about to doze off. At this point, one's eyes may shut for a few seconds without them even realizing. Countless number of accidents and mishaps have occurred due to this phenomenon of microsleep, including the infamous 1986 Chernobyl Nuclear Plant Disaster.", unsafe_allow_html=True)
		st.markdown("SteerSecure's goal is to keep drivers safe, alert and hence, secure.")


		st.title("Statistics 📈")
		st.markdown("Overworked truck drivers in India often report being forced to drive for more than 20-25 hours without stopping. This leads to a huge amount of the overturning of trucks and lorries that we often see on the highways.")

		st.markdown("Below is a graph highlighting the percentage of fatigue-related crashes and after what amount of time they occurred. As you can see, the number increases exponentially, and is dangerously high for more than 12 hours. Most long-haul truck drivers drive for much longer time spans so we can clearly see how dangerous it can get not just for them but for everyone on the highway.")

		st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Hours_of_service_FMCSA_study.svg/560px-Hours_of_service_FMCSA_study.svg.png', use_column_width=True)

		st.subheader("Not only trucks and lorries, but SteerSecure is geared towards cars as well.")

		st.markdown("Several studies and surveys done throughout India highlight the dire state of drivers. More than 70% admitted to talking on the phone, listening to music while more than 5% of all surveyed drivers confessed to watching videos while driving.")

		st.subheader("All-in-all, the statistics on traffic accidents in India are quite appalling. Over 400 people die tragically every day and more than 1500 are hospitalized with injuries. This is a tragic reality that we all need to try and set right. Not only is it a needless and tragic loss of life, but it also ends up costing the economy more than 1 trillion rupees in property damage and overhead economic costs every year.")

		st.title("Our aim is to try and rectify this in our own small way.")

	elif option == 'Distraction Detector':
		html = """
		<style>
		.element-container:nth-child(4)
		{
			color: #40E0D0;
		}
		</style>
		"""

		st.markdown(html, unsafe_allow_html=True)

		st.image('https://images.unsplash.com/photo-1520088096110-20308c23a3cd?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80', use_column_width=True)
		model = load_model('models/vgg_model.h5')
		distractionDet(model)

	elif option == 'Drowsiness Detector':

		html = """
		<style>
		.element-container:nth-child(4)
		{
			color: #40E0D0;
		}
		</style>
		"""
		st.markdown(html, unsafe_allow_html=True)

		st.image('https://atlinjurylawgroup.com/wp-content/uploads/2019/12/image001-1.jpg', use_column_width=True)
		drowsinessDet()

	elif option == 'Nav ~ the Chatbot':

		html = """
		<style>
		.element-container:nth-child(3)
		{
			color: #40E0D0;
		}
		</style>
		"""
		st.markdown(html, unsafe_allow_html=True)

		model = load_chatbot('models/chatbot_model.h5')

		st.title("Beep, boop. This is Nav. 🤖")

		hints = ["Say hello!", "Type steersecure to ask Nav about our product!", "Ask Nav what is microsleep to get a detailed explanation!", "Ask how do accidents happen to see what he has to say.", "Ask Nav how to solve this and see what he says!", ""]
		st.subheader("Hint: {}".format(random.choice(hints)))

		textInput = st.text_input("You: ", value = "Ask Nav something! When you're done, just type exit to leave!")
		response = navChatbot(model, textInput)

		st.text_area("Nav:", value=response, height=200)

if __name__ == "__main__":
	main()
