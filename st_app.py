import streamlit as st
import keras
import cv2

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
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224,224))

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
			# placeHolder.image(frame)
			# figure out how to have cv2 close with esc key

			key = cv2.waitKey(1)
			
			# cv2.waitKey not working
			if key == 27:
				break
		
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

def drowsinessDet():
	pass

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
		model = load_model('vgg_model.h5')
		distractionDet(model)

	elif option == 'Drowsiness Detector':
		drowsinessDet()

if __name__ == "__main__":
	main()
