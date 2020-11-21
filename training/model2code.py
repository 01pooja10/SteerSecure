import dlib
import cv2
import numpy as np
from playsound import playsound
# from scipy.spatial import distance

def detect():
    capture = cv2.VideoCapture(0)

    det = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(r'\shape_predictor_68_face_landmarks.dat')
    c = 0
    alarm = 0

    while True:
        ret, frame = capture.read()
        #cv2.imshow("Webcam",frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det(grey)

        for face in faces:
            p = face.left()
            q = face.top()
            r = face.right()
            s = face.bottom()
            # cv2.rectangle(frame,(p,q),(r,s),(255,0,0),2)
            landmarks = predict(grey,face)

            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                #cv2.circle(frame,(x,y),3,(0,255,0),-1)

            yawn_threshold = 20
            eye_closed = 0.26
            eye_threshold = 47
            #eye_min=30

            x = EAR(landmarks,frame)
            if x < eye_closed:
                c += 1
                if c >= eye_threshold:
                    playsound(r'C:\Users\Pooja\Downloads\alarm.mp3')
                    c = 0

            y = yawn(landmarks, frame)
            if y > yawn_thresh:
                cv2.putText(frame, "YOU SEEM DROWSY!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

def EAR(landmarks,frame):
    left = []
    right = []
    alarm = 0

    # left eye
    for i in range(36,42):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        left.append((x,y))

        nextpt1 += 1
        if i == 41:
            nextpt1 = 36

        x1 = landmarks.part(nextpt1).x
        y1 = landmarks.part(nextpt1).y
        # cv2.line(frame,(x,y),(x1,y1),(0,255,0),1)

    # right eye
    for i in range(42,48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        right.append((x,y))

        nextpt2 = i+1
        if i == 47:
            nextpt2 = 42

        x1=landmarks.part(nextpt2).x
        y1=landmarks.part(nextpt2).y
        # cv2.line(frame,(x,y),(x1,y1),(0,255,0),1)

    # calculate ratio
    left_EAR = aspect_ratio(left)
    right_EAR = aspect_ratio(right)
    ratio = ((left_EAR + right_EAR)/2)
    return ratio

def aspect_ratio(eye):
    #numpy alternative for euclidean distance
    eye=np.squeeze(eye)
    a = np.linalg.norm(eye[1]-eye[5])
    b = np.linalg.norm(eye[2]-eye[4])
    c = np.linalg.norm(eye[0]-eye[3])

    final_ratio = (a + b)/(2.0 * c)
    return final_ratio

def yawn(landmarks,frame):
    top, bottom= [], []

    for i in range(51,54):
        x = landmarks.part(i).x
    for i in range(62,65):
        y = landmarks.part(i).y
    top.append((x,y))

    for i in range(65,68):
        x1 = landmarks.part(i).x
    for i in range(56,59):
        y1 = landmarks.part(i).y
    bottom.append((x1,y1))

    #cv2.line(frame,(x,y),(x1,y1),(0,255,0),1)

    top_mean = np.mean(top,axis=0)
    bottom_mean = np.mean(bottom,axis=0)

    dist = abs(top_mean[1]-bottom_mean[1])
    return dist

detect()
