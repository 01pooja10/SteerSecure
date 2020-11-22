# SteerSecure
__Driver Safety and Security__

## Data
__Dataset Used__: State Farm Distracted Driver Detection.

You may find the dataset for distracted drivers below at:

https://www.kaggle.com/c/state-farm-distracted-driver-detection

## Description
**SteerSecure** is a driver-safety app built using machine learning libraries such as Tensorflow and OpenCV, and scripted and deployed to the web using Streamlit. Our aim with SteerSecure is to make vehicular travel safer. Our two primary web-applications are:

-  Distraction Detection
-  Drowsiness Detection


The data is very clear: more than 30% of accidents are caused due to fatigue caused by long, nonstop hours of driving and sleep-deprication. A large majority of accidents are also caused by distracted drivers: texting, speaking to passengers, talking on the phone, etc. Such accidents cause more than 400 deaths everyday in India. Not only is it a tragic loss of life, but it also costs our economy upwards of 1 trillion rupees every year.

Our goal, very simply, is to curb this loss of life and property as best we can. SteerSecure aims to discourage distracted drivers and encourage drowsy drivers to take rest, or pull over before continuing on the journey later.

## Problem Statement
An application called __SteerSecure__ for driver safety using distraction detection, drowsiness detection and a proposed chatbot for informing customers.

## Screenshots of the Web-App
__Welcome Page__
![Screen Shot 2020-11-22 at 08 09 26](https://user-images.githubusercontent.com/20011207/99892796-d7171080-2c9e-11eb-82c7-4b79c844f186.png)

__Distracted Detector Page__
![Screen Shot 2020-11-22 at 08 11 25](https://user-images.githubusercontent.com/20011207/99892820-0594eb80-2c9f-11eb-9462-85e9c59624cc.png)
![Screen Shot 2020-11-22 at 08 11 45](https://user-images.githubusercontent.com/20011207/99892808-f0b85800-2c9e-11eb-920b-80a062738bcc.png)

__Drowsiness Detector Page__
![Screen Shot 2020-11-22 at 08 10 56](https://user-images.githubusercontent.com/20011207/99892826-2a895e80-2c9f-11eb-82aa-af34ce9379fa.png)
![Screen Shot 2020-11-22 at 08 11 11](https://user-images.githubusercontent.com/20011207/99892831-3bd26b00-2c9f-11eb-9bac-0eb2b2e8549f.png)

__Nav ~ the Chatbot__
![Screen Shot 2020-11-22 at 08 12 42](https://user-images.githubusercontent.com/20011207/99892782-b5b62480-2c9e-11eb-866a-218a03fedc6e.png)

## Future Developments to SteerSecure:
1) Integrate with an interface inside your car, similar to a dashboard, for quick access to our services
2) Chatbot automatically sends messages to your loved ones alerting them if it notices regular patterns of drowsiness or Distraction
3) Car slows down and pulls over automatically when the SteerSecure app detects driver getting very drowsy
4) Car can also automatically go into self-driving mode when it detects that the user is too exhausted

## Challenges Faced
1) Deploying to Streamlit and Heroku failed for a number of reasons, including working in different OSes and Conda environments but mostly due to size of the imports and the resource limits.
2) Getting the webcam to shut off after Stop button is pressed on Streamlit to preserve user privacy.
3) Getting correct predictions in notebook but incorrect ones in our web-app for distracted drivers.
4) We were not able to solve the last issue entirely but definitely look forward to doing so in the future!

## License
MIT © Pooja Ravi

This project is licensed under the MIT License - see the [License](LICENSE) file for details

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with ❤️ by Team Armada
</p>
