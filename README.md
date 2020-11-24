# SteerSecure
__Driver Safety and Security__

## Data
__Dataset Used__: State Farm Distracted Driver Detection.

You may find the dataset for distracted drivers below at:

https://www.kaggle.com/c/state-farm-distracted-driver-detection

## Devfolio Project Page & Youtube Demo Video
SteerSecure was built as Pooja Ravi and Aditya Shukla's official submission at Octahacks 3.0, a 36 hour virtual hackathon, hosted by Chitkara University. Check out our Devfolio submission page and a Youtube video demonstrating SteerSecure below.

Devfolio Page: https://devfolio.co/submissions/steersecure

Youtube: https://www.youtube.com/watch?v=zp9t6rTe8Ws

## Description
**SteerSecure** is a driver-safety app built using machine learning libraries such as Tensorflow and OpenCV, and scripted and deployed to the web using Streamlit. Our aim with SteerSecure is to make vehicular travel safer. Our three primary web-applications are:

-  Distraction Detection
-  Drowsiness Detection
-  Chatbot to provide insight into our web app

The data is very clear: more than 30% of accidents are caused due to fatigue caused by long, nonstop hours of driving and sleep-deprivation. A large majority of accidents are also caused by distracted drivers: texting, speaking to passengers, talking on the phone, etc. Such accidents cause more than 400 deaths everyday in India. Not only is it a tragic loss of life, but it also costs our economy upwards of 1 trillion rupees every year.

Our goal, very simply, is to curb this loss of life and property as best we can. SteerSecure aims to discourage distracted drivers and encourage drowsy drivers to take rest, or pull over before continuing on the journey later.

## Problem Statement
As statistics show, there occur more than 400 deaths per day in India only due to drivers getting distracted or too drowsy to keep their eyes on the road. So, we decided to build an application called __SteerSecure__ to tackle this problem. Our deep learning based web application is built for driver safety; it contains components like distraction detection, drowsiness detection and a chatbot for informing customers about information regarding driver safety, app usage and navigation through different components.

## Features of SteerSecure
-  Detect 9 types of distractions faced by drivers like texting, adjusting hair or makeup, talking to a passenger etc.
-  Detect drowsiness and yawns among drivers using real time video from a camera
-  Chatbot provides useful statistics and information regarding our problem statement and usage of SteerSecure web app.

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

## Challenges Faced
1) Deploying to Streamlit and Heroku failed for a number of reasons, including working in different OSes and Conda environments but mostly due to size of the imports and the resource limits.
2) Getting the webcam to shut off after Stop button is pressed on Streamlit to preserve user privacy.
3) Getting correct predictions in notebook but incorrect ones in our web-app for distracted drivers.
4) We were not able to solve the last issue entirely but definitely look forward to doing so in the future!

## Future Developments to SteerSecure:
1) Integrate with an interface inside your car, similar to a dashboard, for quick access to our services
2) Chatbot automatically sends messages to your loved ones alerting them if it notices regular patterns of drowsiness or Distraction
3) Car slows down and pulls over automatically when the SteerSecure app detects driver getting very drowsy
4) Car can also automatically go into self-driving mode when it detects that the user is too exhausted

## Contributors


<table align="center">
<tr align="center">


<td width:25%>
Aditya Shukla

<p align="center">
<img src = "https://avatars1.githubusercontent.com/u/20011207?s=400&u=7570f3915eca3bcd55cd72c60038e4f68965db4b&v=4"  height="120" alt="Aditya Shukla">
</p>
<p align="center">
<a href = "https://github.com/adityashukzy"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/aditya-shukla-975940188/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td width:25%>

Pooja Ravi

<p align="center">
<img src = "https://avatars3.githubusercontent.com/u/66198904?s=460&u=06bd3edde2858507e8c42569d76d61b3491243ad&v=4"  height="120" alt="Ansh Sharma">
</p>
<p align="center">
<a href = "https://github.com/01pooja10"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/pooja-ravi-9b88861b2/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

</table>

## License
MIT © Pooja Ravi

This project is licensed under the MIT License - see the [License](LICENSE) file for details

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with ❤️ by Team Armada
</p>
