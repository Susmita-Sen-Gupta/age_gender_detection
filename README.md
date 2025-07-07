#real_time_age_gender_detection

**Age and Gender Detection using OpenCV DNN**


This project performs real-time age and gender detection using OpenCV's Deep Neural Network (DNN) module and pre-trained Caffe models. It captures video from your webcam, detects faces, and predicts the age group and gender of the detected person.

 Features::


 
Real-time face detection from webcam


Age prediction using a pre-trained Caffe model


Gender prediction using a pre-trained Caffe model


Bounding boxes and labels displayed on the video feed


Requirements::





Python 3.6+

OpenCV (preferably 4.x)



 Models Used::

 
Face Detection: opencv_face_detector.pb & opencv_face_detector.pbtxt

Age Detection: age_net.caffemodel & age_deploy.prototxt

Gender Detection: gender_net.caffemodel & gender_deploy.prototxt

All models are based on Caffe and are publicly available through OpenCV's model zoo.


 
 
 Sample Output::



 
Real-time webcam feed with age and gender overlay.
![image](https://github.com/user-attachments/assets/03d5d316-b5ac-4fc1-b8da-a0d0cd28f6ab)





