Human-Action-Recognition

Final Project for my ECE 5831 (Pattern Recognition and Neural Networks) class.

For more information about me, please visit my LinkedIn:

[![LinkedIn][LinkedIn.js]][LinkedIn-url]


<div align="center">
  <h3 align="center">A brief Read.me introducing the project and its contents</h3>
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

CImplemented ConvLSTM and LRCN models for human action recognition using TensorFlow and Keras. and Integrated pose prediction with action recognition using OpenCV and MediaPipe.

###  Data Preprocessing:

* UCF50 Dataset is used
* Frames are extracted from videos using OpenCV, resized to 64x64 pixels, and normalized.
* 20 frames are sampled from each video, organized as sequences with corresponding one-hot encoded labels.

### ConvLSTM Model:

* A ConvLSTM model is built using TensorFlow and Keras, with layers including ConvLSTM, MaxPooling3D, Dropout, Flatten, and Dense.
* The model uses key hyperparameters like varying filter sizes, (3, 3) kernels, tanh activation, and recurrent dropout to prevent overfitting.
*MaxPooling3D and TimeDistributed dropout are used for spatial down-sampling and enhancing robustness.

### LRCN Model:

* An LRCN model is constructed with TimeDistributed 2D convolutional layers, followed by MaxPooling, Dropout, LSTM, and a Dense layer for classification.
* The model is optimized for video classification, using 'relu' activation in Conv2D, MaxPooling2D for reducing spatial dimensions, and Dropout to prevent overfitting.

### Results:

## Comparitive Analysis:

![image](https://github.com/user-attachments/assets/42e32dfe-f03a-4950-b1cf-8edadd22ff12)


### Built With

The frameworks and libraries used within this project are:

[![Pafy][Pafy.js]][Pafy-url]
[![OS][OS.js]][OS-url]
[![OpenCV][OpenCV.js]][OpenCV-url]
[![Math][Math.js]][Math-url]
[![Random][Random.js]][Random-url]
[![NumPy][NumPy.js]][NumPy-url]
[![Datetime][Datetime.js]][Datetime-url]
[![TensorFlow][TensorFlow.js]][TensorFlow-url]
[![Deque][Deque.js]][Deque-url]
[![Matplotlib][Matplotlib.js]][Matplotlib-url]
[![Keras][Keras.js]][Keras-url]
[![MoviePy][MoviePy.js]][MoviePy-url]
[![Scikit-learn][Scikit-learn.js]][Scikit-learn-url]


<!-- Dataset -->

UCF50 is an action recognition data set with 50 action categories, consisting of realistic videos taken from youtube. This data set is an extension of YouTube Action data set (UCF11) which has 11 action categories. The dataset could be accessed here: https://www.crcv.ucf.edu/data/UCF50.php

More information:
* Number of Classes: 50 different action categories.
*  Number of Videos: 6,618 video clips.
*  Categories: Includes activities like "Basketball", "Biking", "Diving", "PushUps", "Skateboarding", etc.
*  Data Source: Videos collected from YouTube.
*  Video Format: .avi format with varying resolutions and durations.


<!-- GETTING STARTED -->
## Getting Started



### Prerequisites


### Steps to run the code:

1. Download files "human_action_recognition_and_pose_detection.ipynb" and "human_action_recognition.ipynb"
2. Insure python and Jupyter Notebook are installed. Alteratively, you can run using Google Colab
3. Insure necessary libraries and frameworks are downloadeed
   
<!-- NEW LABELS IMAGE EXAMPLES -->
## Image examples




<!-- LICENSE -->
## License

No License used.

<!-- CONTACT -->
## Contact

Sali E-loh - [@Sali El-loh](https://www.linkedin.com/in/salielloh12/) - ellohsali@gmail.com


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[LinkedIn.js]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[LinkedIn-url]: https://www.linkedin.com/in/salielloh12/
[Tensorflow.js]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Keras.js]: https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
[NumPy.js]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib.js]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/

[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[Pafy.js]: https://img.shields.io/badge/Pafy-FF6600?style=for-the-badge
[Pafy-url]: https://github.com/mps-youtube/pafy

[OS.js]: https://img.shields.io/badge/OS-44a833?style=for-the-badge
[OS-url]: https://docs.python.org/3/library/os.html

[OpenCV.js]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/

[Math.js]: https://img.shields.io/badge/Math-000000?style=for-the-badge
[Math-url]: https://docs.python.org/3/library/math.html

[Random.js]: https://img.shields.io/badge/Random-44a833?style=for-the-badge
[Random-url]: https://docs.python.org/3/library/random.html

[NumPy.js]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Datetime.js]: https://img.shields.io/badge/Datetime-44a833?style=for-the-badge
[Datetime-url]: https://docs.python.org/3/library/datetime.html

[TensorFlow.js]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/

[Deque.js]: https://img.shields.io/badge/Deque-44a833?style=for-the-badge
[Deque-url]: https://docs.python.org/3/library/collections.html#collections.deque

[Matplotlib.js]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/

[Keras.js]: https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/

[MoviePy.js]: https://img.shields.io/badge/MoviePy-FF4500?style=for-the-badge
[MoviePy-url]: https://zulko.github.io/moviepy/

[Scikit-learn.js]: https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit-learn-url]: https://scikit-learn.org/




