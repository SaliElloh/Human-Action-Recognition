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

### Built With

The frameworks and libraries used within this project are:

* [![TensorFlow][Tensorflow.js]][Tensorflow-url]
* [![Keras][Keras.js]][Keras-url]
* [![NumPy][NumPy.js]][NumPy-url]
* [![Matplotlib][Matplotlib.js]][Matplotlib-url]
* [![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-yellowgreen?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)


<!-- Dataset -->


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites


### Steps to run the code:

   
<!-- NEW LABELS IMAGE EXAMPLES -->
## Image examples

Here is a snapshot of how the images for the new labels, after they've been preprocessed, look like:

![image](https://user-images.githubusercontent.com/112829375/235897301-c6795324-af8c-4a87-a9ea-fc7425dda553.png)

![image](https://user-images.githubusercontent.com/112829375/235897529-89753437-1842-477f-b3a4-319e2e482a8c.png)

![image](https://user-images.githubusercontent.com/112829375/235896781-38f344dc-1cea-4d6f-8505-53e09542459f.png)

<!-- LICENSE -->
## License

No License used.

<!-- CONTACT -->
## Contact

Sali E-loh - [@Sali El-loh](https://www.linkedin.com/in/salielloh12/) - ellohsali@gmail.com


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was inspired by the TensorFlow, "Basic classification: Classify images of clothing" tutorial.

* [TensoFlow: Clothes Image Classification Tutorial](https://www.tensorflow.org/tutorials/keras/classification)

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

 arab-clothes-image-classifcation
This project expands on the TensorFlow tutorial "Basic classification: Classify images of clothing" by training the model to recognize three new labels that represent Muslim/Arab clothing.

For more information about me, please visit my LinkedIn:

[![LinkedIn][LinkedIn.js]][LinkedIn-url]


<div align="center">
  <h3 align="center">A brief Read.me introducing the project and its contents</h3>
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

The "Basic classification" Clasify images of clothing" Tensorflow tutorial trains a neural network model to classify images of clothing, like sneaker and shirts. The guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories.

For this project, I've create a new dataset that includes additional labels not present in the current algorithm. Specifically, I've added three labels - abaya, thobe, and hijab - which are types of clothing commonly associated with Islamic and Arabic cultures.

After generating the new dataset with the additional labels, I've retrained the model using this updated data. I then evaluated the performance of the retrained model to assess its accuracy and effectiveness in classifying the new labels.

### What are Hijabs, Abayas, and Thobes? 

Hijab generally refers to headcoverings worn by some Muslim women. It is similar to the wimple, apostolnik, and mantilla worn by some Christian women.

The abaya is a simple, loose over-garment, essentially a robe-like dress, worn by some women in parts of the Muslim world including North Africa, the Arabian Peninsula, and most of the Middle East.

The Thobe, also written as "Thawb", is an ankle-length robe, usually with long sleeves. It is commonly worn in the Arabian Peninsula, the Middle East, North Africa, and other neighbouring Arab countries, and some countries in East and West Africa.

Heres a demonstration of each clothing garment, along with their classification:
![image](https://user-images.githubusercontent.com/112829375/235896519-90dc82e1-8552-4081-9eed-20241fad5ff0.png)

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

* [![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-yellowgreen?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)

<!-- Dataset -->
### Dataset


<!-- GETTING STARTED -->
## Getting Started

To get the project running, there's a couple of programs and steps needed. here are the steps: 

### Prerequisites

If using Google colab to test the project, you need:

1. a Google colab account
2. Access to a GPU
3. Internet/Wi-Fi

If you plan on running it on python, you need to install on your computer the following:

1. Python
2. pip 
3. PyCharm 
4. Tensorflow, Keras, NumPy, Matplotlib

### Steps to run the code:


<!-- NEW LABELS IMAGE EXAMPLES -->
## Image examples


<!-- LICENSE -->
## License

No License used.

<!-- CONTACT -->
## Contact

Sali E-loh - [@Sali El-loh](https://www.linkedin.com/in/salielloh12/) - ellohsali@gmail.com


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


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




