<p align="center">
  <img width='650' src='https://github.com/chivington/Neural-Network-Performance-Analyzer/blob/main/imgs/mnist.png' alt='MNIST Digits'/>
</p>

# CuPy GPU-Powered Neural Network Performance Analyzer
------------------------------------------------------
This neural network uses your GPU to train on the MNIST dataset and learn to recognize images of hand-written digits.

If you don't have a GPU or don't have CuPy installed on your system, simply change the MATH_ENV variable to use NumPy instead. The training will run much slower, but it will achieve the same results.

Email john@discefasciendo.com with questions.

[Video demo](https://youtube.com/@discefasciendo/video) in my [AI Finance series](https://youtube.com/@discefasciendo/playlist)

Enjoy!

![Build Status](https://img.shields.io/badge/build-Stable-green.svg)
![License](https://img.shields.io/badge/license-NONE-green.svg)
<br/><br/><br/>

## Contents
* [Prerequisites](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#prerequisites)
* [Installation](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#installation)
* [Usage](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#usage)
* [Authors](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#authors)
* [Contributing](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#contributing)
* [Acknowledgments](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#acknowledgments)
* [License](https://github.com/chivington/Neural-Network-Performance-Analyzer/tree/master#license)
<br/>

## Prerequisites
  * Python
  * NumPy
  * CuPy (if you want to use your GPU)
  * Requests
  * Matplotlib
<br/><br/>


## Installation
First, clone this repository:
```bash
  git clone https://github.com/chivington/Neural-Network-Performance-Analyzer.git
```

Next, navigate into the directory and download the datasets:
```bash
  cd Neural-Network-Performance-Analyzer
  python mnist-nn.py get_data
```

Then, follow the "Usage" steps below.
<br/>

## Usage
Open "mnist-nn.py" and set desired options. Save the file and run it with Python in your terminal or command prompt. The program will:

1. Load & pre-process the MNIST dataset.
2. Optionally, load trained weights from a previous run.
3. Train models with the parameters you've chose.
4. Display plots of performance metrics for all models.
5. Display a random sample of images & predictions for the model with the best performance.
7. Optionally, save the performance metrics and/or best model weights.
8. End.

For more details, see the [demo](https://youtube.com/@discefasciendo)

Feel free to ask me questions on [GitHub](https://github.com/chivington)

<!-- <br/>
<p align="center">
  <img width='600' src='https://github.com/chivington/Neural-Network-Performance-Analyzer/blob/master/imgs/random-img.jpg' alt='Random Digit'/>
</p><br/>

<p align="center">
  <img width='600' src='https://github.com/chivington/Neural-Network-Performance-Analyzer/blob/master/imgs/errors-and-times.jpg' alt='Training & Validation Errors'/>
</p><br/>

<p align="center">
  <img width='600' src='https://github.com/chivington/Neural-Network-Performance-Analyzer/blob/master/imgs/classification.jpg' alt='Classification Test'/>
</p>
<br/><br/> -->


## Authors
* **Johnathan Chivington:** [GitHub](https://github.com/chivington)

## Contributing
Not currently accepting outside contributors, but feel free to use as you wish.

## License
There is no license associated with this content.
<br/><br/>
