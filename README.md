# Handwritten digit detector backed by ANN created from scratch

In this app, the user can draw on a digit on a canvas and a trained (on the spot or using cached values) ***neural network*** "tells" which digit is drawn

## Features
- *Autotraining with preesxiting data taken from the MNIST database
- A canvas that can be drawn upon
- Digit detection*

>[!IMPORTANT]
> Because the resulting data from the canvas is not a very good approximation of the trainig data (alias grayscale values are random), the detection is not that accurate in some cases.

>[!WARNING]
> As I said earlier, the neural network was created and trained from zero, with no Machine Learning libraries used, only numpy.

## Screenshots from the app
![Digit '2'](/img/2_img.png)
![Digit '3'](/img/3_img.png)
![Digit '4'](/img/4_img.png)
![Inconclussive](/img/inconclussive_img.png)

## How to run the app on Linux machines
1. Clone the repository
```
git clone git@github.com:andrei-muri/handwritten-digit-detector.git
```
2. Create a virtual environment (optional but recommended)
```
python3 -m venv my_env
source my_env\bin\activate
```
3. Install the modules
```
pip install -r requirements.txt
```
4. Run
```
cd src
python3 main.py
```

>[!NOTE]
> All the concepts regarding the creation and training of the neural network were learned from Michael Nielsen's book on neural networks and deep learning.

## Author
Created by ***Muresan Andrei***  
[GitHub](https://github.com/andrei-muri)
[LinkedIn](https://www.linkedin.com/in/andrei-muresan-muri/)
