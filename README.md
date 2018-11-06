# Letter Recognition (Machine Learning)
Letter recognition using fast Fourier transform. This algorithm will recognise the characters S, T and V.

### S,T,V
The following image shows the 30 hand drawn characters used to train the letter recognition algorithm.
![Characters](images/letters.png)

### FFT
A fast Fourier transform is applied to the characters to get their frequency representation. The images below shows this representation (frequency against angle) - there are clearly different patterns created for each letter.

![FFT](images/fft.png)

### Masks
Three different masks are created, each mask filters for the patterns found above. These masks can be found below.

![](images/mask.png)

### Nearest Neighbour

After masking out all frequencies that do not correspond to the patterns, the data can be separated into three groups using a nearest neighbour algorithm. This separation can be found in the image below.

<img src="images/indexV.png" width="400">

### NN Test

New characters were created to test the above algorithm. The images below show the letters created, as well as their classification. In addition to these letters 


<img src="images/testChars.png" width="800">
<img src="images/test1.png" width="400">

### Probability Density

An alternate clustering, using a probablity density function, can also be found in the code.

## Requirements

* Python2.7
* Numpy
* Scipy
* Skimage
* heapq
* Matplotlib

* Jupyter notebook (Optional)

## Usage

Simply open letter.ipynb in jupyter notebook.
