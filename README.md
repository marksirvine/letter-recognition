# Letter recognition
Letter recognition using fast fourier transform. This algorithm will recognise the characters S, T and V.

![Characters](images/letters.png)

A fast fourier transform is appleid to the characters to get their frequency representation. The images bellow shows this representation (frequency against angle) - there are clearly different patterns created for each letter.

![FFT](images/fft.png)

Three different masks are created, each mask filters for the patterns found above. These masks can be found below.

![](images/mask.png)

After masking out all frequencies that do not correspond to the patterns, the data can be separated into three groups using a nearest neighbour algorithm. This separation can be found in the image below.

![](images/indexV.png)


## Requirements 

* Python 
* Numpy
* Scipy
* Skimage
* heapq 

* Jupyter notebook (Optional)
