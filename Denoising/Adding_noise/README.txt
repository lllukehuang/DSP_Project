Environment：python3.7
Use package：opencv, numpy, random, scikit-image

We provided 2 functions，sknoise() and noise()；In which sknoise() used scikit-image pack.
There are 2 parameters for those functions：imgpath and m，both are str. imgpath is the path of the image.

m is a str type parameter, which decide the type of noise to add;
In our project, we only used gaussian, salt and pepper noise.
‘0’ No noise.
‘gaussian’ Gaussian-distributed additive noise.
‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
‘poisson’ Poisson-distributed noise generated from the data.
‘salt’ Replaces random pixels with 1.
‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signedimages.
‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.