## DSP Project: Image Frequency Domain Analysis and Denoising ##

### Requirement of project ###

#### 1.Image frequency domain conversion

- Calculate and draw the center frequency spectrum of the image
- Process image in frequency domain with Gaussian low pass filter and Gaussian high pass filter respectively
- Use the frequency domain Laplace operator to sharpen the image

#### 2.Add noise to the image

- Add at least two kinds of noise to the image, such as Gaussian noise, Salt-and-pepper noise, periodic noise, etc., or other noise

#### 3.Image denoising and result analysis

- Gaussian template, spatial domain method of median filtering and frequency filtering method of Fourier transform and wavelet transform are respectively used to de-noising images. Based on PSNR (peak signal-to-noise ratio) value and visual effect, the denoising ability of these four denoising methods to different types of noise added before should be compared.

## Our code file structure

### Frequency Domain Analysis of Images

For the original image, we have made a series of conversion and analysis in the frequency domain, which can help us understand and grasp the image's frequency domain characteristics and frequency domain filtering properties more intuitively.

All test files and code are located at [Frequency Transform](./Frequency%20Transform)

### Images Denoising

In [Denoising](./Denoising) folder we tried different denoising methods in different directions. 

We first systematically added different kinds of noise to the original test image through [CreateNoiseImg.py](./Denoising/CreateNoiseImg.py), then test a quantity of denoising methods in space domain and frequency domain. Some of the results are shown in [Imgs](./Denoising/Imgs) folder. More results and analysis are shown in our report paper. 

### On-line inspection

You can clone this code base and change your files in [Imgs](./Denoising/Imgs) folder. Then you can test different denoising method on your own images.