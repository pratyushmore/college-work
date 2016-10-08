from cv2 import *
import numpy as np
from scipy.integrate import dblquad
from math import pi, exp

IMAGE_NAME = "testing_four.jpg"

SIGMA = 8
SIDE_LENGTH = 30

def haar_edge_x():
	haar_x = edge_x_kernel(20)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img,haar_x))
	imshow('image', processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_haar_edge_x.png', processed_image)

def haar_edge_y():
	haar_y = edge_y_kernel(20)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img,haar_y))
	imshow('image', processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_haar_edge_y.png', processed_image)

def haar_line_x():
	haar_x = line_x_kernel(20)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img, haar_x))
	imshow('image', processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_haar_line_x.png', processed_image)

def haar_line_y():
	haar_y = line_y_kernel(20)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img, haar_y))
	imshow('image', processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_haar_line_x.png', processed_image)

def haar_four_rectangle():
	haar_kernel = four_rectangle_kernel(20)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img, haar_kernel))
	imshow('image', processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_haar_four.png', processed_image)

def sobel_edge():
	sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	img = imread(IMAGE_NAME, 0)
	gx = convolve(img, sobel_x)
	gy = convolve(img, sobel_y)
	processed_image = normalise(np.add(np.absolute(gx), np.absolute(gy)))
	imshow('image',processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_sobel.png', processed_image)

def gaussian_blur():
	kernel = gaussian_kernel(SIGMA, SIDE_LENGTH)
	img = imread(IMAGE_NAME, 0)
	processed_image = normalise(convolve(img, kernel))
	imshow('image',processed_image)
	waitKey(0)
	destroyAllWindows()
	imwrite(IMAGE_NAME[:len(IMAGE_NAME) - 4] + '_gaussian.png', processed_image)

def edge_x_kernel(x):
	kernel_x = np.full((2*x,x), -1)
	kernel_x[0:x,:] = 1
	return kernel_x

def edge_y_kernel(x):
	kernel_y = np.full((x,2*x), -1)
	kernel_y[:,0:x] = 1
	return kernel_y

def line_x_kernel(x):
	kernel_x = np.full((2*x,x), 1)
	kernel_x[x/2:3*x/2,:] = -1
	return kernel_x

def line_y_kernel(x):
	kernel_y = np.full((x, 2*x), 1)
	kernel_y[:,x/2:3*x/2] = -1
	return kernel_y

def four_rectangle_kernel(x):
	kernel = np.full((x,x), 1)
	kernel[x/2:x, 0:x/2] = -1
	kernel[0:x/2, x/2:x] = -1
	return kernel

def gaussian_kernel(sigma, side_length):
	def function(x, y):
		return (1/(2*pi*(sigma**2)))*exp(-(x**2 + y**2)/(2*(sigma**2)))
	xmin = -3*sigma
	xmax = 3*sigma
	step = float(xmax - xmin)/side_length
	limits = np.arange(xmin, xmax, step)
	kernel_temp = [[dblquad(function, x, x + step, lambda c: y, lambda c:y + step) for x in limits] for y in limits]
	kernel = [map(lambda (x, y): x, seq) for seq in kernel_temp]
	return kernel

def normalise(image):
	print "HERE: ", np.amin(image)
	minv = np.amin(image)
	image = np.add(image, abs(minv))
	scale = np.amax(image)
	scale = scale/255.
	scaled_image = np.array(np.divide(image, scale), dtype='uint8')
	return scaled_image

def convolve(image, kernel):
	height, width = image.shape
	conv_img = np.zeros((height + 1 - len(kernel) , width + 1 - len(kernel[0])), dtype='int32')
	for i in range(0, len(conv_img)):
		for j in range(0, len(conv_img[0])):
			conv_img[i,j] = np.sum(np.multiply(image[i:i+len(kernel), j:j+len(kernel[0])], kernel))
	return conv_img

haar_four_rectangle()
haar_line_x()
haar_line_y()
haar_edge_x()
haar_edge_y()
sobel_edge()
#gaussian_blur()
