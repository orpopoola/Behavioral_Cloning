import cv2
import numpy as np
import os
from sklearn.utils import shuffle
import scipy.misc
from scipy.stats import bernoulli

# This script gets samples and batch_sizes from model.py
# the samples are lines from the excel sheet
# The generator is used to return processed image batches  
correction = 0.3
path = "data/"
## Functions used in processing images

def crop(image, top_crop, bottom_crop):
    top = int(np.ceil(image.shape[0] * top_crop))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_crop))
    return image[top:bottom, :]

def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)

def random_shear(image,steering,shear_range=200):
    #source https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    return image,steering

def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def brightness(image):
    #source https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
    
def random_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle

def process_image(image, steering_angle):
    head = bernoulli.rvs(0.9)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)
        image, steering_angle = random_shear(image, steering_angle)
    image = crop(image, 0.35, 0.1)
    image, steering_angle = random_flip(image, steering_angle)
    image = random_gamma(image)
    image = resize(image, (160,320))
    return image,steering_angle



## Read in and generate data

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        #this loop gets batch_size number of samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction 
                steering_right = steering_center - correction 
                #print(path+batch_sample[0].split('\\')[-1].lstrip())
                img_center = cv2.imread(path+batch_sample[0].split('\\')[-1].lstrip())
                img_left = cv2.imread(path+batch_sample[1].split('\\')[-1].lstrip())
                img_right = cv2.imread(path+batch_sample[2].split('\\')[-1].lstrip())
                
                prep_img_center, prep_steering_center = process_image(img_center, steering_center)
                prep_img_left, prep_steering_left = process_image(img_left, steering_left)
                prep_img_right, prep_steering_right = process_image(img_right, steering_right)
                car_images.extend((prep_img_center, prep_img_left, prep_img_right))
                steering_angles.extend([prep_steering_center, prep_steering_left, prep_steering_right]) 

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train) #this performs matched shuffling
            
