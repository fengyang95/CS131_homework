import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    h, w, channel = image.shape[:3]
    out = np.zeros_like(image,np.float32)
    for i in range(h):
        for j in range(w):
            for k in range(channel):
                out[i,j,k]=0.5*image[i,j,k]*image[i,j,k]
                if out[i,j,k]<0:
                    out[i,j,k]=0
                if out[i,j,k]>255:
                    out[i,j,k]=255
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]
    gray=0.229*r+0.587*g+0.114*b
    out=gray
    ### END YOUR CODE
    
    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out=np.zeros_like(image)
    index=None
    if channel=='R':
        index=0
    elif channel=='G':
        index=1
    elif channel=='B':
        index=2
    else:
        raise ValueError("channel invalid")
    for i in range(3):
        if i!=index:
            out[:,:,i]=image[:,:,i]
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    channel_map={'L':0,'A':1,'B':2}
    out=lab[:,:,channel_map[channel]]
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    channel_map={'H':0,'S':1,'V':2}
    out=hsv[:,:,channel_map[channel]]
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    out_left=rgb_decomposition(image1,channel1)
    out_right=rgb_decomposition(image2,channel2)
    out=np.concatenate((out_left,out_right),axis=1)
    ### END YOUR CODE

    return out
