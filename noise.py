from skimage.util import random_noise

def add_noise_additive(image, var=0.01):
    return random_noise(image, mode='gaussian', var=var)

def add_noise_salt_pepper(image, amount=0.05):
    return random_noise(image, mode='s&p', amount=amount)

def add_noise_multiplicative(image, var=0.01):
    return random_noise(image, mode='speckle', var=var)
