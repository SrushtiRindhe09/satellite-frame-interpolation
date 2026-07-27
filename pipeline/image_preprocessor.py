import cv2

def preprocess_images(image1, image2):

    TARGET_SIZE = (512, 512)

    image1 = cv2.resize(image1, TARGET_SIZE)
    image2 = cv2.resize(image2, TARGET_SIZE)

    return image1, image2