import cv2

def load_images():

    image1 = cv2.imread("input/image1.png")
    image2 = cv2.imread("input/image2.png")

    if image1 is None:
        raise FileNotFoundError("image1.png not found")

    if image2 is None:
        raise FileNotFoundError("image2.png not found")

    return image1, image2