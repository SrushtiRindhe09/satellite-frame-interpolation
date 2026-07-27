def validate_images(image1, image2):

    # Check if images have same dimensions
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image sizes do not match!\n"
            f"Image 1: {image1.shape}\n"
            f"Image 2: {image2.shape}"
        )

    print("✅ Validation Successful")
    print("Both images have the same dimensions.")