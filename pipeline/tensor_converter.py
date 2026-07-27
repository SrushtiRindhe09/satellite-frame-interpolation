from torchvision import transforms

def convert_to_tensor(image1, image2):

    transform = transforms.ToTensor()

    tensor1 = transform(image1)
    tensor2 = transform(image2)

    # Add batch dimension
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    return tensor1, tensor2