from pipeline.image_loader import load_images
from pipeline.image_preprocessor import preprocess_images
from pipeline.validator import validate_images
from pipeline.tensor_converter import convert_to_tensor
from models.raft_model import load_raft_model
from pipeline.flow_visualizer import visualize_flow

# Step 1
image1, image2 = load_images()

print("Original Shapes")
print(image1.shape)
print(image2.shape)

# Step 2
image1, image2 = preprocess_images(image1, image2)

print("\nAfter Preprocessing")
print(image1.shape)
print(image2.shape)

# Step 3
validate_images(image1, image2)

# Step 4
tensor1, tensor2 = convert_to_tensor(image1, image2)

print("\nTensor Shapes")
print(tensor1.shape)
print(tensor2.shape)

print("\nTensor Data Type")
print(tensor1.dtype)
print(tensor2.dtype)

print("\nLoading AI Model...")

raft_model = load_raft_model()

from models.raft_model import (
    load_raft_model,
    predict_optical_flow
)
print("\nRunning RAFT...")

flow_predictions = predict_optical_flow(
    raft_model,
    tensor1,
    tensor2
)

print("RAFT Finished!")

print("\nType of Output:")
print(type(flow_predictions))

print("\nNumber of Predictions:")
print(len(flow_predictions))

print("\nFinal Flow Shape:")
print(flow_predictions[-1].shape)
print(flow_predictions[-1].shape)
print("\nGenerating Optical Flow Image...")

visualize_flow(flow_predictions[-1])