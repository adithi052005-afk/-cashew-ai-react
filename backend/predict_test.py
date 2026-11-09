import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = r".\models\efficientnet_v2_s_best.pth"

classes = ["Aphids", "Leaf blight", "Leaf miner", "TMB"]

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

# Create model
model = models.efficientnet_v2_s(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# CHANGE THIS PATH to your test image
IMAGE_PATH = r"C:\Users\SUNIL KUMAR\Downloads\1000127668.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)

confidence, predicted = torch.max(probabilities, 1)

print("\n==============================")
print("PREDICTION RESULT")
print("==============================")
print("Predicted class :", classes[predicted.item()])
print("Confidence      :", f"{confidence.item() * 100:.2f}%")
print("==============================")

print("\nAll class probabilities:")
for class_name, probability in zip(classes, probabilities[0]):
    print(f"{class_name:15s}: {probability.item() * 100:.2f}%")