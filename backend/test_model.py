import torch
from torchvision.models import efficientnet_v2_s

MODEL_PATH = r".\models\efficientnet_v2_s_best.pth"

classes = ["Aphids", "Leaf blight", "Leaf miner", "TMB"]

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = efficientnet_v2_s(weights=None)

# Replace the final classifier for our 4 classes
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    4
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("================================")
print("MODEL LOADED SUCCESSFULLY")
print("Model      :", checkpoint["model_name"])
print("Classes    :", classes)
print("Image size :", checkpoint["image_size"])
print("Val Acc    :", checkpoint["best_val_accuracy"])
print("================================")