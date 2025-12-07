import os
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import ResNetForImageClassification, AutoImageProcessor
import torch

data_folder = r"D:\planes_dataset"

labels = ["civil", "military"]

vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
id2label_vit = vit_model.config.id2label

resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
id2label_resnet = resnet_model.config.id2label

def map_label(original_label):
    label_lower = original_label.lower()
    if "warplane" in label_lower or "military" in label_lower or "fighter" in label_lower:
        return "military"
    elif "airliner" in label_lower or "civil" in label_lower or "passenger" in label_lower:
        return "civil"
    else:
        return "civil"

def predict_vit(img):
    inputs = vit_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        probs = outputs.logits.softmax(1)
    pred_id = int(probs.argmax())
    original_label = id2label_vit[pred_id]
    return map_label(original_label)

def predict_resnet(img):
    inputs = resnet_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = resnet_model(**inputs)
        probs = outputs.logits.softmax(1)
    pred_id = int(probs.argmax())
    original_label = id2label_resnet[pred_id]
    return map_label(original_label)

vit_correct = 0
resnet_correct = 0
total = 0

for filename in os.listdir(data_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(data_folder, filename)
        img = Image.open(img_path).convert("RGB")

        true_label = "military" if "military" in filename.lower() else "civil"

        vit_pred = predict_vit(img)
        resnet_pred = predict_resnet(img)

        if vit_pred == true_label:
            vit_correct += 1
        if resnet_pred == true_label:
            resnet_correct += 1

        total += 1

vit_acc = vit_correct / total
resnet_acc = resnet_correct / total

print("Number of images:", total)
print("ViT Accuracy:", vit_acc)
print("ResNet Accuracy:", resnet_acc)
