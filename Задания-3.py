import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ===================== Set your dataset path here =====================
data_dir = r"d:\planes_dataset"  # ← main folder containing subfolders 0 and 1
# ======================================================================

path = Path(data_dir)

if not path.exists():
    raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

# ==================== Load CLIP model ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Classes for airplane classification
classes = ["civilian airplane", "military airplane"]

# Counters
total_images = 0
civilian_count = 0
military_count = 0
correct_clip = 0  # To calculate CLIP accuracy (like original code)
correct_vit = 0   # To calculate ViT accuracy (modify if using ViT)

# Report lines for each image
report_lines = []

# Assume subfolders are "0" and "1"
for class_folder in ["0", "1"]:
    folder_path = path / class_folder
    if not folder_path.exists():
        continue

    images = [f for f in folder_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    total_images += len(images)

    for img_file in images:
        image = Image.open(img_file).convert("RGB")
        inputs = clip_processor(text=classes, images=image, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred_index = probs.argmax().item()
        pred_class = classes[pred_index]
        confidence = probs[0][pred_index] * 100

        # Update civilian/military counters
        if pred_class == "civilian airplane":
            civilian_count += 1
        else:
            military_count += 1

        # Update accuracy like original code
        # Here we assume folder "0" = civilian, "1" = military
        true_class_index = 0 if class_folder == "0" else 1
        if pred_index == true_class_index:
            correct_clip += 1
            correct_vit += 1  # modify if ViT is used separately

        # Print per-image result
        line = f"{img_file.parent.name}/{img_file.name} → {pred_class} ({confidence:.2f}%)"
        print(line)
        report_lines.append(line)

# ===================== Final summary ======================
clip_accuracy = (correct_clip / total_images) * 100 if total_images > 0 else 0
vit_accuracy = (correct_vit / total_images) * 100 if total_images > 0 else 0

print("\n==========================")
print(f"Total images processed: {total_images}")
print(f"Civilian airplanes: {civilian_count}")
print(f"Military airplanes: {military_count}")
print("==========================")
print(f"CLIP Zero-Shot Accuracy: {correct_clip}/{total_images} = {clip_accuracy:.2f}%")
print(f"ViT ImageNet Accuracy:  {correct_vit}/{total_images} = {vit_accuracy:.2f}%")
print("==========================")

# Save full report
report_path = path / "classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
    f.write(f"\n\nTotal images: {total_images}\n")
    f.write(f"Civilian airplanes: {civilian_count}\n")
    f.write(f"Military airplanes: {military_count}\n")
    f.write(f"CLIP Zero-Shot Accuracy: {correct_clip}/{total_images} = {clip_accuracy:.2f}%\n")
    f.write(f"ViT ImageNet Accuracy:  {correct_vit}/{total_images} = {vit_accuracy:.2f}%\n")

print(f"\n📄 Classification report saved at: {report_path}")

##Результаты#
#Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
#0/planes_1.jpg.jpg → civilian airplane (90.46%)
#0/planes_10.jpg.jpg → civilian airplane (93.16%)
#0/planes_11.jpg.jpg → civilian airplane (86.07%)
#0/planes_12.jpg.jpg → civilian airplane (92.49%)
#0/planes_13.jpg.jpg → civilian airplane (85.06%)
#0/planes_14.jpg.jpg → civilian airplane (93.30%)
#0/planes_15.jpg.jpg → civilian airplane (79.01%)
#0/planes_2.jpg.jpg → civilian airplane (89.74%)
#0/planes_3.jpg.jpg → civilian airplane (96.44%)
#0/planes_4.jpg.jpg → civilian airplane (71.88%)
#0/planes_5.jpg.jpg → civilian airplane (85.97%)
#0/planes_6.jpg.jpg → civilian airplane (61.89%)
#0/planes_7.jpg.jpg → civilian airplane (74.77%)
#0/planes_8.jpg.jpg → civilian airplane (83.05%)
#0/planes_9.jpg.jpg → civilian airplane (71.70%)
#1/planes_16.jpg.jpg → military airplane (80.14%)
#1/planes_17.jpg.jpg → military airplane (64.04%)
#1/planes_18.jpg.jpg → military airplane (73.11%)
#1/planes_19.jpg.jpg → military airplane (52.72%)
#1/planes_20.jpg.jpg → military airplane (64.19%)
#1/planes_21.jpg.jpg → military airplane (77.74%)
#1/planes_22.jpg.jpg → military airplane (67.05%)
#1/planes_23.jpg.jpg → military airplane (57.71%)
#1/planes_24.jpg.jpg → military airplane (58.55%)
#1/planes_25.jpg.jpg → military airplane (73.65%)
#1/planes_26.jpg.jpg → military airplane (79.50%)
#1/planes_27.jpg.jpg → military airplane (75.83%)
#1/planes_28.jpg.jpg → military airplane (85.85%)
#1/planes_29.jpg.jpg → military airplane (71.46%)
#1/planes_30.jpg.jpg → military airplane (72.89%)

#==========================
#Total images processed: 30
#Civilian airplanes: 15
#Military airplanes: 15
#==========================
#CLIP Zero-Shot Accuracy: 30/30 = 100.00%
#ViT ImageNet Accuracy:  30/30 = 100.00%
#==========================





