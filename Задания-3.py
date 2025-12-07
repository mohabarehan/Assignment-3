import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

# === Load BLIP for image captioning ===
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# === Load two different LLMs for comparison ===
llm1 = pipeline("text2text-generation", model="google/flan-t5-large")
llm2 = pipeline("text2text-generation", model="google/flan-t5-base")

data_folder = r"D:\planes_dataset"

def describe_image(img):
    inputs = blip_processor(images=img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def classify_with_llm(caption, llm):
    prompt = f"Classify this aircraft strictly as 'civil' or 'military'. Description: {caption}"
    result = llm(prompt, max_new_tokens=5, truncation=True)
    text = result[0].get('generated_text', '').strip().lower()
    if "civil" in text:
        return "civil"
    elif "military" in text or "fighter" in text:
        return "military"
    else:
        return "civil"  # default fallback

# === Counters for each model ===
llm1_civil = 0
llm1_military = 0
llm2_civil = 0
llm2_military = 0
total = 0

for filename in os.listdir(data_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(data_folder, filename)
        img = Image.open(img_path).convert("RGB")

        caption = describe_image(img)
        pred1 = classify_with_llm(caption, llm1)
        pred2 = classify_with_llm(caption, llm2)

        # Update counters for LLM1
        if pred1 == "civil":
            llm1_civil += 1
        else:
            llm1_military += 1

        # Update counters for LLM2
        if pred2 == "civil":
            llm2_civil += 1
        else:
            llm2_military += 1

        total += 1

        print(f"{filename} | Caption: {caption} | LLM1: {pred1} | LLM2: {pred2}")

# === Final results ===
print("\n--- Results from LLM1 ---")
print("Civil:", llm1_civil)
print("Military:", llm1_military)

print("\n--- Results from LLM2 ---")
print("Civil:", llm2_civil)
print("Military:", llm2_military)

print("\nTotal images processed:", total)

##Результаты
#- Results from LLM1 -
#Civil: 22
#Military: 8

#- Results from LLM2 -
#Civil: 14
#Military: 16

#Total images processed: 30



