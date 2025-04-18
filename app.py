# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import base64
import torch
import numpy as np
from pymongo import MongoClient
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
import ollama

# ----------------- Load Models -----------------
nlp = spacy.load("en_core_web_sm")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

valid_moods = ["happy", "sad", "angry", "neutral", "surprised", "confused", "excited", "calm"]

# ----------------- FastAPI Setup -----------------
app = FastAPI()

# ----------------- Mongo Setup (Optional) -----------------
# client = MongoClient("mongodb+srv://dev:CkVPtXuiNweaYZ8t@doodle-dj.c3vk0.mongodb.net/doodle-dj-db")
# db = client["doodle-dj-db"]
# users = db["users"]

# ----------------- Core Logic -----------------
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert('RGB')

def infer_mood_from_color(image):
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    r, g, b = np.mean(image_np, axis=(0, 1))
    if r > g and r > b:
        return "happy"
    elif g > r and g > b:
        return "calm"
    elif b > r and b > g:
        return "neutral"
    else:
        return "neutral"

def generate_caption_from_pil(image):
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def detect_mood_from_caption(caption, image):
    prompt = f"""
    You are a mood detection model. Based on the following caption, classify the mood of the image.
    Caption: "{caption}"
    Your output should only be one of these moods: {', '.join(valid_moods)}.
    """
    try:
        response = ollama.chat(model="llama3:latest", messages=[{"role": "user", "content": prompt}])
        mood_text = response.message.content.strip()
        for mood in valid_moods:
            if mood.lower() in mood_text.lower():
                return mood
        return infer_mood_from_color(image)
    except:
        return "neutral"

# ----------------- API Route -----------------
class ImageRequest(BaseModel):
    # user_id: str
    image_base64: str
    # image_name: str

@app.post("/analyze/")
def analyze_image(req: ImageRequest):
    try:
        # Decode and process image
        image = base64_to_image(req.image_base64)
        caption = generate_caption_from_pil(image)
        mood = detect_mood_from_caption(caption, image)

        # # Store in MongoDB
        # record = {
        #     "user_id": req.user_id,
        #     "image_name": req.image_name,
        #     "image": req.image_base64,
        #     "caption": caption,
        #     "mood": mood
        # }
        # users.insert_one(record)

        return {"caption": caption, "mood": mood}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
