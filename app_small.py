# app.py hello world
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
import os

# ----------------- Load Models -----------------
nlp = spacy.load("en_core_web_sm")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

BLOCKED_WORDS = set(["drawing", "painting", "image", "photo", "picture", "of", "sketch", "artwork", "a", "the", "is", "on", "in"])

#valid_moods = ["happy", "sad", "angry", "neutral", "surprised", "confused", "excited", "calm"]

# ----------------- FastAPI Setup -----------------
app = FastAPI()

# ----------------- Mongo Setup (Optional) -----------------
# client = MongoClient("mongodb+srv://dev:CkVPtXuiNweaYZ8t@doodle-dj.c3vk0.mongodb.net/doodle-dj-db")
# db = client["doodle-dj-db"]
# users = db["users"]

# ----------------- Warmup Functions -----------------
def warmup_model():
    dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    inputs = processor(dummy_image, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)
    print("BLIP Model Warmed Up")

def warmup_ollama():
    try:
        _ = ollama.chat(model="llama3:latest", messages=[{"role": "user", "content": "Hello"}])
        print("Ollama Model Warmed Up")
    except Exception as e:
        print(f"Ollama Warmup Failed: {e}")

# ----------------- Core Logic -----------------
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert('RGB')

def infer_mood_from_color(image):
    print("Inferring mood from color...")
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
    #inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    full_caption = processor.decode(out[0], skip_special_tokens=True).strip()
    print(f"Generated full caption: {full_caption}")
    doc = nlp(full_caption)
    important_tokens = [token for token in doc if not token.is_stop and token.is_alpha and token.text.lower() not in BLOCKED_WORDS]
    if not important_tokens:
        print("Warning: No important words left after filtering. Using fallback word.")
        short_caption = "fallback"
    else:
        # Prefer nouns or adjectives if available
        ranked_tokens = [token for token in important_tokens if token.pos_ in ["NOUN", "ADJ"]]

        if ranked_tokens:
            selected_token = ranked_tokens[-1]
        else:
            selected_token = important_tokens[-1]

        short_caption = selected_token.text.lower()
    print(f"Generated short caption: {short_caption}")
    return full_caption, short_caption

def detect_mood_from_caption(caption, image):
    prompt = f"""
    You are a mood detection model. Based on the following caption, classify the mood of the image.
    Caption: "{caption}"
    Your output should only be one to two words.
    """
    
    try:
        response = ollama.chat(model="llama3:latest", messages=[{"role": "user", "content": prompt}])
        mood_text = response.message.content.strip().lower()
        print(f"Response from model: {mood_text}")
        print("Length of mood text:", len(mood_text.split()))
        # Remove any non-alphabetic characters from mood_text
        mood_array = [word for word in mood_text.split() if word.isalpha()]
        mood_text = ' '.join(mood_array)
        print("Filtered mood text:", mood_text)
        # If mood text is empty or contains only non-alphabetic characters, fallback to color analysis
        if len(mood_text) == 0:
            print("Mood text is empty or contains only non-alphabetic characters, falling back to color analysis.")
            return infer_mood_from_color(image)
        # If mood text is more than 2 words return the first two words
        elif len(mood_text.split()) == 1:
            return mood_text
        elif len(mood_text.split()) == 2:
            return ' '.join(mood_text.split()[:2])
        # If mood text is more than 2 words, return the first two words
        elif len(mood_text.split()) > 2:
            return ' '.join(mood_text.split()[:2])
        else:
            print("Mood text is not valid, falling back to color analysis.")
            return infer_mood_from_color(image)
    except Exception as e:
        print(f"Error in mood detection model: {e}")
        print("Error in mood detection model, falling back to color analysis.")
        return 'neutral'

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
        full_caption, small_caption = generate_caption_from_pil(image)
        mood = detect_mood_from_caption(full_caption, image)

        # # Store in MongoDB
        # record = {
        #     "user_id": req.user_id,
        #     "image_name": req.image_name,
        #     "image": req.image_base64,
        #     "caption": caption,
        #     "mood": mood
        # }
        # users.insert_one(record)

        return {"caption": small_caption, "mood": mood}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # warmup_model()
    # warmup_ollama()
    port = int(os.environ.get("PORT", 8000))  # GCR injects PORT=8080
    uvicorn.run("app_small:app", host="0.0.0.0", port=port)