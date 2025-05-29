# 🧠 Image Captioning and Mood Detection API

This FastAPI-based RESTful service accepts base64-encoded images and returns:
- A concise caption generated using a vision-language model (BLIP)
- An inferred mood based on the caption, analyzed by LLaMA3 (via Ollama)
- A fallback mood estimate using image color composition if LLM inference fails

Deployed on a **Google Cloud VM**, this backend API is ready for use by frontend applications.

---

## 🚀 Features

- 📸 **BLIP Captioning**: Generates image captions using `Salesforce/blip-image-captioning-base`.
- 🧠 **Mood Inference via LLaMA3**: Analyzes caption semantics using `LLaMA3` via `Ollama`.
- 🎨 **Color-Based Fallback**: If LLM-based mood detection fails, RGB-based heuristic fallback is used.
- 🧠 **spaCy Filtering**: Extracts key tokens from captions for a meaningful summary.
- ☁️ **Deployed on Google Cloud VM** with Uvicorn

---

## 📦 API Endpoint

### `POST /analyze/`

**Request Body**:
```json
{
  "image_base64": "<Base64-encoded image string>"
}
```
**Response**:
```json
{
  "caption": "cat on a couch",
  "mood": "calm"
}
```

## ⚙️ Tech Stack
**Backend Framework**: FastAPI
**Machine Learning Models**:
- Salesforce/blip-image-captioning-base (image captioning via Hugging Face Transformers)
- LLaMA3 via Ollama (text-based mood inference)
**NLP Toolkit**: spaCy (en_core_web_sm)
**Image Processing**: Pillow, NumPy
**Optional Storage**: MongoDB (integration available but commented out)
**Deployment**: Google Cloud VM (Linux, Uvicorn)

---

## 🌐 Frontend Integration Example

Frontend developers can send a base64-encoded image string via a POST request and receive the caption and mood:
```
const response = await fetch("http://<your-vm-ip>:8000/analyze/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image_base64: base64ImageString
  }),
});

const result = await response.json();
console.log(result.caption); // e.g., "cat on a couch"
console.log(result.mood);    // e.g., "calm"
```
- 🔐 Make sure CORS is enabled if accessing from the browser and IP is allowed through VM firewall settings.

## ☁️ Google Cloud VM Deployment Notes
To deploy or restart the API on your Google Cloud VM:
```
# SSH into your VM instance
gcloud compute ssh <your-vm-name> --zone <your-zone>

# Navigate to your app directory
cd /path/to/app

# Run the server
python3 app.py
```
- ⚠️ Open the API Port: Ensure your VM allows traffic on port 8000 (or reverse proxy to port 80 or 443 using Nginx).

## 💻 Local Development
Clone the repository and set up the environment locally:
1. Install dependencies
```
pip install fastapi uvicorn pillow numpy torch transformers spacy pymongo
python -m spacy download en_core_web_sm
```
2. (Optional) Warm up Ollama locally
```
ollama run llama3
```
3. Start FastAPI server
```
python app.py
```
By default, this runs on http://localhost:8000/analyze/
