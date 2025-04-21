from transformers import AutoProcessor, Blip2ForConditionalGeneration

MODEL_NAME = "Salesforce/blip2-opt-2.7b"
SAVE_DIR = "./models/blip2-opt-2.7b"

# Download and cache the processor and model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.save_pretrained(SAVE_DIR)

model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_DIR)

print("Model and processor saved to", SAVE_DIR)
