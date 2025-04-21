import requests
import base64

# Load an image and encode as base64
def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Test payload
url = "http://localhost:8080/analyze/"  # Change to your actual endpoint
#"https://mood-api-37578726372.us-west2.run.app/analyze/"
payload = {
    "image_base64": encode_image("./sadface.png"),  # Replace with your test image
}
# Make the POST request
response = requests.post(url, json=payload)

# Print the results
print("Status Code:", response.status_code)
print("Response:", response.json())
