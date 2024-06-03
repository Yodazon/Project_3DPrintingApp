from transformers import pipeline
HF_API_Key = "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"



import torchvision.transforms as transforms
from PIL import Image


image_size = 227

def preProcess(uploaded_image):
        # Load and preprocess the image
    image = Image.open(uploaded_image).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_image = transform(image).unsqueeze(0)

    return input_image







from langchain_community.llms import huggingface_hub

model = huggingface_hub(repo_id = "Yodazon/3DPrintFailureType", huggingface_api_token = HF_API_Key)

processed_img = preProcess("3d-print-fail.jpg")

print(model.invoke(processed_img))