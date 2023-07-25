from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import torch
import base64
import torchvision.transforms as transforms
from model_file import VAE  # Import your VAE model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = VAE()
model.load_state_dict(torch.load('vae.pth', map_location=torch.device('cpu')))
model.eval()
image_size = 64

@app.post("/decode_image/", response_class=HTMLResponse)
async def decode_image(request: Request, file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    preprocessed_image = transform(pil_image).unsqueeze(0)

    # Perform VAE decoding
    with torch.no_grad():
        encoded_image = model.encode(preprocessed_image)
        decoded_image = model.decode(encoded_image).cpu()

        # Convert the decoded image to base64
    decoded_bytes = io.BytesIO()
    decoded_pil_image = transforms.ToPILImage()(decoded_image.squeeze(0))
    decoded_pil_image.save(decoded_bytes, format='PNG')
    decoded_bytes.seek(0)
    decoded_base64 = base64.b64encode(decoded_bytes.getvalue()).decode()

    # Convert the original image to base64
    original_bytes = io.BytesIO()
    pil_image.save(original_bytes, format='PNG')
    original_bytes.seek(0)
    original_base64 = base64.b64encode(original_bytes.getvalue()).decode()

    return templates.TemplateResponse("result.html", {"request": request, "original_image": original_base64, "decoded_image": decoded_base64})
    

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, warning: str = None):
    return templates.TemplateResponse("home.html", {"request": request, "warning": warning})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
