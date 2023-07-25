# Anime VAE-GAN WebApp

This is a web application that uses an Anime VAE-GAN model to reconstruct images. The VAE-GAN model is capable of encoding and decoding anime images, allowing you to visualize the reconstructed versions.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/anime-vae-gan-webapp.git
cd anime-vae-gan-webapp
```

2. Install the required Python packages. It is recommended to create a virtual environment before installing the dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Download the pre-trained VAE-GAN model weights:
Before running the application, you need to download the pre-trained VAE-GAN model weights and save them as 'vae.pth' in the root directory of the project. You can get the pre-trained model from here.

## Usage
To run the web application, use the following command:

``` bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

This will start the FastAPI server, and you can access the web application at http://localhost:8000 in your web browser.

### Instructions

1. Access the home page of the web application at http://localhost:8000.
2. Click on the "Choose File" button to upload an anime image for reconstruction.

3. ![image](https://github.com/utkarsh-iitbhu/Synthetic-Serendipity-EnigmaticAnimeVAE-GAN/assets/84759422/4d1596f7-517e-4961-ba53-d39b6fa95268)

4. Click the "Upload" button to submit the image.

5. ![image](https://github.com/utkarsh-iitbhu/Synthetic-Serendipity-EnigmaticAnimeVAE-GAN/assets/84759422/fa317da1-41e0-4e94-9f2c-a33757577155)

6. Wait for the image to be processed and view the original and reconstructed images side by side.

7. ![image](https://github.com/utkarsh-iitbhu/Synthetic-Serendipity-EnigmaticAnimeVAE-GAN/assets/84759422/5a3c925a-f103-4ebd-a2f9-3ba61a364860)

8. Please note that the model is trained on anime images and may not work as expected with other images.
