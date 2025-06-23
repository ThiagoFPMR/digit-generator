import streamlit as st
from PIL import Image
import io
import torch
import torch.nn as nn
import numpy as np


latent_dim = 100
n_classes = 10
img_size = 28
channels = 1

img_shape = (channels, img_size, img_size) 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# Dummy placeholder for your model's image generation function
# Replace this with your actual function
def generate_images(number, model):
    """
    Takes an integer 'number' as input and generates 5 images using the Generator model.
    Returns a list of 5 PIL Image objects
    """
    # Example: just create 5 blank images for demo
    images = []
    for i in range(5):
        noise = torch.randn(1, latent_dim)
        labels = torch.tensor([number % n_classes])
        img = model(noise, labels)
        img = img.detach().numpy().reshape(img_shape)
        img = (img * 127.5 + 127.5).astype(np.uint8)
        img = Image.fromarray(img[0], 'L')
        images.append(img)
    return images

def main():
    st.title("Image Generator App")

    model = Generator()
    # Load your pre-trained model here if available
    model.load_state_dict(torch.load('gen.pth', map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Input from user
    user_input = st.number_input("Enter a number", min_value=0, max_value=1000, step=1)

    if st.button("Generate Images"):
        with st.spinner("Generating images..."):
            images = generate_images(user_input, model)
        
        st.success("Done! Here are your images:")

        # Display images
        for idx, img in enumerate(images):
            st.image(img, caption=f"Generated Image {idx+1}", use_container_width=True)

if __name__ == "__main__":
    main()
