import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from openai import OpenAI

# ---------------------------
# Page & App Configuration
# ---------------------------
st.set_page_config(
    page_title="Text to Image Generation",
    page_icon="camera",
    layout="wide"
)

# App title and instructions
st.markdown("<h1 style='text-align: center;'>Text to Image Generation</h1>",
            unsafe_allow_html=True)
st.markdown(
    """
    Generate images from your text prompts using either **OpenAI DALL-E** or **Stable Diffusion**.
    
    For OpenAI DALL-E, please provide your own API key.
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar Configuration
# ---------------------------
st.sidebar.header("Configuration")
model_choice = st.sidebar.radio(
    "Choose a generation model",
    ("OpenAI DALL-E", "Stable Diffusion (Open Source)")
)

# Only show the API key input if OpenAI DALL-E is selected.
if model_choice == "OpenAI DALL-E":
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", type="password")
else:
    openai_api_key = None

# ---------------------------
# Main Input Section
# ---------------------------
st.subheader("Image Generation Settings")
img_description = st.text_input(
    "Enter the image description",
    placeholder="e.g., A futuristic city skyline at sunset"
)
num_of_images = st.number_input(
    "Number of images", min_value=1, max_value=10, value=1, step=1
)

# ---------------------------
# OpenAI DALL·E Generation Function
# ---------------------------


def generate_image_openai(image_description, num_images, api_key):
    """
    Generate images using OpenAI's DALL-E model.
    Returns a list of image URLs.
    """
    client = OpenAI(api_key=api_key)
    img_response = client.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard",
        n=num_images
    )
    # Extract image URLs from the response
    image_urls = [img.url for img in img_response.data]
    return image_urls

# ---------------------------
# Stable Diffusion Setup & Generation Function
# ---------------------------


@st.cache_resource
def load_stable_diffusion():
    """
    Load and cache the Stable Diffusion pipeline.
    The NSFW safety checker is disabled to avoid returning black images.
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    # Disable the safety checker to prevent black images
    pipe.safety_checker = lambda images, clip_input, **kwargs: (
        images, [False] * len(images))
    return pipe


def generate_image_stable_diffusion(image_description, num_images):
    """
    Generate images using Stable Diffusion.
    Returns a list of PIL Image objects.
    """
    pipe = load_stable_diffusion()
    result = pipe(
        image_description,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=num_images
    )
    return result.images


# ---------------------------
# Image Generation Trigger
# ---------------------------
if st.button("Generate Image"):
    if not img_description:
        st.error("Please enter an image description.")
    else:
        if model_choice == "OpenAI DALL-E":
            if not openai_api_key:
                st.error("Please provide a valid OpenAI API key in the sidebar.")
            else:
                with st.spinner("Generating image(s) using OpenAI DALL-E..."):
                    try:
                        images = generate_image_openai(
                            img_description, num_of_images, openai_api_key)
                        st.success("Image generation complete!")
                        st.image(images, caption=img_description,
                                 use_container_width=True)
                    except Exception as e:
                        st.error(
                            f"Error generating image with OpenAI DALL-E: {e}")
        else:
            with st.spinner("Generating image(s) using Stable Diffusion..."):
                try:
                    images = generate_image_stable_diffusion(
                        img_description, num_of_images)
                    st.success("Image generation complete!")
                    st.image(images, caption=img_description,
                             use_container_width=True)
                except Exception as e:
                    st.error(
                        f"Error generating image with Stable Diffusion: {e}")
