import os
import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from openai import OpenAI
from st_audiorec import st_audiorec
from pydub import AudioSegment
import whisper  # Using OpenAI's Whisper

# ---------------------------
# Page & App Configuration (must be first)
# ---------------------------
st.set_page_config(
    page_title="Text to Image Generation",
    page_icon="camera",
    layout="wide"
)

# ---------------------------
# Custom CSS for enhanced UI
# ---------------------------
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
    <style>
    /* Page background and font */
    body {
        background-color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Title styling */
    h1 {
        color: #333333;
        margin-bottom: 0;
    }
    /* Subtitle styling */
    h3 {
        color: #008CBA;
        margin-bottom: 0;
    }
    /* Input fields styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 10px;
    }
    /* Generate Button styling */
    .stButton>button {
        background-color: #008CBA;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #007399;
    }
    /* Voice recorder container styling */
    .voice-recorder-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .voice-recorder-container h3 {
        margin: 0;
        padding-bottom: 10px;
        font-size: 20px;
    }
    .voice-recorder-container p {
        margin: 0;
        font-size: 14px;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# App Title and Instructions
# ---------------------------
st.markdown("<h1 style='text-align: center;'>Text to Image Generation</h1>",
            unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #555555; font-size:18px;'>
    Generate images from your text prompts using either <strong>OpenAI DALL-E</strong> or <strong>Stable Diffusion</strong>.<br>
    For OpenAI DALL-E, please provide your own API key.
    </p>
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

if model_choice == "OpenAI DALL-E":
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", type="password")
else:
    openai_api_key = None

# Initialize voice_text in session state if not already set.
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ---------------------------
# Main Input Section
# ---------------------------
st.subheader("Image Generation Settings")

# Create a placeholder for the prompt input field so it can be updated dynamically.
prompt_placeholder = st.empty()
default_prompt = st.session_state.voice_text if st.session_state.voice_text != "" else ""
img_description = prompt_placeholder.text_input(
    "Enter the image description",
    placeholder="e.g., A futuristic city skyline at sunset",
    value=default_prompt,
    key="img_description"
)

# Create two columns: one for the prompt input (above) and one for voice recording.
col1, col2 = st.columns([4, 1])
# Audio recording section
with col2:
    st.markdown("""
        <div class="voice-recorder-container">
            <h3><i class="fas fa-microphone"></i> Record Voice Input</h3>
            <p>Click below to record your prompt</p>
            <p style="font-size:12px; color:#999;">Ensure your browser has microphone access enabled.</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        # Add a warning message about browser compatibility
        st.markdown("⚠️ Audio recording requires Chrome/Firefox/Edge browser")

        # Wrap the audio recorder in error handling
        audio_data = st_audiorec()
        if audio_data is not None:
            st.success("Audio recorded successfully!")

    except Exception as e:
        st.error(f"Audio recording error: {str(e)}")
        st.info("Please try:\n1. Using Chrome/Firefox/Edge browser\n2. Allowing microphone access\n3. Refreshing the page")
        audio_data = None
        
# Check if audio data was captured; if not, inform the user.
if audio_data is None:
    st.warning(
        "No audio data received. Please check that your microphone is enabled and try again.")

# Process voice recording if available and of sufficient length.
if audio_data is not None:
    if len(audio_data) < 1000:
        st.warning(
            "The recorded audio seems too short. Please record a longer voice prompt.")
    else:
        if "prev_audio_data" not in st.session_state or st.session_state.prev_audio_data != audio_data:
            st.session_state.prev_audio_data = audio_data

            # Save raw audio to a temporary file.
            raw_audio_path = "temp.wav"
            with open(raw_audio_path, "wb") as f:
                f.write(audio_data)

            # Convert audio to 16 kHz, mono, 16-bit PCM (recommended for Whisper)
            try:
                sound = AudioSegment.from_file(raw_audio_path, format="wav")
                sound = sound.set_frame_rate(
                    16000).set_channels(1).set_sample_width(2)
                converted_audio_path = "temp_converted.wav"
                sound.export(converted_audio_path, format="wav")
            except Exception as e:
                st.error("Error converting audio file: " + str(e))
                converted_audio_path = raw_audio_path  # fallback

            # Use Whisper for transcription.
            try:
                whisper_model = whisper.load_model("base")
                result = whisper_model.transcribe(converted_audio_path)
                text = result["text"].strip()
                if text:
                    st.session_state.voice_text = text
                    st.info("Voice prompt converted to text: " + text)
                    # Update the prompt input field with the new transcription.
                    prompt_placeholder.text_input(
                        "Enter the image description",
                        placeholder="e.g., A futuristic city skyline at sunset",
                        value=text,
                        key="img_description_updated"
                    )
            except Exception as e:
                st.error("Whisper transcription failed: " + str(e))
            finally:
                # Clean up temporary files.
                try:
                    os.remove(raw_audio_path)
                    os.remove(converted_audio_path)
                except Exception:
                    pass

# Final prompt is taken from the updated text input field if available.
final_prompt = st.session_state.get("img_description_updated", img_description)

# ---------------------------
# OpenAI DALL-E Generation Function
# ---------------------------


def generate_image_openai(image_description, api_key):
    client = OpenAI(api_key=api_key)
    img_response = client.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard"
    )
    image_urls = [img.url for img in img_response.data]
    return image_urls

# ---------------------------
# Stable Diffusion Setup & Generation Function
# ---------------------------


@st.cache_resource
def load_stable_diffusion():
    model_id = "runwayml/stable-diffusion-v1-5"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, clip_input, **kwargs: (
        images, [False] * len(images))
    return pipe


def generate_image_stable_diffusion(image_description):
    pipe = load_stable_diffusion()
    result = pipe(
        image_description,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    return result.images


# ---------------------------
# Image Generation Trigger
# ---------------------------
if st.button("Generate Image"):
    if not final_prompt.strip():
        st.error("Please provide an image description or record a voice prompt.")
    else:
        if model_choice == "OpenAI DALL-E":
            if not openai_api_key:
                st.error("Please provide a valid OpenAI API key in the sidebar.")
            else:
                with st.spinner("Generating image using OpenAI DALL-E..."):
                    try:
                        images = generate_image_openai(
                            final_prompt, openai_api_key)
                        st.success("Image generation complete!")
                        st.image(images, caption=final_prompt,
                                 use_container_width=True)
                    except Exception as e:
                        st.error(
                            f"Error generating image with OpenAI DALL-E: {e}")
        else:
            with st.spinner("Generating image using Stable Diffusion..."):
                try:
                    images = generate_image_stable_diffusion(final_prompt)
                    st.success("Image generation complete!")
                    st.image(images, caption=final_prompt,
                             use_container_width=True)
                except Exception as e:
                    st.error(
                        f"Error generating image with Stable Diffusion: {e}")
