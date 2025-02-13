import os
import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from openai import OpenAI
import speech_recognition as sr
from st_audiorec import st_audiorec
from pydub import AudioSegment

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

# In order for the voice text to appear in the prompt text field,
# we set its default value to st.session_state.voice_text if available.
default_prompt = st.session_state.voice_text if st.session_state.voice_text != "" else ""
img_description = st.text_input(
    "Enter the image description",
    placeholder="e.g., A futuristic city skyline at sunset",
    # Default value comes from the voice transcription (if available)
    value=default_prompt
)

# Voice recording column
col1, col2 = st.columns([4, 1])
with col2:
    st.write("Record Voice Input:")
    audio_data = st_audiorec()

# Process voice recording if available and of sufficient length.
if audio_data is not None:
    # Check if the recorded audio is long enough (adjust threshold as needed)
    if len(audio_data) < 1000:
        st.warning(
            "The recorded audio seems too short. Please record a longer voice prompt.")
    else:
        # Avoid reprocessing the same audio.
        if "prev_audio_data" not in st.session_state or st.session_state.prev_audio_data != audio_data:
            st.session_state.prev_audio_data = audio_data

            # Save the raw audio data to a temporary file.
            raw_audio_path = "temp.wav"
            with open(raw_audio_path, "wb") as f:
                f.write(audio_data)

            # Convert the audio file to the format required by pocketsphinx:
            # 16 kHz, mono, 16-bit PCM.
            try:
                sound = AudioSegment.from_file(raw_audio_path, format="wav")
                sound = sound.set_frame_rate(
                    16000).set_channels(1).set_sample_width(2)
                converted_audio_path = "temp_converted.wav"
                sound.export(converted_audio_path, format="wav")
            except Exception as e:
                st.error("Error converting audio file: " + str(e))
                converted_audio_path = raw_audio_path  # fallback

            # Use SpeechRecognition to transcribe the converted audio.
            r = sr.Recognizer()
            try:
                with sr.AudioFile(converted_audio_path) as source:
                    audio = r.record(source)
                    try:
                        # First, try offline recognition using pocketsphinx.
                        text = r.recognize_sphinx(audio)
                    except Exception as e:
                        st.warning("Offline recognition failed (" +
                                   str(e) + "). Trying online recognition...")
                        try:
                            text = r.recognize_google(audio)
                        except Exception as e2:
                            st.error(
                                "Online recognition also failed: " + str(e2))
                            text = ""
                    if text:
                        st.session_state.voice_text = text
                        st.info("Voice prompt converted to text: " + text)
                        # Attempt to force a rerun so that the text input is re-rendered with the new default.
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Recognition error: {e}")
            finally:
                # Clean up temporary files.
                try:
                    os.remove(raw_audio_path)
                    os.remove(converted_audio_path)
                except Exception:
                    pass

# ---------------------------
# Final Prompt Creation
# ---------------------------
# Now, the final prompt is directly taken from the text input field.
final_prompt = img_description

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
    if not final_prompt.strip():
        st.error("Please provide an image description or record a voice prompt.")
    else:
        if model_choice == "OpenAI DALL-E":
            if not openai_api_key:
                st.error("Please provide a valid OpenAI API key in the sidebar.")
            else:
                with st.spinner("Generating image(s) using OpenAI DALL-E..."):
                    try:
                        images = generate_image_openai(
                            final_prompt, num_of_images, openai_api_key)
                        st.success("Image generation complete!")
                        st.image(images, caption=final_prompt,
                                 use_container_width=True)
                    except Exception as e:
                        st.error(
                            f"Error generating image with OpenAI DALL-E: {e}")
        else:
            with st.spinner("Generating image(s) using Stable Diffusion..."):
                try:
                    images = generate_image_stable_diffusion(
                        final_prompt, num_of_images)
                    st.success("Image generation complete!")
                    st.image(images, caption=final_prompt,
                             use_container_width=True)
                except Exception as e:
                    st.error(
                        f"Error generating image with Stable Diffusion: {e}")
