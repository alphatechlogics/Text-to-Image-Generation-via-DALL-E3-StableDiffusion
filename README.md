# Text to Image Generation 🖼️🚀

Welcome to **Text to Image Generation** – a Streamlit-based web application that allows you to generate images from text prompts using cutting-edge AI models. Choose between **OpenAI DALL-E 3** and **Stable Diffusion** to create stunning visuals based on your creative ideas!

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Voice Input Functionality](#voice-input-functionality)

## Overview

**Text to Image Generation** is an interactive web application built with [Streamlit](https://streamlit.io/) that lets you convert text prompts into images. You can choose between:

- **OpenAI DALL-E 3**: Generate images using OpenAI's advanced DALL-E model (requires your own API key).
- **Stable Diffusion (Open Source)**: Generate images locally with the Stable Diffusion model provided by RunwayML using the [Diffusers](https://github.com/huggingface/diffusers) library.

This app is perfect for experimenting with AI-generated art and prototyping creative ideas.

## Features

- **Model Selection**: Easily switch between OpenAI DALL-E 3 and Stable Diffusion.
- **User-Friendly Interface**: Clean and responsive design with input fields for text prompts and image count.
- **Dynamic API Key Entry**: For OpenAI DALL-E 3 users, enter your API key directly in the app sidebar.
- **Voice Input for Prompts**:
  - **Record Your Voice**: A new voice recording feature allows you to record a voice prompt directly within the app.
  - **Automatic Voice-to-Text Conversion**: The recorded audio is automatically transcribed into text using a combination of offline (pocketsphinx) and online (Google) speech recognition services.
  - **Seamless Prompt Integration**: The transcribed voice text is automatically inserted into the prompt text field for a streamlined experience.
- **Enhanced Image Generation**: Displays generated images directly in the browser.
- **NSFW Safety Bypass for Stable Diffusion**: (⚠️ **Note**: The safety checker is disabled for Stable Diffusion to prevent blank (black) images from being returned. Use responsibly.)

## Models Used

- **OpenAI DALL-E 3**  
  Learn more at the official [OpenAI DALL-E 3 page](https://openai.com/dall-e-3).

- **Stable Diffusion v1.5**  
  Check out the model details on [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and the [Diffusers GitHub repository](https://github.com/huggingface/diffusers).

## Installation

### Prerequisites

- Python 3.10 or later
- [pip](https://pip.pypa.io/en/stable/)
- (Optional for voice input) **ffmpeg** installed on your system for audio conversion

### Usage

1. **Clone the repository (if not already done):**

```bash
git clone https://github.com/alphatechlogics/Text-to-Image-Generation-via-DALL-E3-StableDiffusion.git
cd Text-to-Image-Generation-via-DALL-E3-StableDiffusion
```

### Install Required Packages

Open your terminal and install the dependencies:

```bash
pip install -r requirements.txt
```

2. **Run the Streamlit app:**

```bash
streamlit run app.py
```

3. **Select a Model in the Sidebar:**

- For **OpenAI DALL-E 3:** Enter your API key in the sidebar.
- For **Stable Diffusion (Open Source):** No API key is required.

4. **Enter your image prompt and desired number of images** in the main section.

5. **Click on "Generate Image"** and watch as your creative prompt is transformed into art!

## Voice Input Functionality

The new voice input feature lets you simply record your voice to create your image prompt. Here’s how it works:

- **Recording:**
  Click on the voice recording button (displayed as a microphone icon) next to the text prompt field.

- **Transcription:**
  The app uses st_audiorec for recording audio, and then employs SpeechRecognition along with pydub to convert your voice to text.
  The transcription process attempts offline recognition first and then falls back to online recognition if needed.

- **Automatic Insertion:**
  Once transcribed, the voice prompt text is automatically set as the default value in the text input field so you can immediately see and use it for image generation.
