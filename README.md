# Text to Image Generation 🖼️🚀

Welcome to **Text to Image Generation** – a Streamlit-based web application that allows you to generate images from text prompts using cutting-edge AI models. Choose between **OpenAI DALL-E 3** and **Stable Diffusion** to create stunning visuals from your creative ideas.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Voice Input Functionality](#voice-input-functionality)

## Overview

**Text to Image Generation** is an interactive web application built with [Streamlit](https://streamlit.io/) that converts your text prompts into images. You can choose between:

- **OpenAI DALL-E 3**: Generate images using OpenAI's advanced DALL-E model (_requires your own API key_).
- **Stable Diffusion (Open Source)**: Generate images locally with the Stable Diffusion model provided by RunwayML using the [Diffusers](https://github.com/huggingface/diffusers) library.

This app is perfect for experimenting with AI-generated art and prototyping creative ideas.

## Features

- **Model Selection**: Easily switch between OpenAI DALL-E 3 and Stable Diffusion.
- **User-Friendly Interface**: A clean and responsive design with intuitive input fields.
- **Dynamic API Key Entry**: For OpenAI DALL-E 3 users, simply enter your API key in the sidebar.
- **Enhanced Voice Input for Prompts**:
  - **Record Your Voice**: Use the integrated voice recording feature to capture your spoken prompt.
  - **Accurate Transcription with Whisper**: The recorded audio is converted to the required format and transcribed using OpenAI's open-source [Whisper](https://github.com/openai/whisper) model.
  - **Automatic Prompt Update**: The transcribed text is automatically inserted into the prompt field so you can easily review and edit it.
- **Image Generation**: Generates a single image from your prompt and displays it directly in the browser.
- **NSFW Safety Bypass for Stable Diffusion**: (⚠️ **Note**: The safety checker is disabled for Stable Diffusion to prevent blank (black) images. Use responsibly.)

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

The new voice input feature allows you to record your voice to generate your image prompt. Here’s how it works:

- **Recording:**

  - Click on the **Record Voice Input** area (with the microphone icon) to start recording.
  - **Important:** Ensure your browser has microphone access enabled. If the recorder does not start, verify your permissions or try a different browser (Chrome, Firefox, or Edge are recommended).
- **Audio Processing & Transcription:**

  - The recorded audio is saved and converted to 16 kHz, mono, 16-bit PCM using pydub.
  - The converted audio is then transcribed using OpenAI’s open-source Whisper model, which provides improved accuracy over previous methods.
- **Prompt Update:**

  - Once transcribed, the voice prompt text is automatically inserted into the prompt text field so you can immediately see and, if needed, edit the text before generating your image.