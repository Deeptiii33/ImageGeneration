import streamlit as st
import google.generativeai as genai
import requests
import json
from PIL import Image
from io import BytesIO
import base64
import time
from dotenv import load_dotenv
import os
import pickle

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Gemini Pro 1.5 Chatbot with Image Generation",
    page_icon="🤖",
    layout="wide"
)

# Streamlit app styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .user .avatar {
        background-color: #0068C9;
    }
    .assistant .avatar {
        background-color: #FF4B4B;
    }
    .chat-message .content {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Gemini Pro 1.5 Chatbot with Image Generation")

# Get API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY", "")
stability_api_key = os.getenv("STABILITY_API_KEY", "")

# Check if API keys are available
if not google_api_key:
    st.error("Google API Key not found in environment variables. Please add GOOGLE_API_KEY to your .env file.")

if not stability_api_key:
    st.warning("Stability API Key not found in environment variables. Image generation will fall back to Hugging Face (free tier).")

# Sidebar for model configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Output Tokens", min_value=100, max_value=2048, value=1024, step=100)
    
    st.subheader("Image Settings")
    image_size = st.selectbox(
        "Image Size", 
        options=["512x512", "768x768"],
        index=0
    )
    image_steps = st.slider("Generation Steps", min_value=10, max_value=50, value=30, step=5)
    
    # Add reset button for chat history
    if st.button("Clear Chat History"):
        if os.path.exists(".chat_history.pkl"):
            os.remove(".chat_history.pkl")
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "generated_images" in st.session_state:
            st.session_state.generated_images = {}
        st.success("Chat history cleared!")
    
    st.divider()
    st.markdown("## About")
    st.markdown("""This chatbot uses Google's Gemini Pro 1.5 model for chat and Stability AI for image generation.
                API keys are loaded from the .env file in the project directory.""")

# Function to save chat history
def save_chat_history():
    with open(".chat_history.pkl", "wb") as f:
        pickle.dump({
            "messages": st.session_state.messages,
            "generated_images": st.session_state.generated_images
        }, f)

# Function to load chat history
def load_chat_history():
    if os.path.exists(".chat_history.pkl"):
        try:
            with open(".chat_history.pkl", "rb") as f:
                data = pickle.load(f)
                st.session_state.messages = data.get("messages", [])
                st.session_state.generated_images = data.get("generated_images", {})
        except Exception as e:
            st.error(f"Error loading chat history: {str(e)}")
            st.session_state.messages = []
            st.session_state.generated_images = {}
    else:
        st.session_state.messages = []
        st.session_state.generated_images = {}

# Initialize session state to store conversation history
if "messages" not in st.session_state or "generated_images" not in st.session_state:
    load_chat_history()

# Function to configure Gemini API
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Function to get text response from Gemini
def get_gemini_text_response(prompt, history):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        
        # Format the conversation history for the model
        formatted_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
        
        # Add the current prompt
        formatted_history.append({"role": "user", "parts": [prompt]})
        
        response = model.generate_content(formatted_history)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate image using Stability AI (free tier available)
def generate_image_stability(prompt, api_key):
    try:
        # Parse dimensions
        width, height = map(int, image_size.split('x'))
        
        # Stability API endpoint
        api_host = 'https://api.stability.ai'
        engine_id = "stable-diffusion-v1-6"  # Free tier engine
        
        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": image_steps,
            },
        )
        
        if response.status_code != 200:
            return None, f"Error: {response.json().get('message', 'Unknown error')}"
            
        data = response.json()
        
        # Extract base64 image
        for i, image in enumerate(data["artifacts"]):
            img_data = base64.b64decode(image["base64"])
            return "Generated by Stability AI", img_data
        
        return None, "No image was generated"
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Alternative image generation using Hugging Face (completely free)
def generate_image_huggingface(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": "Bearer (add hugging face api here)"}  # Demo API key
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        # If still loading model
        if response.status_code == 503:
            return None, "Model is loading. Try again in a moment."
            
        # Check for other errors    
        if response.status_code != 200:
            return None, f"Error: {response.text}"
            
        return "Generated by Hugging Face", response.content
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Function to refine prompt for better image generation
def refine_image_prompt(prompt):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.8,
            }
        )
        
        response = model.generate_content([
            "Enhance the following image prompt to make it more detailed and visually descriptive. "
            "The enhanced prompt will be used for AI image generation, so include details about style, "
            "lighting, perspective, colors, mood, etc. Don't change the main subject, just enhance the details. "
            "Keep it concise (under 400 characters): ",
            prompt
        ])
        
        return response.text
    except Exception as e:
        # If error, return original prompt
        return prompt

# Function to display messages
def display_messages():
    for i, message in enumerate(st.session_state.messages):
        role_class = "user" if message["role"] == "user" else "assistant"
        avatar_emoji = "👤" if message["role"] == "user" else "🤖"
        
        with st.container():
            st.markdown(f"""
            <div class="chat-message {role_class}">
                <div class="avatar">{avatar_emoji}</div>
                <div class="content">{message["role"].title()}: {message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display generated image if available
            if message.get("image_id"):
                image_id = message["image_id"]
                if image_id in st.session_state.generated_images:
                    if "error" in st.session_state.generated_images[image_id]:
                        st.error(st.session_state.generated_images[image_id]["error"])
                    else:
                        # Display image from binary data
                        img_data = st.session_state.generated_images[image_id]["data"]
                        st.image(img_data, caption=st.session_state.generated_images[image_id]["source"])
                        st.info(f"**Prompt used:** {st.session_state.generated_images[image_id]['prompt']}")

# Callback functions to handle form submissions
def handle_chat_submit():
    if st.session_state.chat_input:
        user_input = st.session_state.chat_input
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from Gemini
        response = get_gemini_text_response(user_input, st.session_state.messages[:-1])
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save chat history
        save_chat_history()

def handle_image_submit():
    if st.session_state.image_prompt_input:
        original_prompt = st.session_state.image_prompt_input
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": f"Generate an image of: {original_prompt}"})
        
        with st.spinner("Refining prompt for better image generation..."):
            # Refine the prompt using Gemini
            refined_prompt = refine_image_prompt(original_prompt)
        
        # Generate a unique ID for this image
        image_id = f"img_{len(st.session_state.generated_images)}"
        
        with st.spinner("Generating image..."):
            # Try Stability AI if API key is provided
            if stability_api_key:
                source, image_data = generate_image_stability(refined_prompt, stability_api_key)
            else:
                # Fall back to Hugging Face (completely free)
                source, image_data = generate_image_huggingface(refined_prompt)
        
        if source:
            # Store image data in session state
            st.session_state.generated_images[image_id] = {
                "data": image_data,
                "prompt": refined_prompt,
                "source": source
            }
            
            # Add assistant response to history with image reference
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here's the image I've generated based on your prompt:",
                "image_id": image_id
            })
        else:
            # Store error in session state
            st.session_state.generated_images[image_id] = {
                "error": image_data  # In this case, image_data contains the error message
            }
            
            # Add assistant response with error
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I encountered an error while generating your image:",
                "image_id": image_id
            })
        
        # Save chat history
        save_chat_history()

# Main layout
if google_api_key:
    configure_genai(google_api_key)
    
    # Chat and image generation tabs
    tab1, tab2 = st.tabs(["Chat", "Image Generation"])
    
    with tab1:
        # Display previous messages
        display_messages()
        
        # Text input for chat with on_change callback
        st.text_area(
            "Your message:", 
            key="chat_input", 
            height=100
        )
        
        st.button(
            "Send", 
            key="send_chat", 
            on_click=handle_chat_submit
        )
    
    with tab2:
        # Display previous messages
        display_messages()
        
        # Text input for image generation with on_change callback
        st.text_area(
            "Describe the image you want to generate:", 
            key="image_prompt_input", 
            height=100
        )
        
        image_source_info = "Using Stability AI API" if stability_api_key else "Using Hugging Face Inference API (free tier)"
        
        st.button(
            "Generate Image", 
            key="generate_image", 
            on_click=handle_image_submit
        )
else:
    st.error("Google API Key not found in environment variables. Please add GOOGLE_API_KEY to your .env file.")
