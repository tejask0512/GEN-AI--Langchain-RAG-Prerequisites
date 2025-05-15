# Mistral AI Chat Application
# A simple web application that interacts with Mistral AI's chat API

import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the Mistral API configuration
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def get_api_key():
    """Retrieve API key from environment variable or user input"""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        api_key = st.session_state.get("api_key", "")
    return api_key

def call_mistral_api(messages, model="mistral-large-latest", temperature=0.7, max_tokens=1024, api_key=None):
    """
    Call the Mistral AI Chat API
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use (default: mistral-large-latest)
        temperature: Controls randomness in responses (0.0-1.0)
        max_tokens: Maximum tokens to generate
        api_key: Mistral API key
        
    Returns:
        API response as dictionary
    """
    if not api_key:
        raise ValueError("API key is required")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response: {e.response.text}")
        return None

def display_chat_message(role, content):
    """Display a chat message with appropriate styling"""
    if role == "user":
        st.write(f"**You:** {content}")
    elif role == "assistant":
        st.write(f"**Mistral AI:** {content}")
    else:
        st.write(f"**{role.capitalize()}:** {content}")

def main():
    st.set_page_config(page_title="Mistral AI Chat App", page_icon="🤖")
    
    st.title("Mistral AI Chat Application")
    st.markdown("Interact with Mistral's powerful AI models through this simple chat interface.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key_input = st.text_input(
            "Mistral API Key", 
            value=st.session_state.get("api_key", ""),
            type="password",
            help="Enter your Mistral API key. Get one at https://console.mistral.ai/"
        )
        
        if api_key_input:
            st.session_state["api_key"] = api_key_input
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            options=[
                "mistral-large-latest", 
                "mistral-medium-latest", 
                "mistral-small-latest",
                "open-mixtral-8x7b"
            ],
            index=0
        )
        
        # Parameter tuning
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="Higher values make output more random, lower values more deterministic")
        
        max_tokens = st.slider("Max Tokens", min_value=64, max_value=4096, value=1024, step=64,
                              help="Maximum length of the generated response")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses the [Mistral AI API](https://docs.mistral.ai/) to generate responses.")
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_chat_message("user", user_input)
        
        api_key = get_api_key()
        
        if not api_key:
            st.warning("Please enter your Mistral API key in the sidebar.")
        else:
            with st.spinner("Mistral AI is thinking..."):
                response = call_mistral_api(
                    st.session_state.messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
                
                if response and "choices" in response and len(response["choices"]) > 0:
                    assistant_response = response["choices"][0]["message"]["content"]
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    display_chat_message("assistant", assistant_response)
                    
                    # Display token usage if available
                    if "usage" in response:
                        with st.expander("Response Metadata"):
                            st.json(response["usage"])
                else:
                    st.error("Failed to get a response from Mistral AI")

if __name__ == "__main__":
    main()