# Importing required libraries
import streamlit as st
import replicate
import time
import os
from PIL import Image
import load_dotenv
# Load environment variables
load_dotenv()


# Initialize debounce variables
last_call_time = 0
debounce_interval = 2  # Set the debounce interval (in seconds) to your desired value

# A Debounced function for replicate.run
@st.cache_resource(ttl=3600)
def debounce_replicate_run(llm, prompt, max_len, temperature, top_p):
    global last_call_time

    # Get the current time
    current_time = time.time()

    # Calculate the time elapsed since the last call
    elapsed_time = current_time - last_call_time

    # Check if the elapsed time is less than the debounce interval
    if elapsed_time < debounce_interval:
        return "Hello! You are sending requests too fast. Please wait a few seconds before sending another request."

    # Update the last call time to the current time
    last_call_time = time.time()
    
    output = replicate.run(llm, input={"prompt": prompt + "Assistant: ", "max_length": max_len, "temperature": temperature, "top_p": top_p, "repetition_penalty": 1})
    return output

# App H title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate CredentialsM
with st.sidebar:
    # Load and display Iridium logo.
    logo_path = "Images/IridiumAILogo.png" 
    iridium_logo = Image.open(logo_path)
    st.image(iridium_logo, use_column_width=False)
    col1, col2 = st.columns([1, 4])
    # Add image to the first column
    with col1:
        st.image("Images/llama2.jpg", width=50)

# Add title to the second column
    with col2:
        st.title("Llama 3 Chatbot")
    #st.title('🦙💬 Llama 3 Chatbot')
    #st.markdown("<style>div.stMarkdown {margin-bottom: -10px;}</style>", unsafe_allow_html=True)
    st.write('Chatbot powered by the Llama 3 LLM model from Meta')

    # Check for A Replicate API token
    replicate_api = os.getenv("REPLICATE_API_TOKEN")
    #replicate_api = st.secrets.get("REPLICATE_API_TOKEN")
    if not replicate_api:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
        st.write("Please ensure to provide your Replicate API token.")

    # Select Llama2 model
    selected_model = st.sidebar.selectbox('Choose a Llama3 model', ['Llama3-8B', 'Llama3-70B'], key='selected_model')

    # Select Llama2 parameters
    max_length = st.sidebar.slider('Max Length', min_value=32, max_value=512, value=160)
    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=0.6, step=0.01)
    top_p = st.sidebar.slider('Top-p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Store LLM generated responses D
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response.
def generate_llama2_response(prompt_input, selected_model, max_length, temperature, top_p):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    if selected_model == 'Llama3-8B':
        llm = 'meta/meta-llama-3-8b'
    elif selected_model == 'Llama3-70B':
        llm = 'meta/meta-llama-3-70b'

    output = debounce_replicate_run(llm, f"{string_dialogue} {prompt_input}", max_length, temperature, top_p)
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt, selected_model, max_length, temperature, top_p)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
