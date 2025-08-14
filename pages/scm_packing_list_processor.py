import streamlit as st
import requests
import json
import base64
import io
import time
import pandas as pd
import re

# --- 1. CONFIGURATION ---
# The URL for the NVIDIA Nemo Retriever OCR API.
NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
# The maximum allowed image size in bytes, based on common API limits.
MAX_IMAGE_SIZE_BYTES = 180_000

# --- 2. HELPER FUNCTIONS ---
def get_ocr_text(image_bytes, api_key):
    """
    Sends an image to the NVIDIA OCR API to extract text using a user-provided API key.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }
        ]
    }

    retries = 5
    delay = 1
    for i in range(retries):
        try:
            st.info(f"Attempting to extract text from image (Attempt {i+1})...")
            ocr_response = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
            ocr_response.raise_for_status()
            
            ocr_data = ocr_response.json()
            if 'predictions' in ocr_data and ocr_data['predictions']:
                # Extract all text from the OCR result
                extracted_lines = [p['text'] for p in ocr_data['predictions']]
                st.success("Text extracted successfully!")
                return extracted_lines
            else:
                st.warning("No text was extracted by the OCR model.")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"OCR API call failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error("Failed to get a response from the OCR API after multiple retries.")
                return []

def parse_packing_list(extracted_lines):
    """
    Parses a list of text lines to find packing list items.
    This is a simple heuristic-based parser.
    """
    items = []
    # Regular expression to find a pattern like: [SKU] [Description] [Quantity]
    # This pattern is highly dependent on the layout of your packing list.
    # We look for lines that contain a product code (e.g., 'A123'), a description, and a number.
    item_pattern = re.compile(r'(\w+-\w+)\s+(.*?)\s+(\d+)')
    
    for line in extracted_lines:
        match = item_pattern.search(line)
        if match:
            items.append({
                "SKU": match.group(1),
                "Description": match.group(2).strip(),
                "Quantity": int(match.group(3))
            })
            
    return items

# --- 3. STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="NVIDIA OCR Packing List Processor", layout="wide")

st.title("📦 NVIDIA OCR Packing List Processor")
st.markdown("Upload an image of a packing list to automatically extract and structure the data.")

# Initialize session state variables
if 'processed_items' not in st.session_state:
    st.session_state.processed_items = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None
if 'extracted_lines' not in st.session_state:
    st.session_state.extracted_lines = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = "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"

with st.sidebar:
    st.header("1. API Key")
    st.info("Your NVIDIA API key is already configured.")
    st.text_input("NVIDIA API Key", value=st.session_state.api_key, type="password", disabled=True)

    st.header("2. Upload Packing List Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.session_state.image_bytes = uploaded_file.getvalue()
        st.image(st.session_state.image_bytes, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully!")

    if st.button("Process Packing List", type="primary"):
        if not st.session_state.api_key:
            st.error("The API key is missing. Please ensure it is set.")
        elif not st.session_state.image_bytes:
            st.error("Please upload an image first.")
        elif len(st.session_state.image_bytes) > MAX_IMAGE_SIZE_BYTES:
            st.error(f"Image is too large! Please upload an image smaller than {MAX_IMAGE_SIZE_BYTES} bytes.")
        else:
            st.session_state.processed_items = None
            st.session_state.extracted_lines = None
            with st.spinner("Extracting text and parsing data..."):
                extracted_lines = get_ocr_text(st.session_state.image_bytes, st.session_state.api_key)
                st.session_state.extracted_lines = extracted_lines
                if extracted_lines:
                    st.session_state.processed_items = parse_packing_list(extracted_lines)
                    st.success("Packing list processed!")
                else:
                    st.error("Could not process packing list. Please try a different image.")

# Main content area
if st.session_state.processed_items:
    st.header("Processed Packing List Items")
    if st.session_state.processed_items:
        df = pd.DataFrame(st.session_state.processed_items)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No items were found in the packing list image.")

elif st.session_state.image_bytes:
    st.info("Click the 'Process Packing List' button in the sidebar to extract the data.")
else:
    st.info("Upload an image of a packing list to see the processed data here.")

# Debugging section to show raw text
if st.session_state.extracted_lines is not None:
    st.markdown("---")
    with st.expander("Show Raw Extracted Text (for debugging)"):
        if st.session_state.extracted_lines:
            st.code("\n".join(st.session_state.extracted_lines))
        else:
            st.info("No text was extracted from the image. This indicates a problem with the image quality, size, or content.")
