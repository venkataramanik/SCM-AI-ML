import streamlit as st
import requests
import json
import base64
import io
import time

# --- 1. CONFIGURATION ---
# The NVIDIA NIM endpoint for the Llama4-Maverick multimodal model.
API_URL = "https://ai.api.nvidia.com/v1/llama4/maverick/chat/completions"

# --- 2. HELPER FUNCTIONS ---
def get_damage_report(image_bytes):
    """
    Sends an image to the NVIDIA NIM API and asks it to generate a detailed,
    markdown-formatted damage report.
    """
    # Encode the image data to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # The prompt instructs the model to act as a quality control expert and generate a Markdown report.
    report_prompt = f"""
    You are an expert quality control manager and risk assessor for a logistics company.
    Examine the provided image of a shipment or product and generate a detailed
    damage report in Markdown format.

    Your report should have the following sections:
    - **Report Title:** A clear title, e.g., "Damage Report for Shipment XYZ".
    - **Date of Assessment:** The current date.
    - **Item Description:** A brief description of the product or shipment.
    - **Description of Damage:** A detailed explanation of the type of damage visible (e.g., crushed box, water damage, torn packaging, broken seal).
    - **Severity Rating:** A rating of "Minor", "Moderate", or "Severe" based on the extent of the damage.
    - **Recommended Action:** A clear, actionable recommendation, such as "Quarantine for further inspection," "Process for return," or "Repackage and resend."
    - **Associated Risks:** A bulleted list of potential risks stemming from this damage (e.g., "Further product degradation," "Safety hazard," "Loss of brand trust").
    - **Additional Notes:** Any other observations or details.

    Do not include any text before or after the Markdown report. Start directly with the title.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer ", # The API key will be provided automatically.
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": report_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "model": "llama4-maverick",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024
    }

    retries = 5
    delay = 1
    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            response_data = response.json()
            # The model is instructed to return only the markdown text.
            report_markdown = response_data['choices'][0]['message']['content']
            return report_markdown

        except requests.exceptions.RequestException as e:
            st.error(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error("Failed to get a response from the API after multiple retries.")
                return None
        except (KeyError, IndexError) as e:
            st.error(f"Attempt {i+1} failed: Unexpected response format from the API. Error: {e}")
            if i < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error("Failed to parse the API response.")
                return None


# --- 3. STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="NVIDIA NIM AI Damage & Quality Control Report Generator", layout="wide")

st.title("NVIDIA NIM AI Damage & Quality Control Report Generator 📝🔍")
st.markdown("Upload an image of a damaged shipment to get an instant, structured quality control report.")

# Initialize session state variables
if 'report_markdown' not in st.session_state:
    st.session_state.report_markdown = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None

with st.sidebar:
    st.header("Upload Image for Report")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.session_state.image_bytes = uploaded_file.getvalue()
        st.image(st.session_state.image_bytes, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully!")

        if st.button("Generate Report", type="primary"):
            st.session_state.report_markdown = None
            with st.spinner("Analyzing image and generating report..."):
                st.session_state.report_markdown = get_damage_report(st.session_state.image_bytes)
                if st.session_state.report_markdown:
                    st.success("Report generated!")
    else:
        st.session_state.report_markdown = None
        st.session_state.image_bytes = None

# Main content area
if st.session_state.report_markdown:
    st.header("Generated Damage Report")
    st.markdown(st.session_state.report_markdown, unsafe_allow_html=True)

elif st.session_state.image_bytes:
    st.info("Click the 'Generate Report' button in the sidebar to create the report.")
else:
    st.info("Upload an image of a damaged product or shipment to see an automated quality control report generated here.")
