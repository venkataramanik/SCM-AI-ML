import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Initialize page settings for this specific multi-page view
st.set_page_config(page_title="Invoice OCR Converter", layout="wide")

st.title("📄 Intelligent Invoice & Document OCR Converter")
st.write(
    "Upload a scanned invoice, receipt, or shipping document. "
    "This module utilizes **PaddleOCR** to parse layout structures and extract text natively in the cloud."
)

# 1. Core Engine Loader (Cached so it only runs once across page switches)
@st.cache_resource
def load_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR: English weights, orientation angle correction enabled
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    except ImportError:
        st.error(
            "Missing dependencies! Ensure you have `paddlepaddle` "
            "and `paddleocr` active in your environment profile."
        )
        return None

ocr_model = load_ocr_engine()

# 2. Control Layout Sidebar
st.sidebar.header("OCR Filters")
conf_threshold = st.sidebar.slider(
    "Confidence Guardrail", 
    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
    help="Filters out low-confidence text detections caused by background artifact noise."
)

# 3. Main Document Drop-zone
uploaded_file = st.file_uploader(
    "Upload scanned paperwork or digital invoice images...", 
    type=["jpg", "jpeg", "png"]
)

# 4. Processing Engine Pipeline
if uploaded_file is not None and ocr_model is not None:
    
    # Read the file payload stream directly into a standard OpenCV image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated_img = img_bgr.copy()
    
    # Formulate a clean side-by-side comparative dashboard
    col_visual, col_data = st.columns(2)
    
    with col_visual:
        st.subheader("📸 Inbound Image Manifest")
        img_display = st.empty()
        img_display.image(uploaded_file, use_container_width=True)
        
    with col_data:
        st.subheader("📊 Structured Extraction Matrix")
        
        with st.spinner("Processing deep learning layout analysis..."):
            raw_ocr_results = ocr_model.ocr(img_bgr, cls=True)
            
        structured_table_rows = []
        raw_text_blocks = []
        
        # Unpack nested model arrays if text properties are discovered
        if raw_ocr_results and raw_ocr_results[0] is not None:
            for element in raw_ocr_results[0]:
                bounding_box = element[0]         # Coordinates: [[x1,y1], [x2,y2]...]
                text_string, match_score = element[1] # Read values and accuracy rankings
                
                if match_score >= conf_threshold:
                    structured_table_rows.append({
                        "Extracted Text Line": text_string, 
                        "Model Confidence": f"{match_score:.2%}"
                    })
                    raw_text_blocks.append(text_string)
                    
                    # Highlight localized bounding box targets on the image
                    pts = np.array(bounding_box, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Repaint OpenCV BGR color spaces to standard RGB for Streamlit rendering
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            img_display.image(annotated_rgb, use_container_width=True)
            
            # Interactive output data tabs
            text_tab, table_tab = st.tabs(["📋 Raw Plain Text Block", "📊 Searchable Data Grid"])
            
            with text_tab:
                compiled_text = "\n".join(raw_text_blocks)
                st.text_area(
                    label="Highlight text directly for downstream manual copy-pasting:", 
                    value=compiled_text, 
                    height=350
                )
                
            with table_tab:
                st.data_editor(structured_table_rows, use_container_width=True, num_rows="dynamic")
                
                # Format raw memory frames into downloadable CSV data strings
                csv_payload = "Text,Confidence\n" + "\n".join(
                    [f'"{row["Extracted Text Line"]}",{row["Model Confidence"]}' for row in structured_table_rows]
                )
                st.download_button(
                    label="📥 Export Table Data to CSV", 
                    data=csv_payload, 
                    file_name="invoice_parsed_data.csv", 
                    mime="text/csv"
                )
                
        else:
            st.warning("PaddleOCR verified execution smoothly, but found no readable alphanumeric text blocks.")
