def get_ocr_text(image_bytes, api_key):
    """Send image to NVIDIA OCR and return list of text lines."""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{img_b64}",
            }
        ]
    }

    try:
        resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        # Show raw failure details prominently
        if not resp.ok:
            st.error(f"OCR API {resp.status_code}: {resp.text[:500]}")
            return []

        data = resp.json()

        # Handle common response shapes
        lines = []
        if isinstance(data, dict):
            # API Catalog OCR usually returns a top-level "predictions" list
            preds = data.get("predictions") or data.get("results") or data.get("detections")
            if isinstance(preds, list):
                for p in preds:
                    # Try common field names
                    txt = p.get("text") or p.get("value") or p.get("string") or ""
                    if txt:
                        lines.append(txt)
            elif "error" in data:
                st.error(str(data["error"]))
        if not lines:
            st.warning("OCR returned no text. Check image quality/rotation and API key permissions.")
        return lines

    except requests.exceptions.RequestException as e:
        st.error(f"OCR request failed: {e}")
        return []
    except ValueError as e:
        st.error(f"OCR response parsing failed: {e}")
        return []
