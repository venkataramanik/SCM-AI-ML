import os
# Force dynamic inference globally across all pages before any ML engines load
os.environ["PADDLEX_INFERENCE_MODE"] = "dynamic"

import streamlit as st

st.set_page_config(
    page_title="Venkat's SCM & Data Portfolio"
)

st.title("Venkat Krishnan")
st.subheader("A playground for AI/ML solutions in Supply Chain Management")

st.write("---")

st.write("""
I'm curious how AI and Machine Learning can solve real-world SCM problems at large and provide business value.

This is a quick playground to explore those ideas. Navigate the sidebar to see projects that demonstrate how these concepts can be applied.
""")

st.write("---")

st.subheader("Connect with Me")
st.write("""
- **LinkedIn:** [Venkat Krishnan](https://www.linkedin.com/in/venkrish1/)  

""")
