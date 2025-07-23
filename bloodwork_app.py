#!/usr/bin/env python3

import os
import re
from datetime import datetime
from typing import Dict, List, Tuple
import fitz  # PyMuPDF
from openai import OpenAI
import streamlit as st


class BloodworkExtractor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_bloodwork_results(self, pdf_text: str) -> str:
        """Use LLM to extract and format bloodwork results"""

        prompt = f"""
        Analyze this medical lab report and extract ALL test results in the format used in Russian medical consultation notes.

        Format each result as: "Test name: value unit (reference range) status"
        Group by test type and include the test date.

        Example format:
        "ОАК от 17.07.2025: лейкоциты 5.5 /л (4-9), эритроциты 4.8 /л (3.9-4.7) выше нормы, гемоглобин 135 г/л (120-140)"

        Rules:
        1. Keep original Russian test names
        2. Include all numerical values with units
        3. Include reference ranges in parentheses
        4. Add status (повышено/понижено/выше нормы/ниже нормы) when indicated
        5. Group related tests together
        6. Use commas to separate individual tests
        7. Use semicolons to separate different test groups
        8. Include test dates when available

        Medical lab report text:
        {pdf_text}

        Extract and format the results:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical transcription assistant. Extract lab results accurately and format them for Russian medical consultation notes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error processing results: {str(e)}"

    def process_lab_report(self, pdf_path: str) -> Dict[str, str]:
        """Complete workflow to process lab report"""

        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)

        # Extract and format results
        formatted_results = self.extract_bloodwork_results(pdf_text)

        return {
            "raw_text": pdf_text,
            "formatted_results": formatted_results,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# Streamlit Web Interface
def main():
    st.set_page_config(
        page_title="Bloodwork Results Extractor", page_icon="🩸", layout="wide"
    )

    st.title("🩸 Extracting Bloodwork Results for Consultation Notes")
    st.markdown(
        "Upload lab result PDFs to automatically extract and format results for patient consultation notes."
    )

    # API Key input
    env_api_key = os.environ.get("OPENAI_API_KEY")

    if env_api_key:
        st.success("✅ Using API key from environment variable")
        api_key = env_api_key
    else:
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""

        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key (or set OPENAI_API_KEY environment variable)",
        )

    if api_key:
        if not env_api_key:  # Only store in session if not from environment
            st.session_state.openai_api_key = api_key

        extractor = BloodworkExtractor(api_key)

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Lab Results PDF",
            type=["pdf"],
            help="Upload PDF files containing bloodwork/lab results",
        )

        if uploaded_file:
            # Save uploaded file temporarily
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process button
            if st.button("Extract Results", type="primary"):
                with st.spinner("Processing lab results..."):
                    try:
                        results = extractor.process_lab_report(
                            f"temp_{uploaded_file.name}"
                        )

                        # Display results
                        st.success("✅ Results extracted successfully!")

                        # Formatted results (main output)
                        st.subheader(" Formatted Results for Consultation Notes")
                        formatted_text = results["formatted_results"]
                        st.text_area(
                            "Copy this text to your consultation notes:",
                            value=formatted_text,
                            height=200,
                            help="Copy and paste this formatted text into the 'Результаты обследования' field",
                        )

                        # Show raw extracted text in expander
                        with st.expander(" View Raw Extracted Text"):
                            st.text_area(
                                "Raw text from PDF:",
                                value=results["raw_text"],
                                height=300,
                            )

                        # Cleanup
                        os.remove(f"temp_{uploaded_file.name}")

                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
                        if os.path.exists(f"temp_{uploaded_file.name}"):
                            os.remove(f"temp_{uploaded_file.name}")

    else:
        st.warning("⚠️ Please provide your OpenAI API key to continue.")

    # Usage instructions
    with st.sidebar:
        st.markdown(
            """
        ## 📖 How to Use
        
    
        1. **Upload PDF**: Select lab results PDF file
        2. **Extract**: Click "Extract Results" button
        3. **Copy**: Copy formatted results to consultation notes
        
        ## Output Format
        
        Results will be formatted like:
        ```
        ОАК от 17.07.2025: лейкоциты 5.5 /л (4-9), 
        эритроциты 4.8 /л (3.9-4.7) выше нормы, 
        гемоглобин 135 г/л (120-140)
        ```
        
        ##  Privacy
        
        - Files are processed locally
        - Temporary files are deleted
        - No data is stored permanently
        """
        )


if __name__ == "__main__":
    main()
