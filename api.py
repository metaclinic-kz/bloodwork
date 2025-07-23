from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from datetime import datetime
import os
import fitz  # PyMuPDF
from openai import OpenAI

app = FastAPI()

# Разрешаем CORS (если вызываешь из Bubble.io)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Класс, как в твоем Streamlit-коде
class BloodworkExtractor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_bloodwork_results(self, pdf_text: str) -> str:
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
        """

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

    def process_lab_report(self, pdf_path: str) -> Dict[str, str]:
        pdf_text = self.extract_text_from_pdf(pdf_path)
        formatted_results = self.extract_bloodwork_results(pdf_text)
        return {
            "raw_text": pdf_text,
            "formatted_results": formatted_results,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    # Сохраняем файл временно
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        extractor = BloodworkExtractor(api_key)
        results = extractor.process_lab_report(file_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
