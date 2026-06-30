"""
utils/salary_parser.py

Extracts structured salary data from uploaded salary slips.
Supports: PDF, PNG, JPG, JPEG files.
Uses Groq's vision/text capabilities for extraction.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import pdfplumber
from groq import Groq


def _parse_json_safe(text: str) -> Dict:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return {}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF salary slip using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def parse_salary_with_groq(raw_text: str) -> Dict[str, Any]:
    """
    Send extracted salary slip text to Groq and parse out structured fields.
    Falls back to regex heuristics if Groq returns unparseable output.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a payroll document parser.
Extract all financial information from this salary slip text and return ONLY valid JSON.
No preamble, no markdown fences, just the JSON object.

Required fields (use 0 if not found):
{{
  "employee_name": "",
  "employee_id": "",
  "month_year": "",
  "company_name": "",
  "gross_salary": 0,
  "basic_salary": 0,
  "hra": 0,
  "allowances": 0,
  "net_salary": 0,
  "tax": 0,
  "tds": 0,
  "pf": 0,
  "epf": 0,
  "esi": 0,
  "professional_tax": 0,
  "other_deductions": 0,
  "total_deductions": 0,
  "monthly_expenses": 0,
  "current_savings": 0,
  "currency": "INR"
}}

Salary Slip Text:
{raw_text[:3000]}"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
    )

    parsed = _parse_json_safe(response.choices[0].message.content)

    # Regex fallback for common Indian salary slip patterns
    if not parsed.get("net_salary"):
        patterns = {
            "net_salary":    r"(?:net\s+(?:salary|pay|amount))[^\d]*(\d[\d,]+)",
            "gross_salary":  r"(?:gross\s+(?:salary|pay))[^\d]*(\d[\d,]+)",
            "tds":           r"(?:tds|income\s+tax)[^\d]*(\d[\d,]+)",
            "pf":            r"(?:pf|provident\s+fund|epf)[^\d]*(\d[\d,]+)",
        }
        text_lower = raw_text.lower()
        for field, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match and not parsed.get(field):
                try:
                    parsed[field] = float(match.group(1).replace(",", ""))
                except ValueError:
                    pass

    return parsed


def parse_salary_from_image(file_path: str) -> Dict[str, Any]:
    """
    Parse salary data from an image file using Groq's vision model.
    Encodes the image as base64 and asks Groq to extract fields.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_type_map.get(ext, "image/jpeg")

    response = client.chat.completions.create(
        model="llava-v1.5-7b-4096-preview",  # Groq's vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": """Extract salary information from this salary slip image.
Return ONLY valid JSON with these fields (use 0 if not found):
{
  "employee_name": "", "company_name": "", "month_year": "",
  "gross_salary": 0, "basic_salary": 0, "hra": 0, "allowances": 0,
  "net_salary": 0, "tax": 0, "tds": 0, "pf": 0, "epf": 0,
  "professional_tax": 0, "other_deductions": 0, "total_deductions": 0,
  "currency": "INR"
}""",
                    },
                ],
            }
        ],
        max_tokens=600,
    )

    return _parse_json_safe(response.choices[0].message.content)


def parse_salary_file(file_path: str) -> Dict[str, Any]:
    """
    Main entry point. Detects file type and dispatches to the right parser.
    Returns structured salary dict.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
        if raw_text.strip():
            return parse_salary_with_groq(raw_text)
        # If PDF had no extractable text (scanned), fall through to image path
        # Convert first page to image would go here in production
        return {"error": "PDF has no extractable text. Please upload a text-based PDF or image."}

    elif ext in (".png", ".jpg", ".jpeg"):
        return parse_salary_from_image(file_path)

    else:
        return {"error": f"Unsupported file type: {ext}. Please upload PDF, PNG, or JPG."}
