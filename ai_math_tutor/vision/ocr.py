"""OCR and LLM conversion helpers.

This module provides a fallback pipeline:
- OCR via EasyOCR (preferred) or pytesseract
- LLM call wrapper for converting OCR output to clean LaTeX (optional)
"""
import os
import logging
from PIL import Image
import re
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
    _have_easyocr = True
except Exception:
    _have_easyocr = False

try:
    import pytesseract
    _have_tesseract = True
except Exception:
    _have_tesseract = False

try:
    import openai
    _have_openai = True
except Exception:
    _have_openai = False
    import requests

from utils.image_utils import to_bytes

class OCREngine:
    def __init__(self, lang_list=["en"]):
        self.lang_list = lang_list
        if _have_easyocr:
            try:
                self.reader = easyocr.Reader(lang_list, gpu=False)
            except Exception:
                self.reader = None
        else:
            self.reader = None

    def extract_text(self, pil_image):
        """Extract raw text from the image using available OCR engines."""
        try:
            if self.reader:
                # easyocr returns list of (bbox, text, conf)
                result = self.reader.readtext(np.asarray(pil_image))
                texts = [r[1] for r in result]
                raw = "\n".join(texts)
                return raw

            if _have_tesseract:
                txt = pytesseract.image_to_string(pil_image)
                return txt

            return ""
        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return ""

    def extract_math(self, pil_image):
        """Return a cleaned math expression derived from OCR output."""
        raw = self.extract_text(pil_image)
        try:
            math = _extract_math_from_text(raw)
            if math:
                return math
        except Exception:
            logger.exception("Math extraction failed; falling back to raw OCR")
        return raw


def _clean_ocr_text(ocr_text: str) -> str:
    """Basic cleaning heuristics from OCR to LaTeX-like string."""
    if not ocr_text:
        return ""
    text = ocr_text.strip()
    # Replace common OCR artefacts
    text = text.replace("×", "*")
    text = text.replace("—", "-")
    text = re.sub(r"[^0-9a-zA-Z\s=+\-*/()\\.^\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_math_from_text(ocr_text: str) -> str:
    """Heuristic: pick the line or substring that looks most like a math expression.

    Strategy:
    - Split into lines, for each line keep only math-allowed characters
    - Score by length and number of math symbols, pick best candidate
    """
    if not ocr_text:
        return ""
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=+-*/^()[]{} \\._%")
    math_lines = []
    for line in ocr_text.splitlines():
        if not line.strip():
            continue
        # remove common leading garbage words
        # keep characters that are typical in math
        filtered = ''.join(ch for ch in line if ch in allowed)
        # remove words like 'Algebra', 'Problems' that may remain
        filtered = re.sub(r"\b(Algebra|Problems|Problem|Exercises)\b", "", filtered, flags=re.IGNORECASE)
        filtered = filtered.strip()
        # compact multiple spaces
        filtered = re.sub(r"\s+", " ", filtered)
        if len(filtered) >= 1:
            math_lines.append(filtered)

    if not math_lines:
        return _clean_ocr_text(ocr_text)

    # score lines: longer and with more math symbols wins
    def score(s):
        math_symbols = sum(1 for ch in s if ch in '=+-*/^()[]{}\\')
        return len(s) + math_symbols * 3

    best = max(math_lines, key=score)
    # final cleanup: replace unicode fraction ½ etc.
    best = best.replace('\u00bd', '1/2')
    # Try to extract the most math-like contiguous substring (e.g. from '0 Algebra 3+18' -> '3+18')
    try:
        # compact spaces and try full string without spaces first
        nospace = re.sub(r"\s+", "", best)
        if re.search(r"[+\-*/^=]", nospace) and re.search(r"\d", nospace):
            return nospace

        # otherwise find candidate contiguous substrings and prefer those with both digits and operators
        candidates = re.findall(r"[0-9A-Za-z\)\]\{\}\\\^\/\*+\-\(\)]+", best)
        def is_math_like(c):
            return bool(re.search(r"[+\-*/^=]", c) and re.search(r"\d", c))

        math_cands = [c for c in candidates if is_math_like(c)]
        if math_cands:
            math_best = max(math_cands, key=lambda x: (len(x), sum(1 for ch in x if ch in '+-*/^=')))
            return math_best
    except Exception:
        pass

    return best


def llm_convert_to_latex(ocr_text: str, image=None, model="gemini-2.5-flash") -> str:
    """Convert OCR output into clean LaTeX using available LLM providers.

    Order of preference:
    1. Google Gemini (via `GEMINI_API_KEY` env var)
    2. Local heuristic cleaning
    """
    cleaned_text = _clean_ocr_text(ocr_text)
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key:
        try:
            import google.generativeai as genai
            import io
            from PIL import Image
            
            genai.configure(api_key=gemini_key)
            vision_model = genai.GenerativeModel(model)
            
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "latex_prompt.txt")
            with open(prompt_path, "r") as f:
                prompt = f.read().replace("<<OCR_TEXT>>", cleaned_text)
            
            contents = [prompt]
            if image is not None:
                img = Image.open(io.BytesIO(image))
                contents.append(img)
            
            # Configure request limits and timeout handling
            config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=512)
            response = vision_model.generate_content(contents, generation_config=config, request_options={"timeout": 30})
            
            if response.text:
                # Remove markdown code formatting if present
                res = response.text.strip()
                if res.startswith("```"):
                    lines = res.split("\n")
                    if len(lines) > 2:
                        res = "\n".join(lines[1:-1])
                return res.strip()
            
        except Exception as e:
            logger.exception("Gemini conversion attempt failed: %s", e)

    # Fallback to cleaned text
    return cleaned_text
