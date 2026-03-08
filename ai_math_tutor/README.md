# AI Math Tutor — Image-Based Algebra Solver

This project is a Streamlit app that accepts an image of an algebra problem, converts it to LaTeX, solves symbolically and numerically using SymPy, checks for common mistakes, and presents a step-by-step solution card.

Key files:
- app.py - Streamlit UI
- utils/image_utils.py - image preprocessing helpers
- vision/ocr.py - OCR and optional LLM conversion
- solver/equation_solver.py - SymPy parsing and solving
- checker/mistake_checker.py - simple mistake heuristics
- prompts/latex_prompt.txt - prompt used for converting OCR text to LaTeX via LLM

Environment:
- Create a virtualenv and install packages from `requirements.txt`.
- Optionally set `OPENAI_API_KEY` to enable LLM-based LaTeX conversion.
- Optionally set `OPENAI_API_KEY` (OpenAI) or `GEMINI_API_KEY` (Google Gemini/PaLM) to enable LLM-based LaTeX conversion.

Security note: Do NOT paste API keys into public places. Set them as environment variables instead.

Example for Windows CMD:
```
set GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Example PowerShell:
```
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

Run locally:
```
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes:
- The LLM conversion requires an OpenAI API key and `openai` installed. If not available, the system uses heuristic OCR cleaning.
- For robust LaTeX->SymPy parsing, additional dependencies may be required.
