"""Simple mistake detection heuristics for algebra problems.

This module inspects the original equation and the sympy-processed forms
to suggest common mistakes like distribution errors, sign errors, and
transposition issues.
"""
from sympy import simplify, Eq, expand
import logging

logger = logging.getLogger(__name__)


def detect_mistakes(latex_str: str, sym_obj=None):
    """Return a list of detected mistakes using Gemini LLM or fallback heuristics."""
    mistakes = []
    
    import os
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            import json
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash") # Using 1.5-flash or 2.5-flash
            
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "mistake_detector_prompt.txt")
            with open(prompt_path, "r") as f:
                prompt_text = f.read().replace("<<EQUATION>>", latex_str)
            
            response = model.generate_content(prompt_text)
            
            if response.text:
                res = response.text.strip()
                if res.startswith("```"):
                    lines = res.split("\n")
                    if len(lines) > 2:
                        res = "\n".join(lines[1:-1])
                if res.lower() != "none" and res != "[]":
                    parsed = json.loads(res.strip())
                    if isinstance(parsed, list):
                        mistakes.extend(parsed)
            return mistakes
        except Exception as e:
            logger.exception("LLM mistake detection failed: %s", e)

    # Fallback heuristic
    try:
        if isinstance(sym_obj, Eq):
            L = sym_obj.lhs
            R = sym_obj.rhs
            # Check distribution: if parentheses on L or R, compare expansion
            if any(str(x).find("(") >= 0 for x in [L, R]):
                try:
                    if expand(L) != L and simplify(expand(L) - L) != 0:
                        mistakes.append("Possible distribution mistake on left side (check expansion)")
                    if expand(R) != R and simplify(expand(R) - R) != 0:
                        mistakes.append("Possible distribution mistake on right side (check expansion)")
                except Exception:
                    pass

            # Sign/transposition: move all terms left and see if simplified matches
            try:
                combined = simplify(L - R)
                s = str(combined)
                if s.count("-") > s.count("+") + 2:
                    mistakes.append("Check sign handling during transposition")
            except Exception:
                pass

        else:
            try:
                if sym_obj:
                    s = simplify(sym_obj)
            except Exception:
                mistakes.append("Expression parsing may be incorrect; check OCR/LaTeX conversion")

    except Exception as e:
        logger.exception("Mistake detection failed: %s", e)
    return mistakes
