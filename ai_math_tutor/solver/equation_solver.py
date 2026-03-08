"""Symbolic parsing and solving using SymPy."""
from sympy import Eq, simplify, solve, sympify, latex as sympy_latex
from sympy.parsing.latex import parse_latex
import re
import logging

logger = logging.getLogger(__name__)


def parse_latex_to_sympy(latex_str: str):
    """Attempt to parse LaTeX string into a SymPy Eq or expression.

    Returns tuple (left, right) for equations or (expr, None) for expressions.
    """
    try:
        # Try parsing as equation with '='
        if "=" in latex_str:
            left_s, right_s = latex_str.split("=", 1)
            try:
                left = parse_latex(left_s.strip())
                right = parse_latex(right_s.strip())
                return Eq(left, right)
            except Exception:
                # Fallback: try naive sympify after cleaning
                pass
        # Try parse single expression
        try:
            expr = parse_latex(latex_str)
            return expr
        except Exception:
            # fallback to heuristics below
            pass
    except Exception as e:
        logger.exception("LaTeX parsing failed: %s", e)
        # Fallback: try sympify naive replacements and heuristics

    # Heuristic cleanup for common OCR outputs / simple LaTeX
    s = latex_str or ""
    s = s.replace('^', '**')
    s = s.replace('X', 'x')
    s = s.replace('\u00bd', '1/2')
    # insert multiplication between number and variable or between ) and variable
    s = re.sub(r"(\d)\s*(?=[A-Za-z(])", r"\1*", s)
    s = re.sub(r"(\))\s*(?=[A-Za-z0-9(])", r"\1*", s)
    s = s.strip()
    if not s:
        raise ValueError("Empty expression")

    try:
        if "=" in s:
            left_s, right_s = s.split('=', 1)
            left_s = left_s.replace(')', '') if left_s.count(')') > left_s.count('(') else left_s
            left_s = left_s.replace('(', '') if left_s.count('(') > left_s.count(')') else left_s
            right_s = right_s.replace(')', '') if right_s.count(')') > right_s.count('(') else right_s
            right_s = right_s.replace('(', '') if right_s.count('(') > right_s.count(')') else right_s

            left = sympify(left_s.strip())
            right = sympify(right_s.strip())
            return Eq(left, right)
            
        s = s.replace(')', '') if s.count(')') > s.count('(') else s
        s = s.replace('(', '') if s.count('(') > s.count(')') else s
        expr = sympify(s)
        return expr
    except Exception as e:
        logger.exception("Heuristic sympify failed: %s", e)
        raise


def solve_equation(sym):
    """Solve an equation or expression. Returns dict with steps and solutions."""
    try:
        if isinstance(sym, Eq):
            # pick symbol(s)
            syms = list(sym.free_symbols)
            if not syms:
                return {"solutions": [], "error": "No symbol to solve for"}
            solutions = solve(sym, syms[0])
            return {"solutions": solutions}

        # If expression equals zero? try solve(expr, sym)
        syms = list(sym.free_symbols)
        if syms:
            solutions = solve(sym, syms[0])
            return {"solutions": solutions}

        return {"solutions": []}
    except Exception as e:
        logger.exception("Solving failed: %s", e)
        return {"solutions": [], "error": str(e)}


def generate_steps(sym_input, latex_str=None):
    """Generate a minimal step-by-step solution using Gemini LLM or SymPy transformations.

    Returns list of tuples (title, latex_step).
    """
    import os
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key and latex_str:
        try:
            import google.generativeai as genai
            import json
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "solver_prompt.txt")
            with open(prompt_path, "r") as f:
                prompt_text = f.read().replace("<<EQUATION>>", latex_str)
            
            response = model.generate_content(prompt_text)
            
            if response.text:
                res = response.text.strip()
                if res.startswith("```"):
                    lines = res.split("\n")
                    if len(lines) > 2:
                        res = "\n".join(lines[1:-1])
                        
                parsed = json.loads(res.strip())
                steps = []
                for i, step in enumerate(parsed):
                    steps.append((step.get("explanation", f"Step {i+1}"), step.get("result_latex", "")))
                
                if steps:
                    return steps
        except Exception as e:
            logger.exception("LLM step generation failed: %s", e)

    # Fallback heuristc
    steps = []
    try:
        if isinstance(sym_input, Eq):
            L = sym_input.lhs
            R = sym_input.rhs
            steps.append(("Original", sympy_latex(sym_input)))
            # Simplify both sides
            Ls = simplify(L)
            Rs = simplify(R)
            if Ls != L or Rs != R:
                steps.append(("Simplified", sympy_latex(Eq(Ls, Rs))))
            # Move everything to left and solve
            expr = simplify(L - R)
            steps.append(("Rearranged", sympy_latex(expr) + " = 0"))
            sol = solve(sym_input)
            steps.append(("Solve", str(sol)))
            return steps

        # For pure expressions
        steps.append(("Original", sympy_latex(sym_input)))
        simp = simplify(sym_input)
        if simp != sym_input:
            steps.append(("Simplified", sympy_latex(simp)))
        syms = list(sym_input.free_symbols)
        if syms:
            sol = solve(sym_input, syms[0])
            steps.append(("Solve", str(sol)))
        return steps
    except Exception as e:
        logger.exception("Generating steps failed: %s", e)
        return [("Original", str(sym_input)), ("Error", str(e))]
