"""Streamlit front-end for AI Math Tutor - Image-Based Algebra Solver"""

import streamlit as st
from PIL import Image
import io
import os
import logging

from utils.image_utils import preprocess_for_ocr, to_bytes
from vision.ocr import OCREngine, llm_convert_to_latex
from solver.equation_solver import parse_latex_to_sympy, solve_equation, generate_steps
from checker.mistake_checker import detect_mistakes
import warnings

# Suppress PyTorch DataLoader pin_memory warnings if no accelerator is found
warnings.filterwarnings("ignore", category=UserWarning, message=".*pin_memory.*")

logging.getLogger().setLevel(logging.INFO)

st.set_page_config(page_title="AI Math Tutor", layout="wide")


def main():
    st.title("AI Math Tutor — Image-Based Algebra Solver")

    st.sidebar.header("Input")
    uploaded = st.sidebar.file_uploader(
        "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    cam = st.sidebar.camera_input("Or capture from camera")
    use_llm = st.sidebar.checkbox(
        "Use LLM for LaTeX conversion (requires OPENAI_API_KEY)", value=False
    )

    img = None
    if uploaded is not None:
        img = Image.open(uploaded)
    elif cam is not None:
        img = Image.open(io.BytesIO(cam.getvalue()))

    if img is None:
        st.info("Upload or capture an image of an algebra problem to begin.")
        if st.button("Generate sample images"):
            st.markdown("See README for sample generator usage")
        return

    st.sidebar.image(img, caption="Input image", use_column_width=True)

    with st.spinner("Preprocessing image..."):
        pre = preprocess_for_ocr(img)

    ocr_engine = OCREngine()
    with st.spinner("Running OCR..."):
        raw_text = ocr_engine.extract_text(pre)
        math_expr = ocr_engine.extract_math(pre)

    st.subheader("OCR Output (raw)")
    st.text_area("Raw OCR text", value=raw_text, height=120)

    st.subheader("Cleaned math expression (heuristic)")
    edited_expr = st.text_input("Detected expression", value=math_expr)

    latex = None
    if use_llm:
        with st.spinner("Converting to LaTeX via LLM..."):
            latex = llm_convert_to_latex(edited_expr, image=to_bytes(pre))
    else:
        latex = edited_expr.strip()

    st.subheader("Detected LaTeX / Expression")
    try:
        st.latex(latex)
    except Exception:
        st.markdown(f"`{latex}`")

    # Solve
    try:
        sym = parse_latex_to_sympy(latex)
    except Exception as e:
        st.error(f"Failed to parse expression: {e}")
        return

    with st.spinner("Solving..."):
        result = solve_equation(sym)
        steps = generate_steps(sym, latex)
        mistakes = detect_mistakes(latex, sym)

    # UI card
    st.subheader("Solution Card")
    st.markdown("- **Problem (LaTeX):**")
    try:
        st.latex(latex)
    except Exception:
        st.write(latex)

    st.markdown("- **Steps:**")
    for i, (title, content) in enumerate(steps, start=1):
        st.markdown(f"**Step {i} — {title}:**")
        # If this is a LaTeX snippet, try render
        try:
            st.latex(content)
        except Exception:
            st.write(content)

    st.markdown("- **Final Answer:**")
    sols = result.get("solutions")
    if sols is None:
        st.write("No solution found.")
    else:
        st.write(sols)

    if mistakes:
        st.error("Potential mistakes detected:")
        for m in mistakes:
            st.write(f"- {m}")
    else:
        st.success("No obvious mistakes detected.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Environment variables:\n- OPENAI_API_KEY (optional for OpenAI LLM LaTeX conversion)\n- GEMINI_API_KEY (optional for Google Gemini/PaLM LLM LaTeX conversion)"
    )


if __name__ == "__main__":
    main()
