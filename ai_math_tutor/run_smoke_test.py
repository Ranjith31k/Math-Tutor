"""Run quick smoke tests on sample images: OCR -> LaTeX -> SymPy solve."""
import os
from PIL import Image

from vision.ocr import OCREngine, llm_convert_to_latex
from utils.image_utils import preprocess_for_ocr
from solver.equation_solver import parse_latex_to_sympy, solve_equation


def run():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    if not os.path.exists(samples_dir) or not os.listdir(samples_dir):
        print("No samples found, generating sample images...")
        try:
            import sample_generator as sg
            sg.generate(samples_dir)
        except Exception as e:
            print("Failed to generate samples:", e)
            return

    ocr = OCREngine()

    for fname in sorted(os.listdir(samples_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(samples_dir, fname)
        print("\n=== Sample:", path)
        img = Image.open(path)
        pre = preprocess_for_ocr(img)

        raw = ocr.extract_text(pre)
        print("OCR raw:\n", raw)

        latex = llm_convert_to_latex(raw, image=None)
        print("Converted LaTeX / expression:\n", latex)

        try:
            sym = parse_latex_to_sympy(latex)
            res = solve_equation(sym)
            print("Solutions:", res.get("solutions"))
        except Exception as e:
            print("Parsing/solving failed:", e)


if __name__ == '__main__':
    run()
