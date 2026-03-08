"""Quick smoke test: generate samples, run preprocessing, OCR, parsing, solving."""
import os
from PIL import Image

from utils.image_utils import preprocess_for_ocr
from vision.ocr import OCREngine, llm_convert_to_latex
from solver.equation_solver import parse_latex_to_sympy, solve_equation, generate_steps
from checker.mistake_checker import detect_mistakes


def run_on_image(path, use_llm=False):
    print('\n---')
    print('Image:', path)
    try:
        img = Image.open(path)
    except Exception as e:
        print('Failed to open image:', e)
        return

    pre = preprocess_for_ocr(img)
    print('Preprocessing done')

    ocr = OCREngine()
    raw = ocr.extract_text(pre)
    math_expr = ocr.extract_math(pre)
    print('OCR raw:', raw)
    print('Extracted math:', math_expr)

    latex = math_expr.strip()
    if use_llm:
        latex = llm_convert_to_latex(raw, image=None)
    print('LaTeX/expr:', latex)

    try:
        sym = parse_latex_to_sympy(latex)
    except Exception as e:
        print('Parse failed:', e)
        return

    res = solve_equation(sym)
    print('Solutions:', res.get('solutions'))

    steps = generate_steps(sym)
    print('Steps:')
    for t, c in steps:
        print(' -', t, ':', c)

    mistakes = detect_mistakes(latex, sym)
    print('Mistakes:', mistakes)


def main():
    base = os.path.join(os.path.dirname(__file__), '')
    samples_dir = os.path.join(base, 'samples')
    if not os.path.isdir(samples_dir):
        print('Samples not found, generating...')
        try:
            from sample_generator import generate
            generate(samples_dir)
        except Exception as e:
            print('Failed to generate samples:', e)
            return

    imgs = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
    if not imgs:
        print('No sample images found')
        return

    for p in imgs:
        run_on_image(p, use_llm=False)


if __name__ == '__main__':
    main()
"""Smoke test: generate samples, run OCR + parsing + solving on them.

Run from the `ai_math_tutor` folder.
"""
import os
from sample_generator import generate
from PIL import Image
from utils.image_utils import preprocess_for_ocr
from vision.ocr import OCREngine, llm_convert_to_latex
from solver.equation_solver import parse_latex_to_sympy, solve_equation, generate_steps


def run_on_samples(folder='samples'):
    if not os.path.exists(folder):
        generate(folder)

    engine = OCREngine()
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.png'):
            continue
        path = os.path.join(folder, fn)
        print('\n====', path)
        img = Image.open(path)
        pre = preprocess_for_ocr(img)
        raw = engine.extract_text(pre)
        print('OCR raw:', raw)
        # Do NOT call LLM in smoke test by default
        latex = raw.strip()
        print('LaTeX/expr:', latex)
        try:
            sym = parse_latex_to_sympy(latex)
            print('Parsed sympy:', sym)
            sol = solve_equation(sym)
            print('Solutions:', sol.get('solutions'))
            steps = generate_steps(sym)
            print('Steps:')
            for t, c in steps:
                print('-', t, ':', c)
        except Exception as e:
            print('Parsing/solving failed:', e)


if __name__ == '__main__':
    run_on_samples()
