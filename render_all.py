"""Build the complete Data Science website, slide HTML and PDF downloads.

Run with the Python environment used for the course:
    python render_all.py
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent
RENDER_ENV = os.environ.copy()
PYTHON_BIN_DIR = str(Path(sys.executable).resolve().parent)
RENDER_ENV['PATH'] = os.pathsep.join([PYTHON_BIN_DIR, RENDER_ENV.get('PATH', '')])
RENDER_ENV['QUARTO_PYTHON'] = sys.executable
# The explicit PDF stage below replaces Quarto's automatic post-render hook.
RENDER_ENV['DS_SKIP_SLIDE_PDF'] = '1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-compress', action='store_true', help='Skip PDF compression; keep the usual PDF filenames.')
    args = parser.parse_args()
    print('\n[1/2] Rendering website pages and presentations...', flush=True)
    subprocess.run(['quarto', 'render'], cwd=str(PROJECT_DIR), env=RENDER_ENV, check=True)
    print('\n[2/2] Checking slide resources and exporting PDFs...', flush=True)
    pdf_command = [sys.executable, str(PROJECT_DIR / 'slides2pdf.py'), '--from-rendered']
    if args.no_compress:
        pdf_command.append('--no-compress')
    subprocess.run(pdf_command,
                   cwd=str(PROJECT_DIR), env=RENDER_ENV, check=True)
    print('\nBuild complete. All downloadable PDFs match the rendered presentations.', flush=True)
    print('Publish the finished website without re-rendering with:')
    print('quarto publish gh-pages --no-render')

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
