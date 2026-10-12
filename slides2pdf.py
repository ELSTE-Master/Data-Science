"""Build the website before exporting its Revealjs decks to PDF.

Run `quarto render` for a complete build (the post-render hook invokes this
script), or `python slides2pdf.py --from-rendered` to export existing HTML.
"""
import argparse
from html.parser import HTMLParser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path
import shutil
from socketserver import ThreadingMixIn
import subprocess
import sys
import tempfile
import threading
from urllib.parse import quote, unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parent
DECKS = {
    'class1_intro': ('slides/class1_intro.html', 'class1/class_1_slides_intro.pdf'),
    'class1_pandas': ('slides/class1_pandas.html', 'class1/class_1_slides_pandas.pdf'),
    'class2_EDA': ('slides/class2_EDA.html', 'class2/class_2_slides_EDA.pdf'),
    'class_3_slides': ('class3/class_3_slides.html', 'class3/class_3_slides.pdf'),
    'class_4_slides': ('class4/class_4_slides.html', 'class4/class_4_slides.pdf'),
}

class LocalResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('img', 'script', 'iframe'):
            for key in ('src', 'data-src'):
                if attrs.get(key):
                    self.urls.append(attrs[key])
        if tag == 'link' and attrs.get('rel') in ('stylesheet', 'icon'):
            self.urls.append(attrs.get('href', ''))
        if attrs.get('data-background-image'):
            self.urls.append(attrs['data-background-image'])

def validate(output, names):
    missing = []
    for name in names:
        relative, _ = DECKS[name]
        html = output / relative
        if not html.is_file():
            missing.append(relative)
            continue
        parser = LocalResources()
        parser.feed(html.read_text(encoding='utf-8'))
        for url in parser.urls:
            if not url or urlsplit(url).scheme or url.startswith('//'):
                continue
            target = unquote(urlsplit(urljoin('/' + relative, url)).path).lstrip('/')
            if target and not (output / target).is_file():
                missing.append(relative + ' -> ' + target)
    if missing:
        raise RuntimeError('Missing rendered resources:\n' + '\n'.join(sorted(set(missing))))
    print('Checked slide HTML, images, scripts, styles and embedded pages.', flush=True)

class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

def export(output, names, compress=True):
    decktape = shutil.which('decktape')
    if not decktape:
        raise RuntimeError('DeckTape is required. Install it with: npm install -g decktape')
    ghostscript = shutil.which('gs') if compress else None
    if compress and not ghostscript:
        raise RuntimeError('Ghostscript is required for compression; install it or use --no-compress.')
    # Directory support on SimpleHTTPRequestHandler needs Python 3.7+. Keep the
    # hook compatible with the older Python installed on some teaching machines.
    class RootHandler(Handler):
        def translate_path(self, path):
            relative = unquote(urlsplit(path).path).lstrip('/')
            resolved = (output / relative).resolve()
            if output.resolve() not in resolved.parents and resolved != output.resolve():
                return str(output / '__invalid_path__')
            return str(resolved)
    server = Server(('127.0.0.1', 0), RootHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for name in names:
            html, pdf = DECKS[name]
            destination = output / pdf
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Replace the downloadable PDF only after DeckTape has succeeded.
            with tempfile.TemporaryDirectory(prefix='slide-pdf-') as temporary:
                generated = Path(temporary) / (name + '.pdf')
                command = [decktape, 'reveal', '--load-pause', '1500', '--pause', '500']
                chrome = os.environ.get('DECKTAPE_CHROME_PATH')
                if chrome:
                    command += ['--chrome-path', chrome]
                command += ['http://127.0.0.1:{}/{}'.format(server.server_port, quote(html, safe='/')), str(generated)]
                print('Exporting ' + name + ' -> ' + pdf, flush=True)
                subprocess.run(command, cwd=str(ROOT), check=True)
                if not generated.is_file() or generated.stat().st_size < 100:
                    raise RuntimeError('DeckTape did not create a valid PDF: ' + name)
                if compress:
                    compressed = Path(temporary) / (name + '-compressed.pdf')
                    subprocess.run([ghostscript, '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/ebook',
                                    '-dNOPAUSE', '-dQUIET', '-dBATCH', '-dDetectDuplicateImages',
                                    '-o', str(compressed), str(generated)], check=True)
                    if not compressed.is_file() or compressed.stat().st_size < 100:
                        raise RuntimeError('Compression did not create a valid PDF: ' + name)
                    generated = compressed
                shutil.copy2(str(generated), str(destination))
                # Also keep a PDF next to each HTML deck for direct downloads.
                adjacent = output / str(Path(html).with_suffix('.pdf'))
                if adjacent != destination:
                    shutil.copy2(str(generated), str(adjacent))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decks', nargs='*', metavar='DECK', help='Optional deck names: ' + ', '.join(DECKS))
    parser.add_argument('--from-rendered', action='store_true', help='Export existing website HTML without re-rendering.')
    parser.add_argument('--post-render', action='store_true', help='Quarto hook: run only after a full website build.')
    parser.add_argument('--check', action='store_true', help='Check rendered resources without creating PDFs.')
    parser.add_argument('--no-compress', action='store_true', help='Skip PDF compression; keep the usual PDF filenames.')
    args = parser.parse_args()
    unknown = set(args.decks) - set(DECKS)
    if unknown:
        parser.error('Unknown decks: ' + ', '.join(sorted(unknown)))
    if args.post_render and (os.environ.get('DS_SKIP_SLIDE_PDF') or os.environ.get('QUARTO_PROJECT_RENDER_ALL') != '1'):
        return
    names = args.decks or list(DECKS)
    output = ROOT / os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '_site')
    if not (args.from_rendered or args.post_render or args.check):
        env = dict(os.environ, DS_SKIP_SLIDE_PDF='1')
        # A full build exports every deck so all downloads match the new HTML.
        subprocess.run(['quarto', 'render'], cwd=str(ROOT), env=env, check=True)
        names = list(DECKS)
    validate(output, names)
    if not args.check:
        export(output, names, compress=not args.no_compress)

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
