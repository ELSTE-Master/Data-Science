# Building the course

From the main Data Science folder, activate the Python environment used for the course, then run:

```sh
python render_all.py
```

Like Outils informatique’s `render_all.py`, this entry point runs the build stages sequentially with the current Python interpreter and stops on any error. It processes the current website pages and appendices first, then the Revealjs slides. After all HTML is ready, its PDF stage checks local slide resources and uses DeckTape to produce PDFs. It serves the complete `_site` directory temporarily on localhost so slide images, shared scripts and embedded lesson pages load from the same root.

The PDFs are written to the existing download locations under `_site/class1`, `_site/class2`, `_site/class3` and `_site/class4`, with copies alongside the slide HTML. Publish `_site` only after the entire command succeeds. Source PDFs and `_past-years` are not modified.

DeckTape must be available on the command path (`npm install -g decktape`). The script uses DeckTape’s bundled Chromium browser; `DECKTAPE_CHROME_PATH` can select another Chromium/Chrome executable. The course’s Python and R dependencies remain necessary to execute its examples.

`quarto render` also runs the same PDF export through a post-render hook after a complete build. The entry point avoids running that hook twice.

For editing, use `quarto preview` from the main folder. Partial renders/previews do not regenerate PDFs. Run a complete build before distributing the course.

To re-export PDFs without rebuilding HTML:

```sh
python slides2pdf.py --from-rendered
```

To re-export one existing deck:

```sh
python slides2pdf.py --from-rendered class1_intro
```

To check slide image, stylesheet, script and iframe paths without exporting:

```sh
python slides2pdf.py --check
```

Stop preview processes before the final build so they do not render the same project concurrently.

PDF compression uses Ghostscript by default. To keep the original PDF quality,
run `python render_all.py --no-compress` or
`python slides2pdf.py --from-rendered --no-compress`. Both keep the usual PDF
filenames, replacing the previous downloads.

Class 3: edit `slides/class_3_slides.qmd` (the live deck), then run
`python slides/make_static_slides.py` before building to refresh
`slides/class_3_slides_static.qmd` (the PDF source). Shared Class 3 images
are copied into `slides/img/class3`; the originals remain in `class3/img`.
