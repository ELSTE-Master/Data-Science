"""
Python version of q_a_env.R: renders a multiple-choice question with
per-option feedback, styled by webex_2.css and handled by webex_2.js.

Usage in a Quarto page (jupyter engine):

    ```{python}
    #| echo: false
    #| output: asis
    render_q_a_env([
        {"value": "Wrong option", "feedback": "Why it is wrong."},
        {"value": "Right option", "feedback": "Yes!", "answer": True},
    ])
    ```

Math in options and feedback is written as \\( ... \\) (inline MathJax).
"""

import html
import itertools

_counter = itertools.count(1)


def render_q_a_env(opts):
    if sum(bool(o.get("answer")) for o in opts) != 1:
        raise ValueError("The question must have exactly one correct answer")

    qname = f"radio_q{next(_counter)}"
    labels = []
    for o in opts:
        # Option text is parsed as Markdown by pandoc: double the backslashes so
        # that \( ... \) reaches MathJax unchanged. The feedback is an HTML
        # attribute (not parsed), so it only needs HTML escaping.
        value = o["value"].replace("'", "&apos;").replace("\\", "\\\\")
        feedback = html.escape(o.get("feedback", ""), quote=True)
        name = "answer" if o.get("answer") else ""
        labels.append(
            f'<label><input type="radio" autocomplete="off" name="{qname}" '
            f'value="{name}" data-feedback="{feedback}"></input> <span>{value}</span></label>'
        )
    print(
        f"<div class='webex-radiogroup' id='{qname}'>"
        + "".join(labels)
        + "<div class='webex-feedback-box'></div></div>\n"
    )
