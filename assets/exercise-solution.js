<script>
// Exercise tabsets (Exercise / Hints / Solution): a click on the "Solution" tab
// loads the solution code into the exercise editor itself, so that it can be run
// with "Run Code" (and "Start Over" brings the exercise back). The page stays on
// the Exercise tab; the text of the Solution tab (function reference, remarks)
// is shown below the editor. If the editor is not ready yet, the tab behaves
// normally. Bootstrap handles tab clicks in the capture phase on `document`, so
// the interception is done one level up, on `window`.
(function () {
  function editorView(pane) {
    var content = pane.querySelector(".cm-content");
    return content && content.cmView ? content.cmView.view : null;
  }
  function solutionCode(pane) {
    var pre = pane.querySelector("pre code") || pane.querySelector("pre");
    if (!pre) return null;
    var clone = pre.cloneNode(true);
    clone.querySelectorAll(".code-annotation-anchor, .code-copy-button").forEach(function (e) { e.remove(); });
    return clone.textContent.replace(/[ \t]+\n/g, "\n").replace(/\s+$/, "") + "\n";
  }
  function setup(tabset) {
    var exLink = null, solLink = null;
    tabset.querySelectorAll(":scope > ul.nav-tabs a.nav-link").forEach(function (a) {
      var t = a.textContent.trim().toLowerCase();
      if (t === "exercise") exLink = a;
      if (t === "solution") solLink = a;
    });
    if (!exLink || !solLink) return;
    var exPane = tabset.querySelector(exLink.getAttribute("data-bs-target"));
    var solPane = tabset.querySelector(solLink.getAttribute("data-bs-target"));
    if (!exPane || !solPane || !exPane.querySelector(".exercise-cell")) return;
    solLink.title = "Load the solution into the exercise editor";
    solLink.__exsol = { exPane: exPane, solPane: solPane };
  }
  function loadSolution(info) {
    var view = editorView(info.exPane), code = solutionCode(info.solPane);
    if (!view || !code) return false;                   // editor not ready: normal tab switch
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    var note = info.exPane.querySelector(".solution-note");
    if (!note) { note = document.createElement("div"); note.className = "solution-note"; info.exPane.appendChild(note); }
    note.innerHTML = "";
    info.solPane.querySelectorAll(":scope > *").forEach(function (el) {
      if (el.matches("pre") || el.querySelector("pre")) return;   // the code itself is in the editor
      note.appendChild(el.cloneNode(true));
    });
    if (!note.hasChildNodes()) note.remove();           // nothing to show besides the code
    view.focus();
    return true;
  }
  window.addEventListener("click", function (e) {
    var a = e.target && e.target.closest ? e.target.closest("a.nav-link") : null;
    if (!a || !a.__exsol) return;
    if (loadSolution(a.__exsol)) { e.preventDefault(); e.stopPropagation(); }
  }, true);
  var style = document.createElement("style");
  style.textContent = ".solution-note{margin-top:.6em;padding:.5em .8em;border-left:3px solid #6A5ACD;background:#f4f2fb;font-size:.92em;}.solution-note p:last-child{margin-bottom:0;}";
  document.head.appendChild(style);
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".panel-tabset").forEach(setup);
  });
})();
</script>
