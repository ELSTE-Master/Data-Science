<script>
// Clarify only the summary's explanatory note; retain the original Python code and numerical output.
(function () {
  const original = 'Standard Errors assume that the covariance matrix of the errors is correctly specified.';
  const clearer = 'The coefficient standard errors assume constant error variance and uncorrelated errors across observations.';
  function install() {
    document.querySelectorAll('section.regression-summary').forEach(function (slide) {
    function clarify() {
      slide.querySelectorAll('.cell-output-pyodide').forEach(function (output) {
        output.childNodes.forEach(function (node) {
          if (node.nodeType === Node.TEXT_NODE && node.textContent.includes(original)) {
            node.textContent = node.textContent.replace(original, clearer);
          }
        });
      });
    }
    clarify();
    new MutationObserver(clarify).observe(slide, {childList:true, subtree:true});
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install);
  else install();
})();
</script>
