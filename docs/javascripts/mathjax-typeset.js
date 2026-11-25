// Re-typeset on every page swap (Material "navigation.instant" is an SPA).
document$.subscribe(function () {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
