"""Check static layouts and the canonical exported PDF; requires Playwright and Chromium."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
SITE=ROOT.parents[1]/'_site'
HTML=SITE/'class4/class_4_slides_static.html'
PDF=SITE/'class4/class_4_slides.pdf'
OUT=Path(os.environ.get('CPX_CHECK_OUTPUT',str(ROOT/'checks')));OUT.mkdir(parents=True,exist_ok=True)
OUT.mkdir(exist_ok=True)
BOUNDS='''() => {
 const s=Reveal.getCurrentSlide(), b=s.getBoundingClientRect(), bad=[];
 const footer=document.querySelector('.footer');
 const bottom=Math.min(window.innerHeight,b.top+Reveal.getConfig().height*Reveal.getScale(),footer?footer.getBoundingClientRect().top-4:Infinity);
 const check=(r,t)=>{if(r.width && r.height && (r.left<b.left-2||r.right>b.right+2||r.top<0||r.bottom>bottom+2))bad.push({text:t,rect:{left:r.left,right:r.right,top:r.top,bottom:r.bottom},slide:{left:b.left,right:b.right,top:b.top},bottom})};
 const w=document.createTreeWalker(s,NodeFilter.SHOW_TEXT);
 while(w.nextNode()){const n=w.currentNode;if(!n.textContent.trim()||n.parentElement.closest('script,style,mjx-assistive-mml'))continue;
 const r=document.createRange();r.selectNodeContents(n);for(const x of r.getClientRects())check(x,n.textContent.slice(0,70));}
 s.querySelectorAll('img,mjx-container').forEach(e=>{if(!e.closest('mjx-assistive-mml'))check(e.getBoundingClientRect(),e.tagName)});
 return {slide:Reveal.getIndices().h+1,bad,math:s.querySelectorAll('mjx-container').length};
}'''
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('BROWSER_EXECUTABLE','/usr/bin/brave-browser'),args=['--no-sandbox'])
 report=[]
 for width,height in [(1440,900),(1024,768)]:
  page=browser.new_page(viewport={'width':width,'height':height})
  page.goto(HTML.as_uri(),wait_until='networkidle')
  page.wait_for_function('typeof Reveal!=="undefined" && Reveal.isReady()')
  page.evaluate('async()=>{await MathJax.startup.promise;await document.fonts.ready;}')
  count=page.evaluate('Reveal.getSlides().length');assert count==41,count
  assert page.locator('mjx-merror').count()==0
  assert 'With thanks to' not in page.locator('body').inner_text()
  assert ':::' not in page.locator('.slides').inner_text()
  for i in range(count):
   page.evaluate('(i)=>Reveal.slide(i)',i);page.wait_for_timeout(120)
   assert page.evaluate('[...Reveal.getCurrentSlide().querySelectorAll("img")].every(i=>i.complete && i.naturalWidth>0)')
   if i in [21,33,36]:
    assert page.evaluate('[...Reveal.getCurrentSlide().querySelectorAll("img")].every(i=>i.getBoundingClientRect().width>300&&i.getBoundingClientRect().height>200)')
   r=page.evaluate(BOUNDS);r['viewport']=[width,height];report.append(r)
   if width==1440:page.screenshot(path=str(OUT/f'slide-{i+1:02}.png'))
  page.close()
 browser.close()
(OUT/'layout.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert len(report)==82 and not any(r['bad'] for r in report)

# Verify exported PDF content too: browser bounds alone miss print pagination loss.
import subprocess
pdf_text = subprocess.check_output(['pdftotext', '-layout', str(PDF), '-'], text=True)
pdf_pages = [' '.join(page.split()) for page in pdf_text.split('\f') if page.strip()]
assert len(pdf_pages) == 41, len(pdf_pages)
for number, markers in {7: ['from cpx_lesson import load_data', 'simple = smf.ols'],
 9: ['707.8019', '27.6596', 'R-squared'],
 10: ['33.66', '0.001', 'rounded', 'causation'],
 11: ['44.7%', '0.447', 'test observations'],
 12: ['multiple = smf.ols', 'T_C ~ MgO_Cpx + Al2O3_Cpx + TiO2_Cpx', 'print(multiple.params.round(2))'],
 14: ['576.3177', '28.3747', '25.1036', '-18.6177', '0.672'],
 15: ['28.37', '25.10', '18.62', 'fixed'],
 16: ['Model diagnostics', 'unexplained', 'residual'],
 17: ['Squaring'],
 18: ['Training RMSE', 'coefficients'],
 19: ['10 coefficients', '0.698'],
 20: ['quadratic = smf.ols(formula, data=train).fit()', 'I(TiO2_Cpx**2)'],
 21: ['794.7026', '16.7772', '1.1636', '0.698', '0.781'],
 22: ['Model diagnostics', 'Full quadratic', 'Training residual', 'systematic patterns'],
 23: ['0.000', '0.447', '0.672', '0.698', 'zero', 'adjusted'],
 24: ['11 coefficients', '114.5', '108.6'],
 25: ['poly10 = smf.ols', 'range(1, 11)', 'lo, hi'],
 26: ['Simulated data'],
 27: ['George'],
 28: ['Training:', 'Test:'],
 29: ['arbitrary units'],
 30: ['TiO', 'Validation error', 'fitting mean'],
 31: ['159.4', '101.2', '80.0', '97.4', 'quadratic'],
 32: ['AIC', 'intercept', 'complexity'],
 33: ['832.9', '105.6', '0.0', 'm.aic'],
 34: ['Model selection',
      '512',
      '9 candidate terms',
      'Forward selection',
      'intercept-only',
      'lowest AIC',
      'cross-validation'],
 35: ['Round', 'Part', 'fitting'],
 36: ['115.8', '90.1', '90.0', 'cv_folds'],
 37: ['How uncertain',
      'conditional mean',
      'residual variation',
      '1,193',
      '1,207',
      '975',
      '1,425',
      'normally distributed',
      'natural crystals',
      'average'],
 38: ['Computing confidence',
      'prediction intervals',
      '97.5th percentile',
      'residual variance',
      'get_prediction',
      'summary_frame(alpha=0.05)',
      'obs_ci_lower',
      '1200'],
 39: ['Simulated teaching example', 'shark'],
 40: ['Shark incidents through the year', 'Summer', 'Jan', 'Dec', 'What do you think?'],
 41: ['common cause', 'Warmer beach weather', 'causation']}.items():
    for marker in markers:
        assert marker in pdf_pages[number-1], (number,marker)
assert all(word in pdf_pages[18] for word in ['Products','(interaction)'])
assert 'Training observations only.' not in pdf_pages[18]
assert 'Training fit only; predictive benefit remains untested.' not in pdf_pages[18]
assert 'Model revision and scientific limits' not in pdf_text
assert 'Conceptual example inspired by' not in pdf_pages[40]
print('PASS: 41 PDF pages, including reordered AIC/CV, test errors and uncertainty intervals.')
