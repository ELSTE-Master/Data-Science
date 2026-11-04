"""Execute the original-style deck's Run Code slide via a localhost server.

Requires Playwright, Brave and internet for the same Pyodide runtime as Class 3.
No remote hosting, upload, or publication is performed.
"""
import functools
import http.server
import json
import os
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'checks'
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith('/course/'):
            self.send_error(404);return
        self.path=self.path[len('/course'):]
        super().do_GET()
    def log_message(self,*args):pass
BOUNDS='''() => {
 const s=Reveal.getCurrentSlide(),b=s.getBoundingClientRect(),bad=[];
 const check=(r,t)=>{if(r.width&&r.height&&(r.left<b.left-3||r.right>b.right+3||r.top<0||r.bottom>Math.min(innerHeight,b.top+Reveal.getConfig().height*Reveal.getScale())+3))bad.push(t)};
 const w=document.createTreeWalker(s,NodeFilter.SHOW_TEXT);
 while(w.nextNode()){let n=w.currentNode;if(!n.textContent.trim()||n.parentElement.closest('script,style,.cm-measure,.cm-announced,aside.notes'))continue;
 if(!n.parentElement.getClientRects().length||getComputedStyle(n.parentElement).visibility==='hidden')continue;
 let r=document.createRange();r.selectNodeContents(n);for(let x of r.getClientRects())check(x,n.textContent.slice(0,60));}
 s.querySelectorAll('img,canvas,.exercise-editor-btn-run-code').forEach(e=>check(e.getBoundingClientRect(),e.tagName));
 return {slide:Reveal.getIndices().h+1,bad};
}'''
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
report=[]
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/usr/bin/brave-browser',headless=True,args=['--no-sandbox'])
  for width,height in [(1440,900),(1024,768)]:
   page=browser.new_page(viewport={'width':width,'height':height});errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(f'http://127.0.0.1:{server.server_port}/course/class4/class_4_slides.html',wait_until='domcontentloaded')
   page.wait_for_function('typeof Reveal!=="undefined" && Reveal.isReady()')
   assert page.evaluate('Reveal.getSlides().length')==41
   page.evaluate('Reveal.slide(6)')
   print('Loading Python runtime',width,flush=True)
   try:
    page.wait_for_function('!!document.querySelector(".exercise-editor-btn-run-code:not(.disabled)")',timeout=240000)
   except Exception:
    print('STATUS',page.locator('body').inner_text()[-3000:],errors,flush=True);raise
   cases=[(8,'summary',['707.8019','27.6596','0.447']),
                              (6,'code',['707.80','27.66']),
                              (11,'multiple-code',['576.32','28.37','25.10','-18.62']),
                              (13,'multiple-summary',['576.3177','28.3747','25.1036','-18.6177','0.672']),
                              (19,'quadratic-code',['Training R² = 0.698']),
                              (20,'quadratic-summary',['794.7026','16.7772','1.1636','0.698']),
                              (24,'degree10-code',['Training R² = 0.505']),
                              (32,'aic-code',['17281.1','16553.7','16448.2']),
                              (35,'cv-code',['115.8','90.1','90.0']),
                              (37,'interval-code',['1200.0','1193.1','1206.9','974.7','1425.3'])]
   selected=os.environ.get('CPX_LIVE_CASES')
   if selected:
    names=set(selected.split(','));assert names <= {case[1] for case in cases}
    cases=[case for case in cases if case[1] in names]
   print('Checking Python cells:',[case[1] for case in cases],flush=True)
   for index,name,wanted in cases:
    page.evaluate('(i)=>Reveal.slide(i)',index)
    cell=page.locator('section.present .exercise-cell');assert cell.count()==1
    assert cell.locator('.cm-content').is_visible()
    assignment={'multiple-code':'multiple = smf.ols','multiple-summary':'multiple = smf.ols','quadratic-code':'quadratic = smf.ols','quadratic-summary':'quadratic = smf.ols','degree10-code':'poly10 = smf.ols','aic-code':'fits =','cv-code':'fold = cv_folds(train)'}.get(name,'simple = smf.ols')
    assert assignment in cell.inner_text()
    cell.locator('.exercise-editor-btn-run-code').click()
    page.wait_for_function('''wanted=>{const e=document.querySelector('section.present .exercise-cell');return e&&wanted.every(v=>e.innerText.includes(v));}''',arg=wanted,timeout=60000)
    assert 'Traceback' not in cell.inner_text()
    if name.endswith('summary'):
     assert cell.locator('.cell-output table').count()==2
     page.wait_for_function("document.querySelector('section.present .exercise-cell').innerText.includes('The coefficient standard errors assume constant error variance and uncorrelated errors across observations.')")
     assert 'covariance matrix of the errors is correctly specified' not in cell.inner_text()
     assert 'Start with' not in page.locator('section.present').inner_text()
    page.wait_for_timeout(300)
    if name in ['quadratic-summary','aic-code','cv-code','interval-code']:
     current=page.evaluate(BOUNDS)
     assert not current['bad'],current
    if name=='interval-code':
     assert page.evaluate('''()=>{const s=Reveal.getCurrentSlide(),top=s.querySelector('.interval-code-assumptions').getBoundingClientRect().top;return [...s.querySelectorAll('.exercise-cell,.interval-output-key')].every(e=>e.getBoundingClientRect().bottom<top-3)}'''), 'Interval code or key overlaps the assumptions'
    page.screenshot(path=str(OUT/f'live-{name}-{width}.png'))
    content=cell.locator('.cm-content')
    original=content.evaluate('(e)=>e.cmView.view.state.doc.toString()')
    content.evaluate('''(e)=>{let v=e.cmView.view;v.dispatch({changes:{from:v.state.doc.length,insert:"\\nprint('EDITED_OK')"}});}''')
    cell.locator('.exercise-editor-btn-run-code').click()
    page.wait_for_function('''()=>document.querySelector('section.present .exercise-cell').innerText.includes('EDITED_OK') && [...document.querySelectorAll('section.present .cell-output-stdout')].some(e=>e.innerText.includes('EDITED_OK'))''',timeout=60000)
    content.evaluate('''(e,code)=>{let v=e.cmView.view;v.dispatch({changes:{from:0,to:v.state.doc.length,insert:code}});}''',original)
    cell.locator('.exercise-editor-btn-run-code').click()
    page.wait_for_function('''wanted=>{const e=document.querySelector('section.present .exercise-cell');return wanted.every(v=>e.innerText.includes(v))&&!e.innerText.includes('EDITED_OK');}''',arg=wanted,timeout=60000)
    print('PASS: Run Code and editable output',name,width,flush=True)
   for i in range(41):
    page.evaluate('(i)=>Reveal.slide(i)',i);page.wait_for_timeout(150)
    bounds=page.evaluate(BOUNDS);bounds['viewport']=[width,height];report.append(bounds)
   print('PASS: executed and edited Python',width,flush=True)
   page.close()
  browser.close()
finally:
 server.shutdown()
 (OUT/'live-code.json').write_text(json.dumps(report,indent=2))
assert len(report)==82 and not any(r['bad'] for r in report),report
print('PASS: all live layouts, selected Python cells, editing and repeated execution at both viewports.')
