import math
import json
import os
import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

URL=os.environ.get('D4RT_CHECK_URL', (Path(__file__).resolve().parents[1]/'index.html').as_uri())
SCREENSHOTS=Path(tempfile.mkdtemp(prefix='d4rt-workbench-'))
CHROME=os.environ.get('CHROME_PATH') or shutil.which('google-chrome') or shutil.which('chromium')
if not CHROME: raise SystemExit('Install Chrome/Chromium or set CHROME_PATH.')
def slider(page,key,value):
    page.locator('#'+key).evaluate('(el,v)=>{el.value=String(v);el.dispatchEvent(new Event("input",{bubbles:true}));}',value)
def state(page):
    return page.evaluate('D4RTWorkbench.snapshot()')
def near(a,b):
    assert len(a)==len(b)
    assert max(abs(x-y) for x,y in zip(a,b))<1e-10,(a,b)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100},reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.goto(URL,wait_until='load')
    page.wait_for_selector('#wb-patch-svg image')
    assert len(state(page)['features'])==6
    first=state(page)['features']
    slider(page,'wb-scalar',1535)
    near(first,state(page)['features'])
    assert 'channel B' in page.locator('#wb-pixel-address').inner_text()
    for k,v in [('wb-pair',23),('wb-column',15),('wb-row',15)]:slider(page,k,v)
    assert 'Token 6143' in page.locator('#wb-token-address').inner_text()
    assert 'Frame 48, pixel (255, 255), channel B' in page.locator('#wb-pixel-address').inner_text()
    assert page.locator('#tube-t').input_value()=='23'
    assert max(abs(x-y) for x,y in zip(first,state(page)['features']))>.01
    slider(page,'tube-t',0)
    assert page.locator('#wb-pair').input_value()=='0'
    assert '1–2' in page.locator('#wb-pair-out').inner_text()
    page.locator('#wb-tab-pixels').focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('#wb-tab-encoder').get_attribute('aria-selected')=='true'
    assert page.locator('#wb-layer-bars i').count()==40
    for op in ['norm','qkv','attention','residual','mlp','output']:
        page.locator('#wb-operation').select_option(op)
        assert len(page.locator('#wb-op-shape').inner_text())>20
    page.locator('[data-wb-mask=global]').click()
    assert '[6144,6144]' in page.locator('#wb-block-svg').text_content()
    page.locator('[data-wb-mask=local]').click()
    assert '24 × [256,256]' in page.locator('#wb-block-svg').text_content()
    page.locator('[data-wb-path=temporal]').click()
    assert 'no direct edge' in page.locator('#wb-path-copy').inner_text()
    page.locator('[data-wb-path=composed]').click()
    assert 'A → B: spatial; B → Q: causal temporal' in page.locator('#wb-route-svg').text_content()
    page.locator('#wb-tab-encoder').focus()
    page.keyboard.press('End')
    assert page.locator('#wb-tab-labels').get_attribute('aria-selected')=='true'
    initial=state(page)
    slider(page,'wb-cam',11)
    changed=state(page)
    assert changed['target']!=initial['target']
    assert changed['visibility']==initial['visibility']
    near(changed['query'][:4],initial['query'][:4])
    slider(page,'wb-src',11)
    shifted=state(page)
    near(changed['target'],shifted['target'])
    assert changed['query'][:2]!=shifted['query'][:2]
    slider(page,'wb-tgt',0)
    assert shifted['target']!=state(page)['target']
    for frame in range(12):
        slider(page,'wb-src',frame);slider(page,'wb-tgt',frame);slider(page,'wb-cam',frame)
        s=state(page)
        x,y,z=s['target'];near(s['query'][:2],[.5+.9*x/z,.5+.9*y/z])
    vis=[page.evaluate('D4RTWorkbench.visible('+str(t)+')') for t in range(12)]
    assert any(vis) and not all(vis),vis
    hidden=vis.index(False)
    slider(page,'wb-src',0);slider(page,'wb-tgt',hidden);slider(page,'wb-cam',0)
    assert 'Target visibility = 0' in page.locator('#wb-visibility-value').inner_text()
    page.locator('[data-wb-train="2"]').click()
    assert 'active, with target 0' in page.locator('#wb-training-copy').inner_text()
    page.locator('#wb-has-visibility').uncheck()
    assert 'Visibility loss: masked' in page.locator('#wb-training-copy').inner_text()
    page.locator('#wb-has-geometry').uncheck()
    page.locator('[data-wb-train="3"]').click()
    assert 'zero data-loss gradient' in page.locator('#wb-training-copy').inner_text()
    page.locator('#wb-has-geometry').check();page.locator('#wb-has-visibility').check()
    assert 'encoder through F' in page.locator('#wb-training-copy').inner_text()
    slider(page,'wb-train-step',2500)
    assert page.locator('#wb-lr-out').inner_text()=='1.00e-4'
    slider(page,'wb-train-step',500000)
    assert page.locator('#wb-lr-out').inner_text()=='1.00e-6'
    slider(page,'wb-train-step',0)
    assert page.locator('#wb-lr-out').inner_text()=='0.00e+0'
    # Every SVG text element must fit its own viewBox; intentional scrolling stays inside a panel.
    bounds=[]
    for width in [390,768,1440]:
        page.set_viewport_size({'width':width,'height':1000})
        for tab in ['pixels','encoder','labels']:
            page.locator('#wb-tab-'+tab).click()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,tab)
            for el in page.locator('.wb-panel:not([hidden]) svg text').all():
                b=el.evaluate('(el)=>{const b=el.getBBox(),v=el.ownerSVGElement.viewBox.baseVal;return {s:el.textContent,x:b.x,y:b.y,r:b.x+b.width,b:b.y+b.height,w:v.width,h:v.height}}')
                if b['x']<0 or b['y']<0 or b['r']>b['w'] or b['b']>b['h']: bounds.append(b)
            page.locator('#model-lifetime .wb').screenshot(path=str(SCREENSHOTS/f'{tab}-{width}.png'))
        page.locator('#model-lifetime').scroll_into_view_if_needed()
        page.screenshot(path=str(SCREENSHOTS/f'page-{width}.png'))
    assert not bounds,bounds
    broken=page.evaluate('''[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href')).filter(h=>h.length>1&&!document.getElementById(h.slice(1)))''')
    assert not broken,broken
    assert not errors,errors
    print(json.dumps({'url':URL,'status':'passed','viewportWidths':[390,768,1440],'checks':['projection invariance and pixel endpoints','cross-linked tubelet controls','encoder operations and tensor domains','Genie composed routes','query coordinate transforms','occlusion vs missing labels','gradient routes','learning-rate endpoints','keyboard tabs','SVG bounds','page overflow','internal links','JS exceptions'],'targetVisibilityByFrame':vis}))
    browser.close()
