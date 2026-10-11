"""Behavioral and numerical checks for the anatomy-first homepage."""
import json
import math
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_DEEP_URL', (root / 'anatomy.html').as_uri())
artifacts = Path(tempfile.mkdtemp(prefix='d4rt-deep-'))

def slider(page, name, value):
    page.locator('#'+name).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}', str(value))

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'), headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440, 'height':1100}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url, wait_until='load')
    page.wait_for_function('window.D4RTDeepDive !== undefined')
    snapshot = lambda: page.evaluate('D4RTDeepDive.snapshot()')
    initial = snapshot()
    assert len(initial['patch']) == 1536
    assert len(initial['features']) == 6
    slider(page, 'dd-scalar', 1535)
    assert snapshot()['features'] == initial['features'], 'inspecting a scalar must not change the token'
    for key,value in [('dd-time',23),('dd-col',15),('dd-row',15)]: slider(page,key,value)
    assert 'Token 6,143' in page.locator('#dd-token-address').inner_text()
    assert 'Frame 48, pixel (255, 255), channel B' in page.locator('#dd-pixel-address').inner_text()
    last = snapshot()
    assert last['patch'][-1] == 178, last['patch'][-1]  # ground blue = 195 - round(510/30)
    assert last['features'] != initial['features']
    # Independent linear-projection calculation from the displayed raw values.
    for channel, actual in enumerate(last['features']):
        expected = sum(v/255*math.cos((k+1)*(channel+1)*.013) for k,v in enumerate(last['patch']))/math.sqrt(1536)
        assert math.isclose(actual,expected,abs_tol=1e-10)
    matrices={}
    page.locator('[data-dd-token="6"]').click()
    for domain,count,total in [('local',4,64),('global',16,256),('temporal',2,40)]:
        page.locator('#dd-domain').select_option(domain)
        s=snapshot(); matrices[domain]=s['attention']
        assert sum(v>0 for v in s['attention'][6])==count
        assert sum(v>0 for row in s['attention'] for v in row)==total
        assert all(math.isclose(sum(row),1,abs_tol=1e-12) for row in s['attention'])
        assert all(math.isfinite(v) for row in s['output'] for v in row)
        assert page.locator('#dd-matrix .masked').count()==256-total
    assert all(matrices['temporal'][6][j]==0 for j in range(16) if j not in (2,6))
    assert matrices['global'][6][15]>0, 'later observed frames are available in global attention'
    assert matrices['local']!=matrices['global']
    # Merely selecting another inspector row must leave the computed block unchanged.
    before=snapshot()['output'];page.locator('[data-dd-token="15"]').click()
    assert snapshot()['output']==before
    for i in range(6):
        page.locator(f'#dd-op-{i}').click()
        assert page.locator('#dd-op-title').inner_text()
        assert page.locator('#dd-block-panel').get_attribute('aria-labelledby')==f'dd-op-{i}'
    page.locator('#dd-op-5').focus();page.keyboard.press('Home')
    assert page.locator('#dd-op-0').get_attribute('aria-selected')=='true'
    page.keyboard.press('ArrowRight')
    assert page.locator('#dd-op-1').get_attribute('aria-selected')=='true'
    for i in [0,1,2,4,5]:
        page.locator(f'[data-jump="{i}"]').click()
        assert page.locator(f'#tab-{i}').get_attribute('aria-selected')=='true'
    page.locator('#genie-route').click()
    assert page.locator('#domain').input_value()=='composed'
    assert page.locator('#domain-count').inner_text().startswith('48 / 64')
    links=page.locator('a[href^="#"]').evaluate_all('(as)=>as.map(a=>a.getAttribute("href")).filter(h=>h.length>1&&!document.getElementById(h.slice(1)))')
    assert not links,links
    for width in [390,768,1440]:
        page.set_viewport_size({'width':width,'height':1100})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
        for section in ['overview','tokenizer','encoder','training']:
            page.locator('#'+section).scroll_into_view_if_needed()
            page.screenshot(path=str(artifacts/f'{section}-{width}.png'))
        assert page.locator('#dd-domain').bounding_box()['width']<=width
    # Previous published workbench URLs still lead to their substantive content.
    page.goto(url.split('#')[0]+'#model-lifetime',wait_until='load')
    page.wait_for_url('**/reference.html#model-lifetime')
    assert page.locator('#wb-tab-pixels').count()==1
    assert not errors,errors
    print(json.dumps({'result':'passed','checks':['tubelet source/address/projection','attention causality and normalization','inspector invariance','keyboard controls','blueprint navigation','legacy deep link','responsive layout'], 'viewports':[390,768,1440], 'screenshots':str(artifacts),'browser_errors':errors},indent=2))
    browser.close()
