"""Verify the visual model route and mask semantics, including future-frame access."""
import json
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_CUTAWAY_URL', (root / 'anatomy.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-cutaway12-'))
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'),
                                headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440, 'height':1100}, reduced_motion='reduce')
    errors=[]
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url, wait_until='load')
    page.wait_for_function('window.D4RTCutaways !== undefined')
    assert page.evaluate('D4RTCutaways.snapshot().node') == 'video'
    nodes=['video','patch','local','global','scene','query','decoder','heads','loss']
    for node in nodes:
        page.locator(f'[data-cut-stage="{node}"]').click()
        assert page.evaluate('D4RTAtlas.snapshot().node') == node
        assert page.locator('[data-cut-stage][aria-current="step"]').count() == 1
        assert page.locator('#atlas-cutaway-svg').get_attribute('aria-label')
        assert page.locator('#atlas-cutaway-caption').inner_text()
    page.locator('#atlas-prev-detail').click()
    assert page.evaluate('D4RTCutaways.snapshot().node') == 'heads'
    page.locator('[data-cut-stage="patch"]').click()
    for field, v in [('group',23),('row',15),('col',15)]:
        page.locator('#atlas-tube-'+field).fill(str(v))
    assert 'Frames 46–47' in page.locator('#atlas-tube-index').inner_text()
    assert '6,143' in page.locator('#atlas-tube-index').inner_text()
    for node in ('local','global'):
        page.locator(f'[data-cut-stage="{node}"]').click()
        for token in range(12):
            page.locator(f'[data-cut-token="{token}"]').click()
            result=page.evaluate('D4RTCutaways.snapshot()')
            expected=[i for i in range(12) if node=='global' or i//4==token//4]
            assert result['readable']==expected, (node,token,result)
            page.locator('#atlas-comparison').check()
            expected=[i for i in range(12) if i%4==token%4 and i//4<=token//4]
            assert page.evaluate('D4RTCutaways.snapshot().readable')==expected
            page.locator('#atlas-comparison').uncheck()
        selected_text=page.locator('#atlas-cutaway-svg text').filter(has_text='11').evaluate_all("es=>es.filter(e=>e.textContent==='11').map(e=>getComputedStyle(e).fill)")
        assert selected_text==['rgb(255, 255, 255)'], selected_text
    # Inference omits the optimizer and label path.
    page.locator('[data-cut-stage="loss"]').click()
    page.locator('#atlas-mode').select_option('infer')
    assert page.evaluate('D4RTCutaways.snapshot().node')=='heads'
    assert page.locator('[data-cut-stage="loss"]').is_disabled()
    assert '8 / 8' in page.locator('#atlas-cutaway-step').inner_text()
    page.locator('#atlas-mode').select_option('train')
    for width in (390,768,1440):
        page.set_viewport_size({'width':width,'height':1100})
        for node in nodes:
            page.locator(f'[data-cut-stage="{node}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width,node)
            page.locator('.cutaway-figure').scroll_into_view_if_needed()
            # SVG text must fit the diagram's coordinate bounds.
            bad=page.locator('#atlas-cutaway-svg text').evaluate_all('''es=>es.map(e=>({
                t:e.textContent,b:e.getBBox()})).filter(({b})=>b.x<0||b.y<0||b.x+b.width>404||b.y+b.height>306).map(x=>x.t)''')
            assert not bad,(width,node,bad)
            if node in ('patch','local','loss'):
                page.screenshot(path=str(screens/f'{node}-{width}.png'))
    assert not errors,errors
    print(json.dumps({'result':'passed','checks':['9 visual blocks','48 selected-token masks',
        'last tubelet address','inference excludes training','back navigation','SVG label bounds',
        '390/768/1440 layouts'], 'screenshots':str(screens),'browser_errors':errors},indent=2))
    browser.close()
