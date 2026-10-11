"""Browser contract for the connected model anatomy and gradient path."""
from pathlib import Path
import json
import os
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
screens = Path(tempfile.mkdtemp(prefix='d4rt-model-map14-'))
url = os.environ.get('D4RT_MAP_URL', (root/'index.html').as_uri())
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH','/usr/bin/google-chrome'),
                                headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1200}, reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_function('window.D4RTModelMap !== undefined')
    assert page.evaluate('D4RTModelMap.snapshot()') == {'selected':'encoder','mode':'forward'}
    for name in ('tubelet','encoder','memory','query','decoder','heads','loss'):
        page.locator(f'[data-model-node="{name}"]').click()
        assert page.evaluate('D4RTModelMap.snapshot().selected') == name
        assert page.locator('[data-model-node][aria-pressed="true"]').count() == 1
        assert page.locator('#map-detail-code').inner_text()
        target=page.locator('#map-detail-lab').get_attribute('href')
        path,anchor=target.split('#')
        assert f'id="{anchor}"' in (root/path).read_text(), target
        for mode in ('backward','forward'):
            page.locator(f'[data-map-mode="{mode}"]').click()
            assert page.locator('#map-detail-gradient').is_visible() == (mode=='backward')
            assert set(page.locator('.map-arrow').all_inner_texts()) == ({'↑ ∂L'} if mode=='backward' else {'↓'})
    page.locator('[data-model-node="encoder"]').click()
    page.locator('[data-map-mode="backward"]').click()
    for width in (1440,768,390):
        page.set_viewport_size({'width':width,'height':1200})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('#model-map').screenshot(path=str(screens/f'anatomy-{width}.png'))
        # Native buttons retain keyboard activation.
        page.locator('[data-model-node="decoder"]').focus()
        page.keyboard.press('Enter')
        assert page.evaluate('D4RTModelMap.snapshot().selected')=='decoder'
        assert 'point queries' in page.locator('#map-detail-code').inner_text()
    assert not errors,errors
    browser.close()
print(json.dumps({'result':'pass','components':7,'modes':2,'widths':[1440,768,390],
                  'screenshots':str(screens),'errors':errors},indent=2))
