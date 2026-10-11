"""Verify shape arithmetic, connected controls, stage permalinks and layout."""
import json
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_TRACE_URL', (root / 'index.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-tensor-trace15-'))
keys = ['video', 'tubelet', 'embed', 'local', 'global', 'query', 'decoder', 'heads', 'train']
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'),
                                headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1200}, reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    for i, key in enumerate(keys):
        page.goto(url + '#walk-' + key)
        page.wait_for_function('window.D4RTTensorTrace !== undefined')
        assert page.evaluate('D4RTPipeline.snapshot().stage') == i
        assert page.evaluate('D4RTTensorTrace.snapshot().key') == key
        assert page.locator('#trace-rows tr').count() >= 3
        assert page.locator('#trace-stage-link').get_attribute('href') == '#walk-' + key
    page.locator('[data-pipe-stage="tubelet"]').click()
    for field, value in [('pair',23),('row',15),('col',15)]:
        page.locator('#pipe-'+field).fill(str(value))
    assert page.evaluate('D4RTTensorTrace.snapshot().index') == 6143
    assert 'Frames 46–47; x 240–255, y 240–255' in page.locator('#trace-rows').inner_text()
    page.locator('[data-pipe-stage="decoder"]').click()
    before=page.evaluate('D4RTPipeline.snapshot()')
    for queries in (1,2048,65536):
        page.locator('#trace-queries').select_option(str(queries))
        s=page.evaluate('D4RTTensorTrace.snapshot()')
        assert s['queries'] == queries
        assert s['localScores'] == 24 * (16 * 16)**2
        assert s['globalScores'] == (24 * 16 * 16)**2
        assert s['decoderScores'] == queries * 24 * 16 * 16
        assert f'{queries * 6144:,} scores' in page.locator('#pipe-readout').inner_text()
        assert page.evaluate('D4RTPipeline.snapshot().raw') == before['raw']
        page.locator('[data-pipe-stage="heads"]').click()
        assert f'[{queries:,}, 13]' in page.locator('#trace-rows').inner_text()
        page.locator('[data-pipe-stage="decoder"]').click()
    page.locator('[data-pipe-stage="train"]').click()
    assert page.evaluate('D4RTTensorTrace.snapshot().queries') == 2048
    assert 'Mean-depth normalization' in page.locator('#trace-meaning').inner_text()
    page.locator('#pipe-mode').select_option('infer')
    page.locator('.anatomy-shortcuts a[href="#walk-train"]').click()
    assert page.locator('#pipe-mode').input_value() == 'train'
    assert page.evaluate('D4RTTensorTrace.snapshot().key') == 'train'
    for width in (1440,768,390):
        page.set_viewport_size({'width':width,'height':1200})
        for key in ('tubelet','local','decoder','train'):
            page.locator(f'[data-pipe-stage="{key}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width,key)
        page.locator('[data-pipe-stage="decoder"]').click()
        page.locator('.tensor-trace').screenshot(path=str(screens/f'trace-{width}.png'))
    assert not errors,errors
    browser.close()
print(json.dumps({'result':'pass','stages':9,'query_counts':[1,2048,65536],
                  'widths':[1440,768,390],'screenshots':str(screens),'errors':errors},indent=2))
