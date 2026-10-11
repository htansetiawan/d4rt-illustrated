"""Check architecture navigation and independently derived attention counts."""
import json
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_ATLAS_URL', (root / 'index.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-atlas-'))
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'),
        headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url, wait_until='load')
    page.wait_for_function('window.D4RTAtlas !== undefined')
    snapshot = lambda: page.evaluate('D4RTAtlas.snapshot()')
    original = snapshot()['counts']
    assert original == {'groups': 24, 'spatial': 256, 'tokens': 6144, 'scalars': 9437184,
        'local': 1572864, 'global': 37748736, 'temporal': 76800, 'cross': 12582912}, original
    # Temporal entries enumerated from an explicit same-address causal mask.
    for frames in (16, 32, 48, 96):
        page.locator('#atlas-frames').select_option(str(frames))
        for queries in (1, 256, 2048, 65536):
            page.locator('#atlas-queries').select_option(str(queries))
            result = snapshot()['counts']
            groups = frames // 2
            assert result['temporal'] == sum(256 for t in range(groups) for k in range(groups) if k <= t)
            assert result['tokens'] * 1536 == result['scalars']
            assert result['global'] == result['tokens'] ** 2
            assert result['cross'] == queries * result['tokens']
    page.locator('#atlas-queries').select_option('2048')
    doubled = snapshot()['counts']
    assert doubled['local'] == 2 * original['local']
    assert doubled['global'] == 4 * original['global']
    assert doubled['cross'] == 2 * original['cross']
    assert 'Counterfactual' in page.locator('#atlas-budget-note').inner_text()
    page.locator('#atlas-frames').select_option('48')
    assert 'reported D4RT training' in page.locator('#atlas-budget-note').inner_text()
    names = page.locator('[data-atlas]').evaluate_all('(es)=>es.map(e=>e.dataset.atlas)')
    assert len(names) == 9
    for name in names:
        page.locator(f'[data-atlas="{name}"]').click()
        assert snapshot()['node'] == name
        assert page.locator('[data-atlas][aria-pressed="true"]').count() == 1
        for field in ('title', 'shape', 'body', 'operation', 'weights', 'activation', 'lifetime'):
            assert page.locator('#atlas-' + field).inner_text()
        assert page.locator('#atlas-source').get_attribute('href').startswith('https://arxiv.org/html/')
    # Inference removes the training operation but retains the complete forward path.
    page.locator('#atlas-mode').select_option('infer')
    assert snapshot()['node'] == 'heads'
    assert page.locator('#atlas-loss').is_disabled()
    assert 'Forward only' in page.locator('#atlas-return').inner_text()
    page.locator('[data-atlas="scene"]').click()
    assert 'reuse F' in page.locator('#atlas-lifetime').inner_text()
    page.locator('#atlas-next').click()
    assert snapshot()['node'] == 'query'
    page.locator('#atlas-mode').select_option('train')
    assert page.locator('#atlas-loss').is_enabled()
    page.locator('[data-atlas="local"]').focus()
    page.keyboard.press('Enter')
    assert snapshot()['node'] == 'local'
    missing = page.locator('a[href^="#"]').evaluate_all('(es)=>es.map(e=>e.getAttribute("href")).filter(h=>h.length>1&&!document.getElementById(h.slice(1)))')
    assert not missing, missing
    for width in (390, 768, 1440):
        page.set_viewport_size({'width': width, 'height': 1100})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
        for area in ('model-atlas', 'atlas-counts'):
            page.locator('#' + area).scroll_into_view_if_needed()
            page.screenshot(path=str(screens / f'{area}-{width}.png'))
        arrow = page.locator('.atlas-between').bounding_box()
        assert arrow['width'] <= 21 and arrow['height'] <= 21, (width, arrow)
        for name in names:
            page.locator(f'[data-atlas="{name}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, name)
    assert not errors, errors
    print(json.dumps({'result': 'passed', 'checks': ['all 9 model blocks', 'training/inference paths',
        'keyboard activation', '16 attention budgets', 'independent causal-mask enumeration',
        'scaling relations', 'fragment links', '390/768/1440 layouts'], 'screenshots': str(screens),
        'browser_errors': errors}, indent=2))
    browser.close()
