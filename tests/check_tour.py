"""Interactive invariants and independent finite-difference checks for the tour."""
import json
import math
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_TOUR_URL', (root / 'anatomy-tour.html').as_uri())
artifacts = Path(tempfile.mkdtemp(prefix='d4rt-tour-'))

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'), headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url + '#learning', wait_until='load')
    assert page.evaluate('location.hash') == '#learning'
    page.wait_for_function('window.D4RTTourChecks !== undefined')
    snapshot = lambda: page.evaluate('D4RTTourChecks.snapshot()')
    initial = snapshot()
    assert initial['parameterCount'] == 249
    assert initial['patches'] == [24] * 16
    gradients = page.evaluate('D4RTTourChecks.gradientCheck()')
    for item in gradients:
        assert math.isfinite(item['analytic'])
        assert abs(item['analytic']) > 1e-10, item
        assert math.isclose(item['analytic'], item['numeric'], rel_tol=1e-4, abs_tol=1e-7), item
    page.locator('#train-one').click()
    trained = snapshot()
    assert trained['iteration'] == 1
    assert trained['encoder'] != initial['encoder']
    assert trained['loss'] < initial['loss']
    page.locator('#reset').click()
    assert snapshot() == initial
    page.locator('#freeze').check()
    page.locator('#train-one').click()
    frozen = snapshot()
    assert frozen['encoder'] == initial['encoder']
    assert frozen['head'] != initial['head']
    assert '0.000e+0' in page.locator('#encoder-update').inner_text()
    page.locator('#freeze').uncheck()
    page.locator('#reset').click()
    page.locator('#train-many').click()
    page.wait_for_function('D4RTTourChecks.snapshot().iteration === 20 && !document.getElementById("train-many").disabled')
    final = snapshot()
    assert final['loss'] < initial['loss']
    page.locator('#learning').scroll_into_view_if_needed()
    page.screenshot(path=str(artifacts / 'learning-desktop.png'))
    assert page.locator('[data-stage]').count() == 6
    page.locator('#tab-1').click()
    for control, value in [('pair', 23), ('col', 15), ('row', 15)]:
        page.locator('#' + control).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}', str(value))
    assert '6,143' in page.locator('#address').inner_text()
    assert 'Frame 48' in page.locator('#diagram').text_content()
    page.locator('#tab-2').click()
    for domain, count in [('local', 16), ('global', 64), ('temporal', 3), ('composed', 48)]:
        page.locator('#domain').select_option(domain)
        assert page.locator('#domain-count').inner_text().startswith(f'{count} / 64')
    page.locator('#tab-2').focus()
    page.keyboard.press('End')
    assert page.locator('#tab-5').get_attribute('aria-selected') == 'true'
    page.locator('#direction').click()
    assert 'BACKWARD' in page.locator('#diagram').text_content()
    assert page.locator('#direction').get_attribute('aria-pressed') == 'true'
    page.locator('#tab-5').focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('#tab-0').get_attribute('aria-selected') == 'true'
    for width in [390, 768, 1440]:
        page.set_viewport_size({'width': width, 'height': 1000})
        for i in range(6):
            page.locator(f'#tab-{i}').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, i)
            outside = page.locator('#diagram').evaluate('svg=>[...svg.querySelectorAll("text")].filter(t=>{const b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>960||b.y+b.height>380}).map(t=>t.textContent)')
            assert not outside, outside
        page.locator('#tab-2').click()
        page.locator('#domain').select_option('global')
        page.locator('#tour').scroll_into_view_if_needed()
        page.screenshot(path=str(artifacts / f'encoder-{width}.png'))
    assert not errors, errors
    print(json.dumps({'result': 'passed', 'gradients': gradients, 'loss_before': initial['loss'], 'loss_after_20_steps': final['loss'], 'viewports': [390, 768, 1440], 'screenshots': str(artifacts), 'browser_errors': errors}, indent=2))
    browser.close()
