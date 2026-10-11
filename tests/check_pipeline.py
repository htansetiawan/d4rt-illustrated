"""Check the reader's path, exact pixel addresses and information-flow claims."""
import json
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_PIPELINE_URL', (root / 'index.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-pipeline13-'))
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'),
                                headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url, wait_until='load')
    page.wait_for_function('window.D4RTPipeline !== undefined')
    # Genie Figure 2 stages and the factorized causal attention path.
    assert page.locator('#ga-grid .ga-time').count() == 4
    page.locator('[data-ga=lam]').click()
    assert 'PHASE 2A' in page.locator('#ga-kicker').inner_text()
    page.locator('[data-ga=world]').click()
    assert 'PHASE 2B' in page.locator('#ga-kicker').inner_text()
    page.locator('[data-route=temporal]').click()
    assert 'never its future' in page.locator('#ga-route').inner_text()
    page.locator('#ga-time').fill('0')
    assert page.locator('#ga-time-label').inner_text() == '1 / 4'
    page.locator('[data-route=both]').click()
    assert 'without future leakage' in page.locator('#ga-route').inner_text()
    page.locator('[data-ga=tokenizer]').click()
    assert '1,024' in page.locator('.ga-numbers').inner_text()
    nodes = ['video', 'tubelet', 'embed', 'local', 'global', 'query', 'decoder', 'heads', 'train']
    for index, node in enumerate(nodes):
        page.locator(f'[data-pipe-stage="{node}"]').click()
        assert page.evaluate('D4RTPipeline.snapshot().stage') == index
        assert page.locator('[data-pipe-stage][aria-current="step"]').count() == 1
        assert page.locator('#pipe-title').inner_text()
        assert page.locator('#pipe-code').inner_text()

    # End-address arithmetic and pixel values are checked independently.
    page.locator('[data-pipe-stage="tubelet"]').click()
    for field, value in [('pair', 23), ('row', 15), ('col', 15)]:
        page.locator('#pipe-' + field).fill(str(value))
    result = page.evaluate('D4RTPipeline.snapshot()')
    assert result['index'] == 6143
    assert len(result['raw']) == 1536
    expected = []
    for t in (46, 47):
        for y in range(240, 256):
            for x in range(240, 256):
                expected.extend([228, 229, 225])  # background at this final patch
    assert result['raw'] == expected
    assert 'frames 46–47' in page.locator('#pipe-readout').inner_text()

    # Independently enumerate every temporal group, with spatial identity fixed.
    page.locator('[data-pipe-stage="local"]').click()
    for group in range(24):
        page.locator('#pipe-pair').fill(str(group))
        page.locator('#pipe-route').select_option('d4rt')
        assert page.evaluate('D4RTPipeline.snapshot().readable') == list(range(group*256, (group+1)*256))
        page.locator('#pipe-route').select_option('genie')
        assert page.evaluate('D4RTPipeline.snapshot().readable') == [i*256+255 for i in range(group+1)]
        page.locator('#pipe-route').select_option('composed')
        assert page.evaluate('D4RTPipeline.snapshot().readable') == list(range((group+1)*256))
    page.locator('[data-pipe-stage="global"]').click()
    assert page.evaluate('D4RTPipeline.snapshot().readable') == list(range(6144))

    # A separate source-time control changes the question, not the chosen tubelet.
    page.locator('[data-pipe-stage="query"]').click()
    before = page.evaluate('D4RTPipeline.snapshot()')
    page.locator('#pipe-tgt').fill('7')
    page.locator('#pipe-cam').fill('12')
    after = page.evaluate('D4RTPipeline.snapshot()')
    assert (after['index'], after['raw']) == (before['index'], before['raw'])
    assert 'frame 7' in page.locator('#pipe-readout').inner_text()
    assert 'frame 12' in page.locator('#pipe-readout').inner_text()

    page.locator('[data-pipe-stage="train"]').click()
    page.locator('#pipe-mode').select_option('infer')
    assert page.evaluate('D4RTPipeline.snapshot().stage') == 7
    assert page.locator('[data-pipe-stage="train"]').is_disabled()
    assert page.locator('#pipe-next').is_disabled()
    page.locator('#pipe-mode').select_option('train')
    page.locator('#pipe-next').click()
    assert page.evaluate('D4RTPipeline.snapshot().stage') == 8
    page.locator('[data-pipe-stage="video"]').click()
    page.locator('#pipe-play').click()
    page.wait_for_timeout(550)
    page.locator('[data-pipe-stage="tubelet"]').click()
    assert page.locator('#pipe-play').inner_text() == 'Play clip'

    page.locator('[data-pipe-stage="tubelet"]').click()
    for field, value in [('pair', 8), ('row', 7), ('col', 6)]:
        page.locator('#pipe-' + field).fill(str(value))
    for width in (1440, 768, 390):
        page.set_viewport_size({'width': width, 'height': 1100})
        for node in nodes:
            page.locator(f'[data-pipe-stage="{node}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, node)
            assert page.locator('#pipe-diagram').is_visible() == (node not in ('local', 'global'))
            assert page.locator('#pipe-attention').is_visible() == (node in ('local', 'global'))
            if node not in ('local', 'global'):
                bad = page.locator('#pipe-diagram text').evaluate_all('''es=>es.map(e=>({t:e.textContent,b:e.getBBox()}))
                    .filter(({b})=>b.x<0||b.y<0||b.x+b.width>1000||b.y+b.height>400).map(x=>x.t)''')
                assert not bad, (width, node, bad)
            if node in ('tubelet', 'local', 'train'):
                page.locator('.stage').screenshot(path=str(screens / f'{node}-{width}.png'))
    page.goto(url + '#encoder')
    page.wait_for_url('**/anatomy.html#encoder')
    assert page.locator('#dd-block-panel').count() == 1
    assert not errors, errors
    browser.close()
print(json.dumps({'result': 'pass', 'masks': 73, 'layouts': [1440, 768, 390],
                  'screenshots': str(screens), 'browser_errors': errors}, indent=2))
