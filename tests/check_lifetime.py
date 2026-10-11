"""Verify the connected computation with independent math and perturbations."""
import json
import math
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_LIFETIME_URL', (root / 'anatomy.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-lifetime-'))
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-9),(a,b)
def slider(page, name, value):
    page.locator('#'+name).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}',str(value))
def changed(a,b):
    return [i for i,(x,y) in enumerate(zip(a,b)) if any(abs(u-v)>1e-12 for u,v in zip(x,y))]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH','/usr/bin/google-chrome'),headless=True,args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1100},reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url,wait_until='load')
    page.wait_for_function('window.D4RTLifetime !== undefined')
    trace=lambda:page.evaluate('D4RTMiniature.trace()')
    initial=trace()
    assert initial['parameterCount']==249
    for actual,expected in zip(initial['targets'],[.35,.75]):close(actual,expected)
    # Pixel -> patch mapping and learned projection are independently recomputed.
    for i,patch in enumerate(initial['patches']):
        g,y,x=i//4,(i%4)//2,i%2
        expected=[initial['pixels'][2*g+dt][2*y+dy][2*x+dx][c]
            for dt in range(2) for dy in range(2) for dx in range(2) for c in range(3)]
        assert patch==expected
        for j in range(4):
            projected=sum(patch[k]*initial['weights']['patch'][k][j] for k in range(24))
            close(projected,initial['projected'][i][j])
            close(projected+.08*math.sin((i+1)*(j+1)),initial['x'][i][j])
    # Rebuild the attention probabilities and value reads, including local mask.
    for name in ['local','global','cross']:
        att=initial[name]
        for i,q in enumerate(att['q']):
            scores=[sum(x*y for x,y in zip(q,k))/2 for k in att['k']]
            live=[j for j in range(16) if name!='local' or j//4==i//4]
            exps=[math.exp(scores[j]-max(scores[k] for k in live)) if j in live else 0 for j in range(16)]
            weights=[x/sum(exps) for x in exps]
            for a,b in zip(weights,att['a'][i]):close(a,b)
            for j in range(4):close(sum(weights[k]*att['v'][k][j] for k in range(16)),att['read'][i][j])
    close(initial['loss'],sum((p-t)**2 for p,t in zip(initial['pred'],initial['targets']))/2)
    # A single pixel changes 1 projection, 4 local outputs, all 16 global outputs.
    slider(page,'life-red',.95)
    edited=trace()
    assert changed(initial['x'],edited['x'])==[8]
    assert changed(initial['l'],edited['l'])==[8,9,10,11]
    assert len(changed(initial['f'],edited['f']))==16
    assert edited['pred']!=initial['pred']
    assert edited['queries']==initial['queries']  # edited pixel is outside source crop
    # Query independence is forward independence, even though training shares weights.
    before=edited
    slider(page,'life-time',7)
    queried=trace()
    assert queried['f']==before['f']
    assert queried['pred'][0]==before['pred'][0]
    assert queried['pred'][1]!=before['pred'][1]
    assert queried['targets'][0]==before['targets'][0]
    # Source appearance takes a second path into Q, as well as changing the encoder.
    page.evaluate('D4RTMiniature.editPixel(0,2,0,.9)')
    source=trace()
    assert source['queries'][0]!=queried['queries'][0]
    assert source['cross']['q'][0]!=queried['cross']['q'][0]
    # Input derivatives include both paths; verify with finite differences.
    gradients=[]
    for t,y,x in [(4,1,1),(0,2,0)]:
        g=page.evaluate('(p)=>D4RTMiniature.pixelGradientCheck(...p)',[t,y,x])
        assert abs(g['analytic'])>1e-10,g
        assert math.isclose(g['analytic'],g['numeric'],rel_tol=1e-4,abs_tol=1e-7),g
        gradients.append(g)
    page.locator('#life-reset').click()
    assert trace()['f']==initial['f']
    assert trace()['pixels']==initial['pixels']
    page.locator('#life-train').click()
    trained=trace()
    assert trained['loss']<initial['loss']
    assert trained['f']!=initial['f']
    assert trained['pixels']==initial['pixels']  # optimizer never updates pixels
    page.locator('#life-reset').click()
    page.locator('#life-freeze').check()
    page.locator('#life-train').click()
    frozen=trace()
    assert frozen['f']==initial['f']
    assert frozen['pred']!=initial['pred']
    assert frozen['deltaEncoder']==0
    page.locator('#life-freeze').uncheck()
    page.locator('#life-reset').click()
    # Same trainable network in both page locations.
    page.locator('#train-one').click()
    assert trace()['iteration']==1
    assert '1 updates' in page.locator('#life-status').inner_text()
    for width in (390,768,1440):
        page.set_viewport_size({'width':width,'height':1100})
        for step in range(7):
            page.locator(f'[data-life-step="{step}"]').click()
            assert page.evaluate('D4RTLifetime.snapshot().step')==step
            assert page.locator('#life-title').inner_text()
            assert 'NaN' not in page.locator('#lifetime').inner_text()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),(width,step)
        page.locator('[data-life-step="2"]').click()
        page.locator('#lifetime').scroll_into_view_if_needed()
        page.screenshot(path=str(screens/f'local-{width}.png'))
        if width==1440:
            page.locator('#lifetime').screenshot(path=str(screens/'full-lab.png'))
        page.locator('[data-life-step="4"]').click()
        page.locator('#life-query').select_option('1')
        assert 'QUERY 2 READS F' in page.locator('#life-detail').inner_text()
    page.locator('#life-frame-2').focus()
    page.keyboard.press('ArrowRight')
    assert 'frame 2' in page.locator('#life-pixel').inner_text()
    assert not errors,errors
    print(json.dumps({'result':'passed','input_gradients':gradients,'loss_before':initial['loss'],
        'loss_after_step':trained['loss'],'checks':['pixel to tubelet mapping','linear projection',
        'masked and global attention math','1→4→16 perturbation propagation',
        'query independence','source RGB bypass','finite difference input gradients',
        'frozen encoder','fixed pixels','shared training controls','7 stages, 3 viewports'],
        'screenshots':str(screens),'browser_errors':errors},indent=2))
    browser.close()
