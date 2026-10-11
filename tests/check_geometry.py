"""Verify geometric labels and analytic derivatives independently in a browser."""
import json
import math
import os
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.environ.get('D4RT_GEOMETRY_URL', (root / 'index.html').as_uri())
screens = Path(tempfile.mkdtemp(prefix='d4rt-geometry-'))

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8), (a, b)

def slide(page, name, value):
    page.locator('#' + name).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"));}', str(value))

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get('CHROME_PATH', '/usr/bin/google-chrome'),
                               headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1100}, reduced_motion='reduce')
    errors=[]
    page.on('pageerror', lambda e:errors.append(str(e)))
    page.goto(url, wait_until='load')
    page.wait_for_function('window.D4RTGeometry !== undefined')
    snapshot=lambda:page.evaluate('D4RTGeometry.snapshot()')
    initial=snapshot()
    # Independently reconstruct the target-time trajectory and camera transform.
    t,c=30,18
    world=[[-.9+.039*t, .24*math.sin(t/12), 4.8],
           [1.05-.007*t,-.45,5.5+.005*t],[-.6,.6-.006*t,6.3]]
    center=[.48*math.sin(c/24),.08*math.sin(c/18),0]
    yaw=.12*math.sin(c/30)
    for actual,w in zip(initial['target'], world):
        x,y,z=w[0]-center[0],w[1]-center[1],w[2]
        target=[math.cos(yaw)*x-math.sin(yaw)*z,y,math.sin(yaw)*x+math.cos(yaw)*z]
        for a,b in zip(actual,target):close(a,b)
    # Source time changes identity coordinates, not the queried physical state.
    slide(page,'geo-src',12)
    source=snapshot()
    assert source['target']==initial['target']
    assert source['source']!=initial['source']
    assert source['pred']==initial['pred']
    slide(page,'geo-tgt',42)
    target=snapshot()
    assert target['world']!=source['world'] and target['source']==source['source']
    assert target['pred']==source['pred']
    slide(page,'geo-cam',2)
    cam=snapshot()
    assert cam['world']==target['world'] and cam['target']!=target['target']
    assert cam['source']==target['source'] and cam['pred']==target['pred']
    # Mean-depth normalization and signed log, rebuilt without site functions.
    valid=[i for i,v in enumerate(cam['valid']) if v]
    zp=sum(cam['pred'][i][2] for i in valid)/len(valid)
    zt=sum(cam['target'][i][2] for i in valid)/len(valid)
    sl=lambda x:math.copysign(math.log1p(abs(x)),x)
    l1=sum(sum(abs(sl(cam['pred'][i][k]/zp)-sl(cam['target'][i][k]/zt))
               for k in range(3)) for i in valid)/len(valid)
    close(cam['loss'],.8*l1-.2*math.log(.8))
    slide(page,'geo-scale',2)
    close(snapshot()['loss'],cam['loss'])
    slide(page,'geo-scale',1)
    # Independently finite-difference every XYZ output, including the mean-Z coupling.
    def loss(pred, target, confidence=.8):
        zp=sum(p[2] for p in pred)/len(pred)
        zt=sum(p[2] for p in target)/len(target)
        return confidence*sum(sum(abs(sl(p[k]/zp)-sl(t[k]/zt)) for k in range(3))
                              for p,t in zip(pred,target))/len(pred)-.2*math.log(confidence)
    gradients=[]
    for i in range(3):
        for k in range(3):
            plus=[p[:] for p in cam['pred']];minus=[p[:] for p in cam['pred']]
            plus[i][k]+=1e-5;minus[i][k]-=1e-5
            numerical=(loss(plus,cam['target'])-loss(minus,cam['target']))/2e-5
            close(cam['gradient'][i][k],numerical)
            gradients.append({'point':i,'axis':k,'analytic':cam['gradient'][i][k],'numeric':numerical})
    page.locator('#geo-missing').check()
    masked=snapshot()
    assert masked['valid']==[False,True,True]
    assert masked['gradient'][0]==[0,0,0]
    assert masked['pred']==cam['pred'] and masked['query']==cam['query']
    close(masked['zp'],sum(p[2] for p in cam['pred'][1:])/2)
    page.locator('#geo-missing').uncheck()
    page.locator('#geo-depth-task').click()
    depth=snapshot()
    assert depth['state']['src']==depth['state']['tgt']==depth['state']['cam']
    for v,pred in zip(depth['source'],depth['target']):
        close(v[0],.5+.9*pred[0]/pred[2]);close(v[1],.5+.9*pred[1]/pred[2])
    page.locator('#geo-track-task').click()
    assert snapshot()['state']['cam']==0
    for point in range(3):
        page.locator('#geo-point').select_option(str(point))
        for n in ('src','tgt','cam'):
            for t in (0,47):
                slide(page,'geo-'+n,t)
                assert all(math.isfinite(v) for row in snapshot()['target'] for v in row)
    # New volume follows the existing pixel picker, including final tubelet.
    slide(page,'dd-time',23);slide(page,'dd-col',15);slide(page,'dd-row',15)
    assert 'frames 47–48' in page.locator('#geo-volume-address').inner_text()
    assert 'column 15, row 15' in page.locator('#geo-volume-address').inner_text()
    missing=page.locator('a[href^="#"]').evaluate_all('(es)=>es.map(e=>e.hash).filter(h=>h.length>1&&!document.getElementById(h.slice(1)))')
    assert not missing,missing
    ids=page.locator('[id]').evaluate_all('(es)=>es.map(e=>e.id)')
    assert len(ids)==len(set(ids))
    for width in (390,768,1440):
        page.set_viewport_size({'width':width,'height':1100})
        for part in ('model-atlas','tokenizer','decoder','geometry-loss','genie'):
            page.locator('#'+part).scroll_into_view_if_needed()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,part)
            assert 'NaN' not in page.locator('#'+part).inner_text()
            page.screenshot(path=str(screens/f'{part}-{width}.png'))
        page.locator('#decoder').screenshot(path=str(screens/f'decoder-full-{width}.png'))
    assert not errors,errors
    print(json.dumps({'result':'passed','checks':['geometric target construction','three independent times',
          'source reprojection','no labels in prediction inputs','independent loss math',
          'scale invariance','nine finite-difference gradients','missing labels',
          'tubelet control linkage','internal links','three responsive viewports'],
          'gradients':gradients,'screenshots':str(screens)},indent=2))
    browser.close()
