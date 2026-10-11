from pathlib import Path

root = Path(__file__).resolve().parents[1]
html = (root / 'index.html').read_text()
js = (root / 'genie.js').read_text()
checks = {
    'Genie ST-ViViT / D4RT distinction': 'ST-ViViT' in html and 'continuous feature' in html,
    'ST-transformer name and factorization': 'causal temporal attention' in html and 'within one time' in html,
    'STTN is explicitly disambiguated': 'STTN video-restoration model' in html,
    'STTN citation present': 'spatio-temporal-transformer-network-for-video-restoration' in html,
    'Tokenizer objective and scale': '300k AdamW steps' in js and '1,024 video codes' in js,
    'Dynamics objective and scale': 'cross-entropy' in js and '125k steps' in js and '256 TPUv5' in js,
    'Genie total count is precise': '10.7B parameters' in html,
    'All three lifecycle phases remain interactive': all(f'data-ga="{k}"' in html for k in ('tokenizer', 'lam', 'world')),
    'Cache versions updated': 'genie.js?v=2' in html and 'pipeline.js?v=16' in html,
}
for label, ok in checks.items():
    print(('PASS' if ok else 'FAIL') + ': ' + label)
assert all(checks.values())
