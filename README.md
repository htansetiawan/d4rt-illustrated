# D4RT, illustrated

An independent, interactive anatomy of **Efficiently Reconstructing Dynamic Scenes One D4RT at a Time**.

**Live:** https://htansetiawan.github.io/d4rt-illustrated/

## Start with the anatomy — edition 09

The homepage now follows a video's lifetime directly. The older, larger workbench remains at [reference.html](https://htansetiawan.github.io/d4rt-illustrated/reference.html); previously published workbench fragment links are forwarded there when needed.

- **Clickable full architecture:** inspect both RGB input paths, frame-local/global encoder operations, the shared feature bank, independent decoder queries and all 13 output channels. Switch training/inference to distinguish learned parameters, activations, cached features and gradient flow.
- **Attention cost calculator:** change clip length and query count to see local, global and decoder score counts; compare Genie-style factorized routing on the same explicitly hypothetical token lattice.
- **Connected overview:** video → tubelets → ViT encoder → shared scene features → independent point queries → output and training losses. Six stages include diagrams, tensor shapes, operations and sources.
- **Pixel-level tokenization:** inspect two actual synthetic RGB crops, all 1,536 scalar addresses, and a computed continuous projection. The final tubelet is token 6,143. Inspector annotations do not enter the pixel data.
- **Inside the encoder:** a 16-token, width-four computed transformer block exposes normalization, Q/K/V, scores, masks, softmax, value mixing, output projection, residual connections and an MLP. Switch between local/global attention and the separately labeled Genie temporal comparison. Each row is a numerical distribution, not a decorative heatmap.
- **How it trains:** distinguish VideoMAE initialization, joint geometry training and fixed-weight inference. Follow geometric target construction, label masks, loss families, query sampling and the optimizer recipe.
- **Train a miniature:** a separate 189-parameter attention network learns two coordinate targets from one synthetic clip using browser-native reverse-mode autodiff. Freeze its encoder to see which weights stop changing.
- **Genie comparison:** continuous scene features versus discrete video codes, global attention versus factorized spatial/causal temporal attention, and geometry querying versus action-conditioned generation.

All illustrations are original teaching tools. They do **not** execute D4RT weights or report D4RT performance. The trainable miniature fits two queries from one clip; its loss decrease is evidence of gradient flow, not generalization. Both teaching networks omit or simplify parts of the full model, as stated beside each experiment.

## Scientific boundaries

Reported architecture: 40-layer ViT-g encoder, 8-layer cross-attention decoder, 2 × 16 × 16 tubelets. Training clips are 48 × 256 × 256; their 6,144 video-token count excludes aspect-ratio metadata. Position encoding, exact feature widths, head counts and the encoder's local/global layer schedule are not invented here.

Training losses use available labels. Occlusion does not necessarily make a synthetic 3D label unavailable. The paper separately normalizes prediction and target sets by their mean depths before signed-log 3D error. Appendix A repeats a confidence term; the companion explains the clear components without silently correcting the full objective.

Genie means **Genie: Generative Interactive Environments (2024)**. Its ST-transformer is not assumed to use D4RT's two-frame tubelet size.

## Files

- `atlas.js`, `atlas.css`: connected architecture inspector and attention shape calculator.
- `index.html`, `deep-dive.js`, `deep-dive.css`: the anatomy-first homepage, raw-pixel inspector and computed encoder block.
- `tour.js`, `tour.css`: six-stage walkthrough and differentiable miniature, also used by the earlier `anatomy-tour.html` permalink.
- `reference.html`: advanced reference experiments and original geometry lab.
- `workbench.*`, `cutaway.*`, `execution.*`, `anatomy.*`, `microscope.*`, `scene.js`, `app.js`, `style.css`: reference-workbench modules.
- `tests/check_atlas.py`: every architecture block, training/inference transitions, independent attention-count checks and responsive layouts.
- `tests/check_deep_dive.py`: pixel addresses/projection, masks, causality, normalization, inspector invariance, keyboard/blueprint navigation, old deep links and responsive layouts.
- `tests/check_tour.py`: independent finite-difference gradient checks, frozen-encoder invariance, loss updates and tour controls.
- `tests/check_workbench.py`: retained reference experiments.

## Run

Buildless HTML, CSS and JavaScript. Open `index.html`, or run `python3 -m http.server 8000`. GitHub Pages serves the repository root on `main`; `.nojekyll` is retained. No model downloads or external fonts.

Browser tests require Python Playwright and Chrome. Set `CHROME_PATH` if needed. `D4RT_DEEP_URL`, `D4RT_TOUR_URL` and `D4RT_CHECK_URL` override their respective target URLs. Example:

```sh
python3 tests/check_deep_dive.py
D4RT_TOUR_URL=file:///absolute/path/index.html python3 tests/check_tour.py
python3 tests/check_workbench.py
```

Tests cover 390, 768 and 1440 pixel widths. Screenshots are written to temporary directories. The teaching network's analytical gradients are checked against finite differences in encoder, query, decoder and output parameters.

## Primary sources

- [Conference paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html)
- [Authors' project](https://d4rt-paper.github.io/)
- [D4RT method and architecture](https://arxiv.org/html/2512.08924v2#S2)
- [D4RT model and training appendix](https://arxiv.org/html/2512.08924v2#A1)
- [VideoMAE initialization ablation](https://arxiv.org/html/2512.08924v2#A4)
- [Genie ST-transformer and tokenizer](https://arxiv.org/html/2402.15391v1#S2)

Research checked October 11, 2026. Credit remains with the paper's authors; no author affiliation is implied.
