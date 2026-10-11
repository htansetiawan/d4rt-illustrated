# D4RT, illustrated

An independent, interactive anatomy of **Efficiently Reconstructing Dynamic Scenes One D4RT at a Time**.

**Live:** https://htansetiawan.github.io/d4rt-illustrated/

## Trace the tensors — edition 15

The nine-stage homepage now carries an attached tensor notebook: shapes, selected tubelet addresses, softmax axes, and per-head attention score counts. Change the decoder query count (1, 2,048, or 65,536) to see which dimensions grow while the encoded clip stays the same. Training retains the paper's 2,048-query setting and explains how shared features and mean-depth normalization connect query losses.

Start directly at [tubelets](https://htansetiawan.github.io/d4rt-illustrated/#walk-tubelet), [encoder self-attention](https://htansetiawan.github.io/d4rt-illustrated/#walk-local), [cross-attention](https://htansetiawan.github.io/d4rt-illustrated/#walk-decoder), or [training](https://htansetiawan.github.io/d4rt-illustrated/#walk-train). Every operation now has a shareable link.

The shape notebook uses arithmetic, not model inference. C, D, head counts and detailed metadata routing remain unspecified; attention counts exclude metadata and do not imply that a full score matrix is allocated. `tensor-trace.js` and `tests/check_tensor_trace.py` implement and verify this addition.

## Open the whole model — edition 14

The homepage now starts with a connected, inspectable model diagram. Both RGB branches remain visible: tubelets → video ViT → scene memory, and source patch/coordinates/times → query → cross-attention. Select any of seven components for its tensor shapes, computation and numerical lab. A gradient toggle reverses the flow and explains which parameters learn.

The weight-lifetime strip distinguishes VideoMAE V2 pretraining, end-to-end D4RT geometry training, and fixed-weight inference. The 40-layer encoder is explicitly not presented as 40 local/global pairs. Unspecified residual/normalization wiring remains unspecified.

New files: `model-map.js`, `model-map.css`, `tests/check_model_map.py`. The nine-stage video walkthrough and prior computation labs remain accessible below.

## Start with the video lifetime — edition 13

The homepage now follows **one persistent video region through nine large, sequential diagrams**. Its full 6,144-address lattice makes frame-local, global and Genie-style causal connectivity inspectable at the reported video-token count. Scrub or play the synthetic clip, trace both patches into a tubelet, then follow the continuous projection, encoder, query branch, decoder and supervised update.

Use `index.html` for this guided entry point; `anatomy.html` retains every edition-12 computation lab. Old published anatomy fragment links redirect to their original experiments. No D4RT weights execute in either page.

New files: `pipeline.js`, `pipeline.css`, `tests/check_pipeline.py`. The new test independently checks token addresses, pixel crops, causal and composed masks, inference/training navigation, legacy links, SVG bounds and responsive layouts.

## Detailed anatomy — edition 12

Edition 12 adds nine visual cutaways directly inside the main architecture inspector. Start at RGB video, scrub a tubelet address, select readable encoder tokens, inspect the two decoder streams, and follow supervision back to learned weights. A reduced 12-token lattice compares D4RT local/global masks with Genie’s causal temporal sublayer; masks show permitted connections, not trained attention weights.

The homepage leads with the reported architecture, then follows tubelets → encoder → decoder → geometric supervision. Small executable networks are placed afterward as optional computation labs. The older, larger workbench remains at [reference.html](https://htansetiawan.github.io/d4rt-illustrated/reference.html); previously published workbench fragment links are forwarded there when needed.

- **Connected geometry and training:** source, target and camera controls compute point trajectories and world-to-camera targets. An inline decoder schematic shows both input streams, query embeddings, cross-attention and the 13 output channels. The loss workshop independently normalizes prediction/label depths and exposes signed-log L1, confidence, missing labels, scale invariance and analytic XYZ derivatives. Predictions are deliberately separate fixed teaching values, not D4RT outputs.
- **Exploded tubelets:** two separated frame planes and their common patch address follow the original RGB inspector. Pixel grouping is distinguished from temporal attention.
- **One clip, one computed lifetime:** edit an RGB pixel and trace changes through tubelet projection, local attention, global attention, an independent query decoder and the loss. The displayed values come from one executable network. Change a query without changing the scene bank; freeze the encoder and inspect its zero parameter update.
- **Clickable full architecture:** inspect both RGB input paths, frame-local/global encoder operations, the shared feature bank, independent decoder queries and all 13 output channels. Switch training/inference to distinguish learned parameters, activations, cached features and gradient flow.
- **Attention cost calculator:** change clip length and query count to see local, global and decoder score counts; compare Genie-style factorized routing on the same explicitly hypothetical token lattice.
- **Connected overview:** video → tubelets → ViT encoder → shared scene features → independent point queries → output and training losses. Six stages include diagrams, tensor shapes, operations and sources.
- **Pixel-level tokenization:** inspect two actual synthetic RGB crops, all 1,536 scalar addresses, and a computed continuous projection. The final tubelet is token 6,143. Inspector annotations do not enter the pixel data.
- **Inside the encoder:** a 16-token, width-four computed transformer block exposes normalization, Q/K/V, scores, masks, softmax, value mixing, output projection, residual connections and an MLP. Switch between local/global attention and the separately labeled Genie temporal comparison. Each row is a numerical distribution, not a decorative heatmap.
- **How it trains:** distinguish VideoMAE initialization, joint geometry training and fixed-weight inference. Follow geometric target construction, label masks, loss families, query sampling and the optimizer recipe.
- **Train a miniature:** the same 249-parameter attention network learns two coordinate targets from one synthetic clip using browser-native reverse-mode autodiff. Freeze its encoder to see which weights stop changing.
- **Genie comparison:** continuous scene features versus discrete video codes, global attention versus factorized spatial/causal temporal attention, and geometry querying versus action-conditioned generation.

All illustrations are original teaching tools. They do **not** execute D4RT weights or report D4RT performance. The trainable miniature fits two queries from one clip; its loss decrease is evidence of gradient flow, not generalization. Both teaching networks omit or simplify parts of the full model, as stated beside each experiment.

## Scientific boundaries

Reported architecture: 40-layer ViT-g encoder, 8-layer cross-attention decoder, 2 × 16 × 16 tubelets. Training clips are 48 × 256 × 256; their 6,144 video-token count excludes aspect-ratio metadata. Position encoding, exact feature widths, head counts and the encoder's local/global layer schedule are not invented here.

Training losses use available labels. Occlusion does not necessarily make a synthetic 3D label unavailable. The paper separately normalizes prediction and target sets by their mean depths before signed-log 3D error. Appendix A repeats a confidence term; the companion explains the clear components without silently correcting the full objective.

Genie means **Genie: Generative Interactive Environments (2024)**. Its ST-transformer is not assumed to use D4RT's two-frame tubelet size.

The geometry workshop uses arbitrary units and zero-based times. Its isolated objective averages coordinate-summed L1 over labeled points with a shared confidence, and excludes unlabeled points from its depth means. These explicit teaching conventions do not claim to recover unspecified paper reduction/masking details. The output gradient includes dependence on the shared predicted mean depth, verified for all nine coordinates by independent finite differences.

## Files

- `atlas-cutaways.js`, `atlas-cutaways.css`: nine visual operations, guided stage navigation, tubelet address sliders and encoder connectivity comparison.
- `tests/check_cutaways.py`: independently enumerated local/global/causal masks, final tubelet index, training/inference navigation and responsive SVG layout.

- `geometry-lab.js`, `geometry-lab.css`: connected physical-query and supervision workshops, decoder anatomy, tubelet-volume diagram.
- `tests/check_geometry.py`: independent geometry, normalization and derivative checks; source/target/camera invariants; no label inputs to predictions; scale invariance; masking; responsive layouts.
- `lifetime.js`, `lifetime.css`: connected video-to-loss inspector, using the same autodiff model as the training miniature in `tour.js`.
- `atlas.js`, `atlas.css`: connected architecture inspector and attention shape calculator.
- `anatomy.html`, `deep-dive.js`, `deep-dive.css`: the anatomy-first homepage, raw-pixel inspector and computed encoder block.
- `tour.js`, `tour.css`: six-stage walkthrough and differentiable miniature, also used by the earlier `anatomy-tour.html` permalink.
- `reference.html`: advanced reference experiments and original geometry lab.
- `workbench.*`, `cutaway.*`, `execution.*`, `anatomy.*`, `microscope.*`, `scene.js`, `app.js`, `style.css`: reference-workbench modules.
- `tests/check_lifetime.py`: independent tensor/projection/attention arithmetic, 1→4→16 pixel-perturbation propagation, source-pixel bypass, query independence, input finite differences and shared training controls.
- `tests/check_atlas.py`: every architecture block, training/inference transitions, independent attention-count checks and responsive layouts.
- `tests/check_deep_dive.py`: pixel addresses/projection, masks, causality, normalization, inspector invariance, keyboard/blueprint navigation, old deep links and responsive layouts.
- `tests/check_tour.py`: independent finite-difference gradient checks, frozen-encoder invariance, loss updates and tour controls.
- `tests/check_workbench.py`: retained reference experiments.

## Run

Buildless HTML, CSS and JavaScript. Open `index.html`, or run `python3 -m http.server 8000`. GitHub Pages serves the repository root on `main`; `.nojekyll` is retained. No model downloads or external fonts.

Browser tests require Python Playwright and Chrome. Set `CHROME_PATH` if needed. `D4RT_DEEP_URL`, `D4RT_TOUR_URL` and `D4RT_CHECK_URL` override their respective target URLs. Example:

```sh
python3 tests/check_geometry.py
python3 tests/check_deep_dive.py
D4RT_TOUR_URL=file:///absolute/path/anatomy.html python3 tests/check_tour.py
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

Edition 11 verification: geometry, architecture, deep-dive and lifetime suites pass at 390, 768 and 1440 pixels; new desktop/mobile diagrams visually reviewed. No browser exceptions or page-level horizontal overflow.

Edition 12 verification: cutaway, architecture, deep-dive and geometric-training suites pass at 390, 768 and 1440 pixels. Checks independently enumerate all 48 selected-token/mask combinations, verify the last tubelet index (6,143), preserve training/inference controls, and check SVG label bounds and selected-token contrast. Desktop/mobile diagrams visually reviewed; no browser exceptions or page-level horizontal overflow.

Edition 13 verification: the guided-page suite passes 73 independently enumerated attention masks, final tubelet pixel/address checks, query controls, training/inference switching, animation stop, legacy deep-link forwarding and SVG bounds at 390/768/1440 pixels. The retained cutaway suite also passes. New tubelet, encoder and training diagrams were visually reviewed; no browser exceptions or page-level horizontal overflow. Wide diagrams scroll inside their panels on narrow screens.

Edition 14 verification: seven anatomy components, forward/backward controls, keyboard activation, local lab anchors and layouts pass at 390/768/1440 pixels. The existing pipeline suite also passes all 73 attention-mask checks and its pixel, query and navigation contracts. Desktop/mobile anatomy screenshots reviewed; no browser exceptions or page-level horizontal overflow.

Edition 15 verification: tensor-trace tests pass all nine direct links, three query counts, final tubelet addressing, fixed training-query count and same-fragment navigation at 390/768/1440 pixels. The retained pipeline suite passes 73 connectivity masks and all pixel/navigation checks. No browser exceptions. Desktop and mobile tensor views were visually reviewed.
