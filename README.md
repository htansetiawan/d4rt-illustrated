# D4RT, illustrated

A buildless interactive companion to **Efficiently Reconstructing Dynamic Scenes One D4RT at a Time**.

Live: https://htansetiawan.github.io/d4rt-illustrated/

## Explore

- An eight-stage video lifetime walkthrough, with training/inference modes and SVG export.
- A tubelet picker over two actual synthetic RGB frames: 2 × 16 × 16 pixels → one continuous token.
- Exact training-shape arithmetic: 48 × 256 × 256 → 6,144 video tokens, plus aspect-ratio metadata.
- Selectable encoder attention masks: local, global, and a separately labeled Genie causal-temporal comparison.
- Query construction, decoder Q/K/V routing, attention cost, and all 13 output dimensions.
- A five-stage end-to-end training walkthrough, loss families, query sampling, and an interactive confidence objective.
- A comparison with the original Genie paper, separating continuous scene features from a VQ video tokenizer.
- A synthetic geometry lab separates source time, target time and camera coordinates.
- Query recipes for tracking, point clouds, depth and camera pose.
- An occupancy-grid simulation explains repeated-work avoidance.
- Keyboard controls, mobile layout, reduced-motion support and print styles.

All simulations are original teaching illustrations, **not D4RT inference or benchmark measurements**. The floor, sphere and cube positions are explicit functions. World-view orbit is separate from the camera reference in the query. Coordinate units are arbitrary. The tubelet lab crops real pixels from a procedurally drawn clip, but does not apply learned patch-projection weights. Attention masks show permitted connections, not learned weights. No neural weights, external assets, analytics or dependencies are loaded.

## Evidence boundaries

The reported model uses a 40-layer ViT-g encoder (1 B parameters), 2 × 16 × 16 patches, and an 8-layer cross-attention decoder (144 M parameters). Its training configuration uses 48 frames at 256 × 256, 2,048 sampled queries, and 500k AdamW steps. These are sourced from §4; the complete diagram is in Appendix A, Figure 7. Feature widths and head counts are left symbolic rather than inferred from a similarly named backbone.

Token counts, attention-score counts and the confidence minimum are derived arithmetic. Metadata tokens are excluded from score estimates, which are not memory or timing benchmarks. The confidence demo isolates `c * error - 0.2 * log(c)` and explores `(0, 1]` without asserting the head activation. The paper's appendix repeats a confidence coefficient/term; the website does not invent a corrected full loss.

Genie refers to **Genie: Generative Interactive Environments (2024)**, not later Genie versions. Its factorized causal temporal attention is presented separately from D4RT's full-clip global attention.

## Files

- `index.html`: all content, research links and accessible control markup.
- `anatomy.js` / `anatomy.css`: model, tokenization, attention and training interactives.
- `scene.js` / `app.js`: synthetic geometry, task recipes and occupancy demo.
- `style.css`: shared gray-on-white design and responsive layout.

## Run and publish

Open `index.html` directly, or run `python3 -m http.server 8000` here. GitHub Pages serves the repository root on `main`; `.nojekyll` disables Jekyll processing. There is no build step.

## Research sources

- [Conference publication](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html)
- [Authors’ project](https://d4rt-paper.github.io/)
- [Technical preprint, method §2, Algorithm 1](https://arxiv.org/html/2512.08924v1)
- [Full model and training recipe, Appendix A](https://arxiv.org/html/2512.08924v1#A1)
- [Genie methodology and ST-transformer](https://arxiv.org/html/2402.15391v1#S2)
- [DeepMind introduction](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/)

Checked 2026-10-10. Independent educational companion; no author affiliation is implied. Research credit remains with the listed authors. Diagrams are generated locally from original code rather than copied from the publication.
