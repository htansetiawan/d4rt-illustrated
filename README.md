# D4RT, illustrated

A buildless interactive companion to **Efficiently Reconstructing Dynamic Scenes One D4RT at a Time**.

Live: https://htansetiawan.github.io/d4rt-illustrated/

## Explore

- A synthetic geometry lab separates source time, target time and camera coordinates.
- Clickable encoder–representation–decoder walkthrough.
- Query recipes for tracking, point clouds, depth and camera pose.
- An occupancy-grid simulation explains repeated-work avoidance.
- Keyboard controls, mobile layout, reduced-motion support and print styles.

All simulations are original teaching illustrations, **not D4RT inference or benchmark measurements**. The floor, sphere and cube positions are explicit functions. World-view orbit is separate from the camera reference in the query. Coordinate units are arbitrary. No neural weights, external assets, analytics or dependencies are loaded.

## Run and publish

Open `index.html` directly, or run `python3 -m http.server 8000` here. GitHub Pages serves the repository root on `main`; `.nojekyll` disables Jekyll processing. There is no build step.

## Research sources

- [Conference publication](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html)
- [Authors’ project](https://d4rt-paper.github.io/)
- [Technical preprint, method §2, Algorithm 1](https://arxiv.org/html/2512.08924v1)
- [DeepMind introduction](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/)

Checked 2026-10-10. Independent educational companion; no author affiliation is implied. Research credit remains with the listed authors. Diagrams are generated locally from original code rather than copied from the publication.
