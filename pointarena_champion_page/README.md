# Point-Agent PointArena Champion Page

Static webpage draft for the ERA @ CVPR 2026 PointArena Challenge follow-up.

## Preview

Open `index.html` directly in a browser, or run a tiny local server:

```bash
cd pointarena_champion_page
python -m http.server 8080
```

Then visit `http://localhost:8080`.

## GitHub Pages

This directory is already wired for GitHub Pages in the PointBench repository.

Deployment is handled by [PointBench/.github/workflows/pages.yml](../.github/workflows/pages.yml), which publishes this directory on pushes to `main`.

Published site URL:

`https://xiuguangli.github.io/PointBenchWorkShop/`

If you update the page, push the changes to `main` and GitHub Actions will redeploy the site automatically.

## Result Snapshot

The page reports PointAgent (ours) together with cited baselines:

```text
Average: 86.05%
Affordance: 96.97%
Spatial: 88.21%
Reasoning: 88.60%
Steerability: 81.50%
Counting: 75.00%
```
