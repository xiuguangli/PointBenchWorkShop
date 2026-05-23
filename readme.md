# PointBench

PointBench is a standalone point prediction evaluation project. The default
entrypoint is `run.sh`, which calls `model_evaluator.py` and evaluates
`molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge`.

## Usage

### 1. Env setup

Create `.env` from `.env.example` and fill in the Gemini API settings:

```bash
cp .env.example .env
```

Note: Before running any `bash` startup scripts, ensure the `uv` CLI tool is installed and available. The project's startup scripts use `uv run ...` to run the evaluation. If `uv` is not installed, install it with:

```bash
pip install uv
```

After installing `uv`, you can run the commands below.

There are two supported Gemini configurations:

1. Third-party Gemini-compatible API

```bash
API_BASE_URL=https://api.uniapi.io/gemini
API_KEY=your_third_party_api_key
```

Use this when the provider gives a Gemini-compatible `base_url`. The code will
create the client with `types.HttpOptions(base_url=API_BASE_URL)`.

2. Official Google Gemini API

```bash
# leave this empty or remove it
API_BASE_URL=
API_KEY=your_official_gemini_api_key
```

Use this when calling Gemini through the official Google API. When
`API_BASE_URL` is empty, the code creates the client with only `API_KEY`.

### 2. Run evaluation

```bash
# only one gpu
bash run.sh --gpu 0

# multigpu
bash run.sh --gpu 0,1,2
```

When multiple GPUs are visible, the current Molmo2 fused pipeline will start at
most one worker process per visible GPU, and the main process will aggregate the
results with a single overall progress bar. You do not need to set a separate
`--workers` value for this pipeline.

The fused pipeline now runs stage-wise across the dataset:
batch rewrite -> batch Gemini box helper -> batch local Molmo2 -> batch judge/fallback.

Each stage prints its own named progress bar, so you can see whether the run is
currently in rewrite, Gemini helper, local Molmo2, judge, or fallback.

The fused pipeline also keeps a per-sample stage cache in the corresponding
`visualizations/.../*_justify_meta.json` file. If the same command is rerun, the
pipeline will resume from the last finished stage for each unfinished sample
instead of repeating completed Gemini API stages. Use `--no-resume` if you want
to ignore the cache and rerun from scratch.

The current project-specific fused pipeline is:

`raw user_input -> transform_gemini_twolines-style Gemini rewrite -> Gemini box/center helper -> refpoint-hybrid Molmo2 -> Gemini judge/fallback`

For Molmo2 weights:

- If you leave `--model_root` empty, the code will load the HuggingFace repo named by `--model` and let `transformers` download/cache it automatically.
- If you provide `--model_root`, the code expects the local weights at `<model_root>/<huggingface_repo_id>`, for example `<model_root>/allenai/Molmo2-4B`.
- You can also pass the concrete model directory itself, for example `/path/to/models/allenai/Molmo2-4B`.

The original base model branches such as `--type gemini` in `model_evaluator.py`
are still kept. The removed experimental pipeline wrappers are no longer exposed.

## Data

`data/` contains the evaluation metadata, images, masks

```text
data/
├── data.json
├── pixmo_metadata.csv
├── images/
│   ├── affordable/
│   ├── counting/
│   ├── reasoning/
│   ├── spatial/
│   ├── steerable/
├── masks/
    ├── affordable/
    ├── counting/
    ├── reasoning/
    ├── spatial/
    └── steerable/
```
