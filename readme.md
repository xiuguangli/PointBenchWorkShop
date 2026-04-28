# PointBench

PointBench is a standalone point prediction evaluation project. The default
entrypoint is `run.sh`, which calls `model_evaluator.py` and evaluates
`point_agent`.

## Usage

### 1. Env setup

Create `.env` from `.env.example` and fill in the Gemini API settings:

```bash
cp .env.example .env
```

There are two supported Gemini configurations:

1. Third-party Gemini-compatible API

```bash
API_BASE_URL=https://api.uniapi.io/gemini
API_KEY=your_third_party_api_key
SA2VA_PLANNER_MODEL=gemini-3.1-pro-preview
```

Use this when the provider gives a Gemini-compatible `base_url`. The code will
create the client with `types.HttpOptions(base_url=API_BASE_URL)`.

2. Official Google Gemini API

```bash
# leave this empty or remove it
API_BASE_URL=
API_KEY=your_official_gemini_api_key
SA2VA_PLANNER_MODEL=gemini-3.1-pro-preview
```

Use this when calling Gemini through the official Google API. When
`API_BASE_URL` is empty, the code creates the client with only `API_KEY`.

`SA2VA_PLANNER_MODEL` is optional and defaults to `gemini-3.1-pro-preview`. It is
used by `point_agent` for query rewriting, judge, and fallback grounding.

### 2. Run evaluation

```bash
# only one gpu
bash run.sh --gpu 0

# multigpu
bash run.sh --gpu 0,1,2
```

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
