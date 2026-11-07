# SeaSift

Sift hours of murky underwater survey video down to minutes of review. Built for Coastal Action at RBC HubHacks.

SeaSift finds the moments worth looking at, enhances murky footage, helps people identify species, and supports Coastal Action's frame-by-frame counting protocol (MeanCount). People make every identification and count.

It runs on your own laptop. The main version uses classical computer vision, with no trained model and no cloud AI. Local AI is an optional add-on.

## Requirements

- Python 3.12 (the version we tested with)
- About 600 MB of disk space for the install, plus room for your videos

## Install

From the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Try it with the test clip

```bash
python tools/make_test_video.py
streamlit run app.py
```

The app opens at http://localhost:8501. Choose `test_murky.mp4` on **1 · Process a video**, click **Process video**, then go to **2 · Review moments**.

The test clip is synthetic. It shows that the app works, not how well it performs on real footage.

## Use your own video

Copy the video into `data/videos/`, then pick it on the **Process a video** page. For big files, copying is better than using the upload box.

Long 4K videos can take a while to process, and clicking elsewhere in the app stops the run. To process from the command line instead:

```bash
python -m seasift.pipeline data/videos/YOUR_VIDEO.mp4 --site coastal_eelgrass_example --date YYYY-MM-DD
```

Results are saved in `data/`, so you only need to process each video once.

## Pages

| Page | What it does |
|---|---|
| 1 · Process a video | Finds moments with movement and scores footage quality |
| 2 · Review moments | Go through each moment, identify the species and save a count |
| 3 · Report | Species, counts and proof pictures, as a one-file HTML report or CSVs |
| 4 · Sightings | New arrivals and watch-list species at each site |
| 5 · Count frames (MeanCount) | Click each animal on sampled frames; SeaSift adds up the totals |
| Species lists | Edit each site's species list |

The species lists and the descriptions on the species cards are sample data. They haven't been checked by a biologist.

## Run the tests

```bash
python tools/smoke_test.py       # should end with "79 of 79 checks passed"
```

Run this with only the test clip in `data/videos/`. If a large video sorts ahead of it, the test tries to process that video and times out.

## Optional: local AI

This adds a fish detector and species suggestions, and a person still confirms every result. The real models haven't been tested in SeaSift yet, so treat this as experimental.

```bash
pip install -r requirements-ai.txt
curl -L -o models/cfd-yolov12x-1.00.pt https://github.com/filippovarini/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt
YOLO_OFFLINE=true python tools/check_ai.py
```

`check_ai.py` downloads BioCLIP the first time it runs, so you need internet for that step. `YOLO_OFFLINE=true` turns off Ultralytics' usage analytics; keep it set whenever you run the app with AI as well.

Restart the app afterwards, then turn on **Use local AI** on the **Process a video** page.

The detector and `ultralytics` are licensed under AGPL-3.0.

## Project layout

| Path | What's there |
|---|---|
| `app.py` | The Streamlit app |
| `seasift/` | Video processing, species lists, counting and reports |
| `seasift/config.py` | All the settings you can tune |
| `tools/` | Test clip, smoke test and evaluation scripts |
| `data/species_lists/` | One species list per site |
| `docs/AI_GUIDELINES_DRAFT.md` | A draft of AI guidelines for Coastal Action |

Videos, model weights and the results database stay on your computer and are ignored by git.
