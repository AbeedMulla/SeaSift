# Model weights go here (AI tier only)

- `cfd-yolov12x-1.00.pt`: the Community Fish Detector. Download it from the Releases page of
  https://github.com/WildHackers/community-fish-detector. License: AGPL-3.0 (fine for in-house use;
  list it in the AI guideline).
- BioCLIP doesn't go here. pybioclip downloads and caches it automatically the first time you
  run `python tools/check_ai.py`. Do that while you still have internet.

The weights aren't committed to git because they're big.
