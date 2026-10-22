# FLAME 2 thermal labels: hand-check validation

_2026-10-21 · Milestone M4 (Phase 4) · data in `data/flame2/handcheck/` (not in git) · scripts: `export_handcheck.py`, `score_handcheck.py`, `tune_rainbow_mask.py`_

## Question

FLAME 2 has frame-level labels only, so the training masks come from the thermal frames, using `hot_mask` in `src/embereye/data/flame2.py`. Its threshold is set relative to each frame and depends on the palette. Almost every fire frame gets a mask, but that is by design and says nothing about quality. Do the masks match what a person marks as fire closely enough to train on?

## Method

- **Sample:** 48 fire frames, 8 per stratum. A stratum is a palette variant (inferno, rainbow with a mostly red background, rainbow with a mostly magenta one) combined with the `labels.txt` smoke flag (YY = smoke fills at least half the RGB frame, YN = it doesn't). The frames were picked without randomness:
  - every one of the 18 fire blocks in `labels.txt` is covered
  - 11 frames are from frame 45,000 on
  - near-duplicate views (the drone hovers) were skipped
- **Labelling:** one labeller (Will), in labelme 6.3.1, on the thermal frames, without seeing `hot_mask`'s output. The rules were fixed before starting:
  - inferno frames: yellow/white, plus orange touching it
  - rainbow frames: orange/yellow, plus green touching it
  - no confetti (scattered specks), haze, or specks under 3 px
  - touching patches merged into one region
- **Score:** IoU between the hand mask and `hot_mask`, with the frame gate on (`max_frame_median=0.25`) and the other parameters at their defaults. Also IoU with the exported label, i.e. the YOLO-seg outlines filled back in. Go/no-go for the export: mean IoU ≥ 0.5 in every stratum.

## Results

| Stratum | n | Mean IoU | Median | Min | 95% CI of mean | Hand area, median % | Mask area, median % |
|---|---|---|---|---|---|---|---|
| inferno YY | 8 | 0.716 | 0.722 | 0.637 | 0.67–0.77 | 8.7 | 9.0 |
| inferno YN | 8 | 0.735 | 0.766 | 0.651 | 0.69–0.78 | 6.2 | 7.6 |
| rainbow-red YY | 8 | 0.532 | 0.504 | 0.425 | 0.47–0.59 | 3.4 | 4.9 |
| rainbow-red YN | 8 | 0.567 | 0.647 | 0.234 | 0.42–0.71 | 2.3 | 2.9 |
| rainbow-magenta YY | 8 | 0.613 | 0.615 | 0.499 | 0.54–0.68 | 1.7 | 2.1 |
| rainbow-magenta YN | 8 | 0.624 | 0.646 | 0.470 | 0.54–0.71 | 0.9 | 1.1 |
| **all** | **48** | **0.631** | **0.653** | **0.234** | **0.60–0.67** | 3.0 | 4.3 |

Every stratum passes, so the export went ahead.

- **Inferno** frames agree best.
- **Rainbow** frames agree less well, and `hot_mask` marks more area than the hand masks do: the median ratio of mask area to hand area is 1.3–1.5 on rainbow, against 1.05–1.15 on inferno. It counts scattered specks, and cyan and blue pixels, that the labelling rule leaves out.

### The export was losing part of the mask (fixed)

| Setting | Label vs raw mask (IoU) | Label vs hand masks: inferno / rainbow-red / rainbow-magenta (mean IoU) |
|---|---|---|
| Old: drop regions with outline area under 20; simplify with ε = 1.0 | 0.86 | 0.73 / 0.57 / 0.61 |
| **New: keep every region; ε = 0.5** | **0.97** | **0.72 / 0.55 / 0.62** |

`hot_mask` already drops regions under 20 **pixels**. The old export then dropped regions whose **outline** enclosed less than 20. An outline encloses less area than its pixel count, so regions `hot_mask` had kept were thrown away. The worst cases were frames 45822 and 52892, which went from 0.47 and 0.48 for the mask to 0.36 and 0.34 for the label.

The old export happened to score higher on rainbow-red, because losing small regions removed some confetti. It scored lower on rainbow-magenta, where real fires are small. Filtering belongs in `hot_mask`, so the export now reproduces the mask. Every stratum's label IoU is within 0.01 of its mask IoU.

### Rainbow tuning, judged on held-out frames

**Protocol, fixed before any result was seen:**
- Within each stratum, alternate frames went to a tuning half and a held-out half (4 + 4). Only rainbow settings were allowed to change.
- 36 candidates were tried: combinations of a higher `min_contrast`, a seed rule (keep a region only if it reaches yellow/orange), and a 7×7 local-density filter against confetti.
- The best candidate on the 16 rainbow tuning frames was scored once on the held-out frames.
- It would be adopted only if all of these held:
  - held-out rainbow IoU improves by at least 0.03
  - no rainbow stratum drops by more than 0.02
  - the share of rainbow fire frames left with an empty mask rises by at most 1 point

The chosen candidate was `min_contrast=0.45`, seed at 0.85 and density at 0.3. On the held-out half:

| Stratum | n | Current | Tuned | Change |
|---|---|---|---|---|
| rainbow-red YY | 4 | 0.524 | 0.611 | +0.086 |
| rainbow-red YN | 4 | 0.652 | 0.700 | +0.048 |
| rainbow-magenta YY | 4 | 0.623 | 0.647 | +0.024 |
| rainbow-magenta YN | 4 | 0.589 | 0.550 | **−0.039** |
| all rainbow | 16 | 0.597 | 0.627 | +0.030 |

The empty-mask share went from 0.1% to 0.3%, measured on every 50th rainbow fire frame (n = 720).

**Decision: keep the current settings.** The tuned filter helps red-background frames but removes some of the tiny real fires in magenta YN frames, which breaks the rule against any stratum dropping more than 0.02. Applying the filter to red-background frames only might keep the gain without the loss. That is a new hypothesis, though, and the held-out half has now been used, so testing it fairly needs new hand labels.

## Caveats

- **One labeller.** There is no inter-rater agreement. IoU measures agreement with one person's reading of the thermal frames, and that reading is itself a judgement, for example about where flame ends and hot ground begins.
- **Tiny fires make IoU unstable.** On the 11 frames where the hand mask covers under 1% of the image (mostly rainbow YN and magenta), a few pixels decide the score. Those frames average 0.53 (sd 0.15), against 0.66 (sd 0.10) for the other 37. The two lowest scores, frames 40021 and 52352 (both rainbow-red YN), have hand masks of 0.2% of the image.
- **Small samples.** There are 8 frames per stratum. The 95% intervals for rainbow-red YY (0.47–0.59) and YN (0.42–0.71) reach below 0.5. The go decision rests on the means.
- **The tuning was held out.** Settings were chosen on half the frames and are reported on the other half. Further tuning needs fresh labels, for example from the planned hand-labelled test set.
- **Thermal only.** The masks were checked against the thermal frames. Whether they line up with the RGB frames, which the RGB detector trains on, is still open. The label audit found best-fit shifts that vary from frame to frame.
- **The 10% cap.** Two inferno frames, 14691 and 16751, have hand masks over 10% of the image. That is more than `hot_mask` can mark (`top_fraction=0.10`), which caps their IoU (0.65 and 0.76).

## Reproduce

```bash
python scripts/export_handcheck.py      # picks the 48 frames (only if the folder has no labels yet)
python scripts/score_handcheck.py       # writes scores.md / scores.csv to data/flame2/handcheck/
python scripts/tune_rainbow_mask.py     # the held-out tuning above
```
