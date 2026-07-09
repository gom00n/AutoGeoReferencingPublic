# Archive

Retired code and artifacts. Nothing here is imported by the active pipeline
(verified 2026-06-10). Kept for reference; restore by moving back to
`scripts/` — they expect to run from there.

## Legacy detectors (superseded by template matching + CNN, see CLAUDE.md)

- `triangle_detector.py` — HOG+SVM sliding-window detector, plus contour
  detector and OCR-based name/height matching. Uses `triangle_classifier.pkl`.
- `template_matcher.py` — first-generation full-image multi-scale template
  matcher (binary templates).
- `feature_classifier.py` — gradient-boosting classifier on hand-crafted
  features. Wrote `triangle_gbm.pkl`.
- `triangle_classifier.pkl`, `triangle_gbm.pkl` — model files for the above
  (the active model is `scripts/triangle_classifier.pth`).

## One-off / superseded tools

- `extract_templates.py` — extracted the original triangle templates from
  M5_4048 control points (templates now live in `scripts/templates/`).
- `generate_yolo_data.py` — YOLO-format dataset generator (YOLO approach
  was never pursued; the QA crop flow superseded its classification crops).
- `generate_review.py` — HTML review page for DB-matched detections
  (superseded by `qa_gui.py` / `curate.py` / `review_positives.py`).

## Stale generated artifacts

- `curate_qa_candidates.html`, `review_positives.html` — labeling pages
  generated 2026-03-31; their exported label JSONs are tracked in
  `training_data/Data_labeling/` and were already applied. Regenerate fresh
  ones with `curate.py` / `review_positives.py`.
