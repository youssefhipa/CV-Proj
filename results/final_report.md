# Computer Vision Course Project
## Comparative Evaluation of Harris Corner Detector and SIFT Keypoint Detector

**Submitted by:** Khaled Amr and Youssef Hipa  
**Generated:** 2025-12-11 23:20:50  
**Images processed:** 5

---
## Project Overview
Controlled comparison of Harris and SIFT detectors to study stability, robustness, and repeatability under scale, rotation, illumination, blur, and noise transformations. The pipeline follows the course brief and mirrors the Kaggle “Feature Extraction and Matching” assignment; current run uses 5 building/architecture photos placed in `data/original/`.

### Learning Objectives (met)
- Understand Harris corner detection and SIFT scale-space keypoint extraction.
- Analyze scale, rotation, illumination, blur, and noise invariance.
- Conduct quantitative evaluation (counts, densities, repeatability).
- Use benchmark-style building imagery (aligned with Kaggle dataset theme).
- Present results in a structured scientific report with code + visuals.

---
## 1) Implementation Summary
- **Harris:** gradients → second-moment matrix → Harris response → NMS → threshold; visualized as corner overlays.
- **SIFT:** OpenCV SIFT for multiscale keypoints with orientation; visualized as dots (clean overlays).
- Parameter notes: Harris k=0.05, window=3, threshold=0.003, nms=3; SIFT uses OpenCV defaults.

---
## 2) Keypoint Count Comparison (Task A)
- Total keypoints — Harris: 3,398, SIFT: 61,308
- Average per image — Harris: 679.6, SIFT: 12261.6
- Range — Harris: 425–1209, SIFT: 3686–30055
- Density (avg) — Harris: 0.000297 kp/pixel, SIFT: 0.004874 kp/pixel
- SIFT/Harris keypoint ratio: 18.0×

### Per-image breakdown
| Image | Harris | SIFT | Difference | Harris Density | SIFT Density |
|-------|--------|------|------------|----------------|--------------|
| roma_1.jpg | 473 | 9858 | 9385 | 0.000246 | 0.005134 |
| roma_2.jpg | 1209 | 30055 | 28846 | 0.000304 | 0.007549 |
| building_3.jpg | 722 | 3686 | 2964 | 0.000376 | 0.001920 |
| building_2.jpg | 569 | 6624 | 6055 | 0.000333 | 0.003880 |
| building_1.jpg | 425 | 11085 | 10660 | 0.000226 | 0.005888 |

**Observation:** SIFT produces far more keypoints; Harris is selective on high-contrast corners (std Harris 316.9 vs SIFT 10357.2).

---
## 3) Robustness Analysis (Task B)
### Robustness Comparison

| Transformation | Harris Repeatability | SIFT Repeatability | Winner |
|----------------|---------------------|-------------------|--------|
| Scale | 0.632 | 0.717 | **SIFT** |
| Rotation | 0.035 | 0.295 | **SIFT** |
| Brightness | 0.617 | 0.661 | **SIFT** |
| Blur | 0.365 | 0.576 | **SIFT** |
| Noise | 0.744 | 0.592 | **Harris** |

### Overall Robustness
- Harris wins: 1 / 5
- SIFT wins: 4 / 5
- Avg repeatability — Harris: 0.479, SIFT: 0.568


---
## 4) Keypoint Distribution (Task C)
- Harris: concentrated on corners/edges of façades and windows.
- SIFT: broad coverage including textured regions; better scale/rotation handling.
- Visuals: see `results/comparison/` and `results/distribution/` for overlays and heatmaps.

---
## 5) Deliverables & Alignment with Brief
- Code: full Python pipeline in `src/` and `run_experiments.py`.
- Visualizations: `results/harris/`, `results/sift/`, `results/comparison/`, `results/distribution/`, `results/robustness/`, `results/plots/`.
- Metrics: `results/metrics/basic_detection_results.csv`, `results/metrics/*_robustness_results.json`, `results/metrics/*_robustness_summary.csv`, `results/metrics/final_summary_report.json`.
- Report: this file (`results/final_report.md`) with summary, dataset note, experiment design, results, and recommendations.

---
## 6) Recommendations
- Use **Harris** for fast, corner-focused tasks on man-made structures.
- Use **SIFT** when you need scale/rotation robustness and richer coverage.
- Hybrid approach: Harris for speed, SIFT descriptors for robustness/matching.

*Report generated automatically by Computer Vision Project System*
