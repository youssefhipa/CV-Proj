# Computer Vision Project Report
## Comparative Evaluation of Harris Corner Detector and SIFT Keypoint Detector

**Date:** 2025-12-11 22:28:00
**Images Processed:** 5

---

## 1. Keypoint Count Comparison (Task A)

### Summary Statistics:
- **Total Harris Keypoints:** 292
- **Total SIFT Keypoints:** 61,308
- **Average per Image:**
  - Harris: 58.4 keypoints
  - SIFT: 12261.6 keypoints
- **Range (min-max):**
  - Harris: 38 - 88
  - SIFT: 3686 - 30055

### Density Analysis:
- **Harris Density:** 0.000026 keypoints/pixel
- **SIFT Density:** 0.004874 keypoints/pixel

### Observations:
- Harris detector typically finds fewer keypoints than SIFT
- Harris shows more consistent results across different images

---

## 2. Robustness Analysis (Task B)

### Robustness Comparison:

| Transformation | Harris Repeatability | SIFT Repeatability | Winner |
|----------------|---------------------|-------------------|--------|
| Scale | 0.632 | 0.717 | **SIFT** |
| Rotation | 0.035 | 0.295 | **SIFT** |
| Brightness | 0.617 | 0.661 | **SIFT** |
| Blur | 0.365 | 0.576 | **SIFT** |
| Noise | 0.757 | 0.591 | **Harris** |


### Overall Robustness:
- **Harris Wins:** 1 out of 5 tests
- **SIFT Wins:** 4 out of 5 tests
- **Average Repeatability:**
  - Harris: 0.481
  - SIFT: 0.568

### Key Findings:
1. **Scale Changes:** SIFT performs better
2. **Rotation:** SIFT performs better
3. **Illumination:** SIFT performs better
4. **Blur:** SIFT performs better
5. **Noise:** Harris performs better

---

## 3. Keypoint Distribution Analysis (Task C)

### Observations:
1. **Harris Detector:**
   - Concentrates on corners and high-contrast edges
   - More uniformly distributed in texture-rich areas
   - Sensitive to local intensity changes

2. **SIFT Detector:**
   - Detects blob-like structures at multiple scales
   - More selective, focusing on distinctive locations
   - Better distributed across different image regions

3. **Comparison:**
   - Harris is better for geometric structures (buildings, windows)
   - SIFT is better for natural textures and scale variations
   - Harris keypoints are more dense but less distinctive
   - SIFT keypoints are fewer but more robust to transformations

---

## 4. Conclusion

### Strengths of Harris:
- Faster computation
- Better for corner-like features
- More consistent in structured environments
- Less parameter-sensitive

### Strengths of SIFT:
- Scale and rotation invariant
- Better for natural scenes
- More distinctive descriptors
- Robust to viewpoint changes

### Recommendations:
1. **Use Harris** for: Building corners, chessboard patterns, man-made structures
2. **Use SIFT** for: Natural scenes, object recognition, cases with scale changes
3. **Consider hybrid approach** for optimal results

---

## Files Generated:

### Visualizations:
1. `results/harris/` - Harris keypoint detection results
2. `results/sift/` - SIFT keypoint detection results
3. `results/comparison/` - Side-by-side comparisons
4. `results/distribution/` - Spatial distribution analysis
5. `results/robustness/` - Transformation examples
6. `results/plots/` - All analysis plots and graphs

### Data Files:
1. `results/metrics/basic_detection_results.csv` - Keypoint counts
2. `results/metrics/*_robustness_results.json` - Detailed robustness data
3. `results/metrics/*_robustness_summary.csv` - Robustness comparison
4. `results/metrics/final_summary_report.json` - Complete summary

---
*Report generated automatically by Computer Vision Project System*
