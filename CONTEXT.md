# Domain Context: Wellbore Geology Prediction

**Competition Link:** [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview)

## Overview
The goal of this project is to develop machine learning models that predict the geology encountered along a horizontal wellbore. These models identify favorable geological layers from drilling data to help guide well placement accurately during operations, minimizing resource waste and improving safety.

## Key Concepts & Glossary
- **TVT (True Vertical Thickness)**: The manually interpreted geological position for each 1 ft of the lateral well. This is the **target variable** to predict.
- **MD (Measured Depth)**: The total length of the wellbore from the surface (in feet).
- **X (Easting)** and **Y (Northing)**: Spatial coordinates in the horizontal plane (ft).
- **Z (True Vertical Depth)**: The vertical distance below sea level (ft).
- **GR (Gamma Ray)**: A log measuring the natural radioactivity of the rock (API). It is a key feature used for geological correlation.
- **Typewell**: A vertical reference log used for geological correlation with the associated horizontal well.
- **Evaluation Zone**: A specific region within the horizontal well where the `TVT` target is hidden (replaced with `NaN` in `TVT_input`) and must be predicted.
- **Geological Formations**: Various layers/units of rock, such as ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA.
- **Self-Correlation DTW**: The process of using Dynamic Time Warping (DTW) to match the Gamma Ray (GR) signature of the evaluation zone against the known past of the *same* horizontal well, rather than relying solely on the Typewell.
- **Sliding Window Matching**: A feature engineering technique where a fixed-size window of recent GR readings in the evaluation zone is swept across the known past to find the minimum DTW distance, yielding the most probable past `TVT` equivalent.
- **Multi-Resolution DTW**: Applying Sliding Window Matching simultaneously across different window sizes (e.g., 20ft, 50ft, 100ft) to capture both high-frequency localized geological features and low-frequency global trends.
- **Geo-Hybrid Features**: Features constructed by explicitly combining the output of DTW matches with spatial coordinates (e.g., calculating the $\Delta Z$ between the current evaluation point and the historical DTW match point) to physically ground the model's perception of geological dip.

## Datasets
The data consists of horizontal well trajectories and vertical reference logs (Typewells) organized into `train/` and `test/` directories. Each well is identified by a unique 8-character hash (e.g., `015fe0d2`).

### Horizontal Wells (`{WELLNAME}__horizontal_well.csv`)
Contains trajectory, geological surfaces, and log data.
- **Features**: `MD`, `X`, `Y`, `Z`, `GR`
- **Geological surfaces (Training only)**: Predicted depth of formations like `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`.
- **Target**: `TVT` (True Vertical Thickness). Note that `TVT_input` is provided as a feature but will contain `NaN` values in the evaluation zone.

### Typewells (`{WELLNAME}__typewell.csv`)
Vertical reference logs for the well.
- **TVT**: Vertical Depth Index (ft) corresponding to the geological position of the associated horizontal well.
- **GR**: Gamma Ray signature used for correlation.
- **Geology**: Categorical label indicating the geological unit.

## Evaluation
- **Metric**: Root Mean Squared Error (RMSE) on the predicted `TVT` values in the evaluation zone (after the Prediction Start (PS) point).
- **Submission Format**: A CSV file containing `id` (formatted as `{WELLNAME}_{row_index}`) and `tvt` (predicted value).

## The Physics of Predicting TVT
As a horizontal well drills through the earth, geological layers might be flat, dipping up, or dipping down. TVT is predicted by matching the **Gamma Ray (GR) signature** of the horizontal well to a known reference:
- If the horizontal GR signature matches the Typewell GR signature perfectly moving forward, the geology is **flat** (TVT is constant).
- If the signature stretches or compresses, the geology is **dipping** (TVT is increasing or decreasing).

## Crucial Hints from the Organizers (from PPTX)
1. **Self-Correlation is better than Typewell-Correlation**: The GR resolution in the horizontal well is better than the typewell. *It may be better to use GR data from the horizontal well before the PS point, combined with deeper TVT data, to correlate the rest of the lateral.*
2. **Spatial Awareness**: Geological dips behave similarly in neighboring wells. The azimuth (direction) of drilling affects the expected dip. **Using offset (neighboring) wells can help predict the geology of the current well!**

## Working Note Award Criteria
A strong, winning solution must not only score well on the leaderboard but also demonstrate rigorous physical reasoning. Key criteria for evaluation include:
1. **Breadth and Depth of Exploration**: Thoroughly documented experiments exploring genuinely different approaches (e.g., feature sets, modeling strategies). Negative results and lessons learned should be well-documented.
2. **Insights About the Data and Wells**: Significant observations about the data, behavioral differences across wells, and how analytical methods were tailored to specific well properties.
3. **Physical Meaningfulness**: The final solution should represent a plausible, physically robust interpretation of the underlying geological data, rather than an over-optimized ensemble that merely chases the evaluation metric.
4. **Contribution of Individual Ideas**: Clear quantification of how each major idea, feature, or component improved validation performance.
5. **Uncertainty Estimation**: The model should ideally estimate its own confidence, communicating where predictions are reliable and where it may be prone to error.
