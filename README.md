# Used Car Price Prediction
Predicting used-car selling prices with Linear Regression, Random Forest and XGBoost. Covers data cleaning, feature engineering, log transforms and model diagnostics (Python, scikit-learn).

## Problem
Given a car's age, mileage, engine, power, fuel type, brand and other features,
how well can we predict its selling price (in Indian rupees)?

## Data
8,130 listings with 13 features. [Add the source and link here.]
Cleaning reduced this to 6,716 records:
- 221 rows with missing values dropped (they were missing engine, power, seats
  and mileage together)
- 1,191 duplicate rows removed
- Impossible values corrected or removed (e.g. years of 200, 700 and 2050,
  a seat count of 0, one 14-seater outlier)
- Units stripped from `max_power`

## Method
- Feature engineering: `car_age` from year, brand and model split from name
- Log transforms for skewed columns, compared against untransformed features
- MinMax scaling, standardisation and one-hot encoding
- Models: Linear Regression, Random Forest, XGBoost (80/20 train/test split)
- Metrics: R², MAE, RMSE, plus Random Forest out-of-bag score and residual plots

## Results
| Model | R² (test) |
|---|---|
| Linear Regression | 0.716 |
| Linear Regression (log-transformed) | 0.852 |
| Random Forest | 0.949 |
| Random Forest (log-transformed) | 0.941 |
| XGBoost | 0.942 (MAE ≈ ₹70,600) |
| XGBoost (log-transformed) | 0.941 |

Log-transforming helped the linear model a lot, but made little difference to
the tree-based models. A learning curve on the Random Forest showed a training R²
of about 0.98 against a lower cross-validation score, i.e. overfitting, with the
validation score still rising as data was added.

## Limitations and what I'd do differently
- Scaling and encoding were fitted on the full dataset before the train/test
  split, which can leak information. A scikit-learn Pipeline would avoid this.
- Results come from one random 80/20 split. Cross-validation would be more reliable.
- No hyperparameter tuning.
- Price outliers were kept in, so a few luxury cars drive the large errors.

## Files
- `car_price_prediction.ipynb`: full analysis
- `requirements.txt`
