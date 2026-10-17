# MSCS 634 – Lab 4: Regression Analysis with Regularization Techniques

**Name:** Sowmya Cheruku  
**Course:** MSCS 634 – Advanced Big Data and Data Mining  
**University:** University of the Cumberlands  
**Instructor:** Satish Penmatsa

## Purpose
The goal of this lab was to predict diabetes disease progression using the scikit-learn Diabetes dataset (442 patients, 10 features). I built Simple Linear, Multiple, and Polynomial Regression models, then added Ridge and Lasso regularization to see whether it reduces overfitting. All models were scored with MAE, MSE, RMSE and R² on the same 80/20 train/test split.

## Files
- `MSCS_634_Lab_4.ipynb` – full notebook with code, plots and written analysis
- `README.md` – this file

## Key Insights
- **Simple linear regression (BMI only)** had a test R² of about 0.23, so one feature is not enough.
- **Multiple regression** improved this to about 0.45 (RMSE ≈ 53.9).
- **Polynomial regression** overfit. Degree 2 was slightly worse than multiple regression on the test set, and degree 3 on all features had a test R² of about -14.6 (285 features for only 353 training rows).
- **Ridge and Lasso** gave small improvements over plain multiple regression (Lasso RMSE ≈ 52.9, R² ≈ 0.47). Lasso set `s2` and `s4` to zero. Regularization made a much bigger difference on the overfitting polynomial model, where Lasso brought R² back up to about 0.48.
- **Alpha matters:** small alpha behaves like ordinary regression. Large alpha shrinks the coefficients too much and underfits (Lasso with alpha = 100 predicted just the mean).
- `bmi`, `s5` and `bp` were the strongest predictors, and several serum features (`s1`/`s2`, `s3`/`s4`) are highly correlated with each other.

## Challenges and Decisions
- The dataset is already scaled and has no missing values, so little cleaning was needed. I still used `StandardScaler` for Ridge and Lasso since the penalty depends on coefficient size.
- To show over/underfitting clearly, I tested polynomial degrees on `bmi` alone (so the curves can be plotted) and then on all features (to compare against multiple regression).
- I used `RidgeCV` and `LassoCV` to choose alpha instead of guessing. The alphas for the Ridge/Lasso + degree 3 polynomial models were picked by hand, which is a limitation.
- With only 89 test rows and a single split, small differences between models may not be reliable. Cross-validation would be a good next step.

## How to Run
```
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook MSCS_634_Lab_4.ipynb
```
