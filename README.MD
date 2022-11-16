### Genentech 404 Challenge (Kaggle) - 3rd place

This is my 3rd place solution to the [Genentech – 404 Challenge](https://www.kaggle.com/competitions/genentech-404-challenge)
The aim of this challenge was to develop an imputation algorithm capable of handling different types of missingness in tabular datasets containing clinical neuroimaging and cognitive measures features.

### General Strategy:

Neuroimaging and cognitive data can be highly variable, so having highly regularized models/low variance models seemed to be a reasonable starting point.
In addition, the most reliable information for a given subject is likely to be from the same subject, because of biological variability and measurement variability
 from different sites/scanners etc. The main strategy was therefore to weight heavily within subject information as much as possible whilst still borrowing information across subjects.

Initially I found that running a KNN imputation within subject to fill as many missing values as possible, then running a multiple imputation already gets a decent public LB score ~ 0.07. Improving on this was challenging as the sklearn iterative imputer
(multiple imputation) works extremely well with the default BayesianRidge estimator. The problem was that this did not take into account correlated within subject information at the iterative imputation stage.
Using a mixed effects model within the iterative imputation framework didn't seem to improve on the score. Fitting the model only across subjects would also be problematic as this would potentially be throwing away data.

The hyperimpute imputation package, where we fit different models for different columns on its own also did not improve on multiple imputation. However, combining the more flexible hyperimpute estimators with
hashing the subject ID did give a boost in performance (~0.065 on the public LB, 0.002 improvement on private LB). Using a hashing dimensionality of 64 seemed to work well,
 even if there were no clashes using a lower dimensionality

 Mixed models are designed for correlated data, so instead of just running knn within subjects, using a mixed model (random intercept) as a first step enables us to borrow information across subjects in the situation where we have some data for that subject/feature should be a better strategy than any within subject model.
 For this we needed to select feature subsets to fit, so this was a mostly manual hyperparameter search. Given the pattern of missing data, sensible combinations seemed to be predicting the neuroimaging features from the age + wholebrain + DX_group, or failing that
just age + DX_group. Then predicting the ventricles and whole brain features from the other neuro features + age + DX_group. Fitting the mixed models to the cognitive features (using neuroimaging as predictors) also
 gave a slight improvement.

 Finally, there was a slight improvement by recognising that KNN where k = 2 was optimal for most features but setting to k = 1 seemed to work better for DX_group.

### Final Submission Details:
1. Fill in missing age from the time-point information.

2. Fill in the missing data that (should be) constant across subject: ('APOE4', 'PTGENDER_num', 'PTEDUCAT').

3. Fit (and predict) mixed models using [GPBoost](https://github.com/fabsig/GPBoost) where some subject information is available for each column: \
 i. Fit neuroimaging features from wholebrain features + age + DX_group, failing that age + dx features, or failing that just age. \
 ii. fit ventricle/wholebrain features using other neuroimaging features + age + DX_group. failing that + age + DX_group features or just age. \
 iii. fit cognitive features using neuro_features + age + DX_group, failing that ventricles + whole brain + age + DX_group, or just whole brain + age_dx_features 

4. Run KNN within subject, where K=2, except for DX_group where K=1.

5. Hash subject IDs and run [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) to fill out the remaining missing data using regression seeds: xgboost_regressor, catboost_regressor, random_forest_regressor, linear_regression.


### Additional considerations:
- For the mixed model, I used the gpboost package, but I also investigated other packages such as statsmodels and Bayesian models using PyMC.
- Adding random slopes did not seem to improve the results.
- Adding the age * DX interaction, or quadratic age terms also didn't improve performance.
- I didn't find any major differences in missing data patterns between test A and test B, but even so it would be challenging to take these into account.