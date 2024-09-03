# Solution Report

1. What steps of the plan were performed and what steps were skipped (explain why)?
    - Skipped feature engineering AdditionalCharges once determining tenure caused data leakage (AdditionalCharges was derived from TotalCharges).
    - Once TotalCharges, AdditionalCharges, and Tenure were determined to cause data leakage, there was no longer a need to scale the features because there was only one continuous feature left.
    - Instead of upsampling the data once and passing it to each tuning function, I used the upsample function in each tuning function. This was necessary to properly perform cross validation with grid search.

2. What difficulties did you encounter and how did you manage to solve them?
    - Data leakage was the most difficult problem to solve. I had to carefully consider if TotalCharges was causing data leakage because TotalCharges is a value that in one sense is known at the time of prediction. When training however, TotalCharges is not an apples to apples comparison between the churned and unchurned customers. Consider that for churned customers TotalCharges is static known value, but for unchurned customers TotalCharges is a value that is still being accumulated.
    - I also had to consider if multiple features interacted to create data leakage. For example, Tenure and BeginDate could cause data leakage if used together because the model could infer if a customer churned based on the sum of the two features compared to the max date in the dataset.
    - Grid search cross validation with upsampled data also caused some difficulty. I had to create a custom approach to grid search cross validation that would upsample the data before each fold. This was necessary to prevent data leakage in the validation process.

3. What were some of the key steps to solving the task?
    - Determining the class imbalance and having multiple strategies to deal with it for each model.
    - Thorough exploratory data analysis to understand the data, recognize risks of data leakage, and guide feature engineering.
    - The use of grid search to efficiently tune hyperparameters.

4. What is your final model and what quality score does it have?
    - My final model was a CatBoostClassifier using the original data set (not upsampled) with the parameters: {  'learning_rate': 0.01,  
    'verbose': 0,  
    'auto_class_weights': 'Balanced',  
    'task_type': 'CPU',  
    'max_depth': 4,  
    'n_estimators': 1250,  
    'random_state': 42,  
    'cat_features': [3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    }
    - The final model is stored in the variable 'best_model', or its parameters can be pulled from the results DataFrame.
    - The final model has a ROC_AUC score of 85.55% and an accuracy score of 81.31%.
