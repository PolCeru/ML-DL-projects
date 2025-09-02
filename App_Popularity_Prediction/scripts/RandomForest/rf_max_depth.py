#!/usr/bin/python3

from pandas                  import read_csv
from sklearn                 import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix
import gc

data = read_csv("./sampled_data_1000000.csv")

# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(data.drop('Minimum Installs', axis=1),
                                                 data['Minimum Installs'],
                                                 test_size=0.2,
                                                 random_state=42,
                                                 stratify=data['Minimum Installs']  # Stratified split for class balance
                                                 )

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Define a parameter grid for Grid Search
param_grid = {
    'max_depth': [5, 10, 15, 18, 20, 22, 23],
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=stratified_cv, n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X_train, y_train)


# Print the scores for all parameter combinations
cv_results = grid_search.cv_results_
for mean_test_score, std_test_score, mean_train_score, params in zip(
    cv_results['mean_test_score'],
    cv_results['std_test_score'],
    cv_results['mean_train_score'],
    cv_results['params']
):
    print("Parameters:", params)
    print(f"Mean Test Score: {mean_test_score:.4f}")
    print(f"Std Test Score: {std_test_score:.4f}")
    print(f"Mean Train Score: {mean_train_score:.4f}")

# Get the best max_depth from Grid Search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

print("----------------------------------------------------------------------------------------")
