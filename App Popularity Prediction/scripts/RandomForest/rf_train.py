#!/usr/bin/python3
from pandas                  import read_csv
from sklearn                 import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix
import gc

data = read_csv("./preprocessed.csv")

# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(data.drop('Minimum Installs', axis=1),
                                                 data['Minimum Installs'],
                                                 test_size=0.2,
                                                 random_state=42,
                                                 stratify=data['Minimum Installs']  # Stratified split for class balance
                                                 )

# Create a Random Forest model with the best parameters
best_rf_model = RandomForestClassifier(max_depth=10, n_estimators=1000, min_samples_split=15, min_samples_leaf=4, random_state=42, n_jobs=-1)

# Train the model with the best parameters on the entire training set
best_rf_model.fit(X_train, y_train)

# Test
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Train
y_train_pred = best_rf_model.predict(X_train)
test_accuracy = accuracy_score(y_train, y_train_pred)
print("Train accuracy:", test_accuracy)

# Generate classification report
class_report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:")
print(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")
