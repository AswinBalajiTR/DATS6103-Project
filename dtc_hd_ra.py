# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#%%
# Prepare the data for modeling
# Since you want to predict 'heart_disease', we set it as the target
X = df.drop(columns=['id', 'stroke', 'heart_disease'], errors='ignore')  # Excluding 'heart_disease' and 'stroke' columns
y = df['heart_disease']

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

#%%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build the Decision Tree model
dt_clf = DecisionTreeClassifier(random_state=42)


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None],  # Remove 'auto' and use 'sqrt', 'log2', or None
}

grid_search = GridSearchCV(estimator=dt_clf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Use the best estimator from the grid search
best_dt_clf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_dt_clf.predict(X_test)
y_pred_proba = best_dt_clf.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve

#%%
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc:.2f}")

#%%
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#%%
# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f"AUC-ROC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()