
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from statsmodels.tools.tools import add_constant

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
#%%
df=pd.read_csv("stroke_data.csv")
df.head()

#%%
print("(Rows,columns):",df.shape)

#%%
df.dtypes

#%%
for i in df.columns:
    print(i,":",df[i].nunique())

print("Gender :")
df['gender'].value_counts()

df["gender"]=df["gender"].replace("Other",np.nan)


#%%
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Visualize outliers with boxplots
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

#%%
for i in df.columns:
    print(i,":",df[i].isna().sum())


#%%
df["bmi"]=df["bmi"].replace(np.nan,df["bmi"].mean())
df["gender"]=df["gender"].replace(np.nan,df["gender"].mode()[0])

for i in df.columns:
    print(i,":",df[i].isna().sum())


#%%
df1=df.copy()
df1.head()

#%%
categorical_columns = df.select_dtypes(include=['object']).columns

# Encode all categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df.head()

#%%

# Function to perform Chi-Square test for categorical variables
def chi_square_test(data, target_variable):
    results = {}
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if col != target_variable:  # Exclude the target variable
            contingency_table = pd.crosstab(data[col], data[target_variable])
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
            results[col] = p  # Store p-values
    return results

# Specify target variable (replace 'stroke' with the actual target variable in your dataset)
target_variable = 'stroke'  # Update if your dataset has a different name
chi_square_results = chi_square_test(df1, target_variable)

# Display statistically significant results
significant_results = {k: v for k, v in chi_square_results.items() if v < 0.05}
print("Statistically significant correlations with the target variable:")
print(significant_results)



#%%
# Heatmap of the p-values for all categorical variables
p_values_df = pd.DataFrame.from_dict(chi_square_results, orient='index', columns=['p_value']).sort_values(by='p_value')
plt.figure(figsize=(10, 6))
sns.heatmap(p_values_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'P-value'})
plt.title('Chi-Square Test P-Values for Categorical Variables')
plt.xlabel('P-value')
plt.ylabel('Categorical Variables')
plt.show()


#%%
# Bar plot to show distribution of stroke occurrence across significant categorical variables
for col in significant_results.keys():
    plt.figure(figsize=(8, 6))
    category_percentages = df.groupby(col)[target_variable].value_counts(normalize=True).unstack() * 100
    category_percentages.plot(kind='bar', stacked=True)
    plt.title(f'Stroke Occurrence by {col}')
    plt.ylabel('Percentage')
    plt.xlabel(col)
    plt.legend(title=target_variable, loc='upper right')
    plt.show()

#%%
# Preparing the data for modeling
X = df.drop(columns=['id', 'stroke', 'heart_disease'], errors='ignore')  
y = df['heart_disease']

# one-hot encoding
X = pd.get_dummies(X, drop_first=True)

#%%
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Building the Decision Tree model
dt_clf = DecisionTreeClassifier(random_state=42)


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None],  # (Note: Run back to check)
}

grid_search = GridSearchCV(estimator=dt_clf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV (Cross verify)
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Use the best estimator (grid search)
best_dt_clf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_dt_clf.predict(X_test)
y_pred_proba = best_dt_clf.predict_proba(X_test)[:, 1]  

#%%
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# evaluation metrics (print statements)
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
# Plotting the ROC curve
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