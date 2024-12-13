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
# Binning age into ranges for better visualization
age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
df1['age_group'] = pd.cut(df1['age'], bins=age_bins, labels=age_labels, right=False)

stroke_data = df1[df1['stroke'] == 1]

# Count stroke occurrences by age group
stroke_counts_by_age = stroke_data['age_group'].value_counts().sort_index()

# Plot: Stroke occurrences by age group
plt.figure(figsize=(10, 6))
sns.barplot(
    x=stroke_counts_by_age.index,
    y=stroke_counts_by_age.values,
    palette="magma",  # Creative and vibrant color palette
    edgecolor="black"
)
plt.title('Stroke Occurrences by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Stroke Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=stroke_data,
    x='age',
    fill=True,
    color='orange',
    alpha=0.6
)
plt.title('Density of Stroke Occurrences by Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# Filter data to include only stroke occurrences
stroke_data = df1[df1['stroke'] == 1]

# Count stroke occurrences by gender
stroke_counts_by_gender = stroke_data['gender'].value_counts()

# Plot: Stroke occurrences by gender
plt.figure(figsize=(8, 6))
sns.barplot(
    x=stroke_counts_by_gender.index,
    y=stroke_counts_by_gender.values,
    palette="Spectral",  # Vibrant and colorful palette
    edgecolor="black"
)
plt.title('Stroke Occurrences by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Stroke Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# Filter data to include only stroke occurrences
stroke_data = df1[df1['stroke'] == 1]

# 1. Hypertension vs Stroke
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='hypertension',
    palette='viridis',
    edgecolor='black'
)
plt.title('Hypertension vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Hypertension (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Hypertension vs Gender
# Create a grouped bar plot: Hypertension vs Gender for Stroke Cases
plt.figure(figsize=(10, 6))
sns.countplot(
    data=stroke_data,
    x='hypertension',
    hue='gender',
    palette='viridis',
    edgecolor='black'
)
plt.title('Hypertension vs Gender for Stroke Cases', fontsize=16)
plt.xlabel('Hypertension (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Gender')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# 1. Ever Married vs Stroke
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='ever_married',
    palette='coolwarm',
    edgecolor='black'
)
plt.title('Ever Married vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Ever Married (No/Yes)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Ever Married vs Gender
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='ever_married',
    hue='gender',
    palette='coolwarm',
    edgecolor='black'
)
plt.title('Ever Married vs Gender (Stroke Cases Only)', fontsize=16)
plt.xlabel('Ever Married (No/Yes)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Gender')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Ever Married vs Hypertension
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='ever_married',
    hue='hypertension',
    palette='viridis',
    edgecolor='black'
)
plt.title('Ever Married vs Hypertension (Stroke Cases Only)', fontsize=16)
plt.xlabel('Ever Married (No/Yes)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Hypertension (0 = No, 1 = Yes)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 4. Ever Married vs Average Glucose Level (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=stroke_data,
    x='ever_married',
    y='avg_glucose_level',
    palette='coolwarm'
)
plt.title('Ever Married vs Average Glucose Level (Stroke Cases Only)', fontsize=16)
plt.xlabel('Ever Married (No/Yes)', fontsize=14)
plt.ylabel('Average Glucose Level', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Ever Married vs Heart Disease
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='ever_married',
    hue='heart_disease',
    palette='magma',
    edgecolor='black'
)
plt.title('Ever Married vs Heart Disease (Stroke Cases Only)', fontsize=16)
plt.xlabel('Ever Married (No/Yes)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Heart Disease (0 = No, 1 = Yes)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# 1. Work Type vs Stroke
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='work_type',
    palette='coolwarm',
    edgecolor='black'
)
plt.title('Work Type vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Work Type', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Work Type vs Gender
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='work_type',
    hue='gender',
    palette='viridis',
    edgecolor='black'
)
plt.title('Work Type vs Gender (Stroke Cases Only)', fontsize=16)
plt.xlabel('Work Type', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%

# 1. Residence Type vs Stroke
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='Residence_type',
    palette='coolwarm',
    edgecolor='black'
)
plt.title('Residence Type vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Residence Type (Urban/Rural)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Residence Type vs Hypertension
plt.figure(figsize=(8, 6))
sns.countplot(
    data=stroke_data,
    x='Residence_type',
    hue='hypertension',
    palette='magma',
    edgecolor='black'
)
plt.title('Residence Type vs Hypertension (Stroke Cases Only)', fontsize=16)
plt.xlabel('Residence Type (Urban/Rural)', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Hypertension (0 = No, 1 = Yes)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%

# 1. Glucose Level vs Stroke
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=stroke_data,
    x='stroke',
    y='avg_glucose_level',
    palette='coolwarm'
)
plt.title('Average Glucose Level vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Stroke (Always 1 for stroke cases)', fontsize=14)
plt.ylabel('Average Glucose Level', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# KDE Plot: Distribution of Average Glucose Levels for Stroke Cases
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=stroke_data,
    x='avg_glucose_level',
    fill=True,
    color='orange',
    alpha=0.6,
    linewidth=2
)
plt.title('Density Plot of Average Glucose Levels (Stroke Cases Only)', fontsize=16)
plt.xlabel('Average Glucose Level', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 2. BMI vs Stroke
# KDE Plot: Distribution of Average Glucose Levels for Stroke Cases
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=stroke_data,
    x='bmi',
    fill=True,
    color='orange',
    alpha=0.6,
    linewidth=2
)
plt.title('Density Plot of BMI (Stroke Cases Only)', fontsize=16)
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Smoked vs Stroke
plt.figure(figsize=(10, 6))
sns.countplot(
    data=stroke_data,
    x='smoking_status',
    palette='viridis',
    edgecolor='black'
)
plt.title('Smoking Status vs Stroke (Stroke Cases Only)', fontsize=16)
plt.xlabel('Smoking Status', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Smoker vs Gender
plt.figure(figsize=(10, 6))
sns.countplot(
    data=stroke_data,
    x='smoking_status',
    hue='gender',
    palette='coolwarm',
    edgecolor='black'
)
plt.title('Smoking Status vs Gender (Stroke Cases Only)', fontsize=16)
plt.xlabel('Smoking Status', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Gender')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Smoker vs Hypertension
plt.figure(figsize=(10, 6))
sns.countplot(
    data=stroke_data,
    x='smoking_status',
    hue='hypertension',
    palette='Set2',
    edgecolor='black'
)
plt.title('Smoking Status vs Hypertension (Stroke Cases Only)', fontsize=16)
plt.xlabel('Smoking Status', fontsize=14)
plt.ylabel('Count of Stroke Cases', fontsize=14)
plt.legend(title='Hypertension (0 = No, 1 = Yes)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%%
correlation_matrix = df.corr()

# Define custom colormap: green for low, white in center, pink for high
custom_cmap = sns.diverging_palette(150, 320, as_cmap=True)

# Visualize correlation matrix using a heatmap with the custom colormap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=custom_cmap,
    cbar=True,
    square=True,
    center=0
)
plt.title("Correlation Matrix - Multicollinearity Check", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#%%
# Select only numerical columns for VIF calculation
numerical_cols = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')

# Add a constant for VIF calculation
X = add_constant(numerical_cols)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Drop the constant column
vif_data = vif_data[vif_data['feature'] != 'const']

# Display VIF
print("VIF values for features:")
print(vif_data)

# Identify multicollinear pairs (VIF > 5 as a threshold)
multicollinear_pairs = vif_data[vif_data['VIF'] > 5]
print("\nPotential multicollinear features:")
print(multicollinear_pairs)

# Highlight the most multicollinear pair (if VIF > 5)
if not multicollinear_pairs.empty:
    highest_vif_pair = multicollinear_pairs.iloc[0]['feature']
    print(f"\nBased on VIF, consider removing '{highest_vif_pair}' to reduce multicollinearity.")
else:
    print("\nNo significant multicollinearity detected.")

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Separate features and target variable
X = df.drop(columns=['stroke', 'id'], errors='ignore')  # Drop 'stroke' (target) and 'id' (identifier)
y = df['stroke']

# Logistic Regression Model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X, y)  # Train on the entire dataset

# Create test datasets
test_data_1 = df[df['stroke'] == 1]

# Process test datasets
X_test_1 = test_data_1.drop(columns=['stroke', 'id'], errors='ignore')
y_test_1 = test_data_1['stroke']

X_test_1 = pd.get_dummies(X_test_1, drop_first=True).reindex(columns=X.columns, fill_value=0)

# Predictions for test dataset 1 (stroke == 1)
y_pred_1 = log_reg.predict(X_test_1)
y_pred_proba_1 = log_reg.predict_proba(X_test_1)[:, 1]

# Evaluate metrics for test dataset 1
accuracy_1 = accuracy_score(y_test_1, y_pred_1)
conf_matrix_1 = confusion_matrix(y_test_1, y_pred_1)

# Prepare data for plotting
print("Accuracy : ",accuracy_1)
datasets = ['Test Data (Stroke == 0)', 'Test Data (Stroke == 1)']
confusion_matrices = [conf_matrix_1]

# Plot confusion matrices
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'Confusion Matrix: {datasets[i]}', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()


#%%

from imblearn.over_sampling import SMOTE

# Encode categorical variables (if any)
df_bal = df.drop(columns=['id','stroke'],axis=1)
y=df['stroke']

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
df_bal, y = smote.fit_resample(X, y)

df_bal['stroke']=y

df_bal['stroke'].value_counts()


#%%

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split


# Separate features and target variable
X = df_bal.drop(columns=['stroke', 'id'], errors='ignore')
y = df_bal['stroke']

# Build the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=log_reg, n_features_to_select=6)  # Selecting top 5 features
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]
ranking = pd.DataFrame({
    'Feature': X.columns,
    'Rank': rfe.ranking_
}).sort_values(by='Rank')

# Output the selected features and their rankings
selected_features, ranking


#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the data
X = df_bal.drop('stroke',axis=1)
y = df_bal['stroke']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred) 

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

#%%
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

# Separate features and target variable
X = df_bal.drop(columns=['stroke', 'id'], errors='ignore')  # Drop 'stroke' (target) and 'id' (identifier)
y = df_bal['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build the Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Calculate training and testing scores
training_score = rf_clf.score(X_train, y_train)  # Accuracy on training data
testing_score = rf_clf.score(X_test, y_test)    # Accuracy on testing data

# Print training and testing scores
print(f"Training Accuracy: {training_score:.2f}")
print(f"Testing Accuracy: {testing_score:.2f}")

# Make predictions
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc:.2f}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

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

# Train a Random Forest with limited tree depth
rf_clf_limited = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=42)
rf_clf_limited.fit(X_train, y_train)

# Extract one tree from the limited model
limited_tree = rf_clf_limited.estimators_[0]

# Plot the smaller decision tree
plt.figure(figsize=(15, 8))
plot_tree(limited_tree, feature_names=X.columns, class_names=['No Stroke', 'Stroke'], filled=True, fontsize=10)
plt.title("Decision Tree Visualization (Limited Depth)", fontsize=16)
plt.show()



#%%


#%%


#%%


#%%



#%%

#%%