import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy


# Load Dataset (KDD Cup 99 - Network Intrusion Detection)
data = fetch_kddcup99(subset="http", as_frame=True).frame
data = pd.DataFrame(data)
# Columns to check
cols_to_check = np.array(data.columns)  # ['duration', 'src_bytes', 'dst_bytes', 'labels']

for col in cols_to_check:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Introduce Anomaly Labels (5% labeled anomalies)
# np.random.seed(42)
data['Anomaly'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])
# Split Data into Labeled (5%) and Unlabeled (95%)
labeled_data = data.sample(frac=0.05, random_state=42)  # I think this a subpopulation
unlabeled_data = data.drop(labeled_data.index)
X_labeled, y_labeled = labeled_data.drop(columns=['Anomaly']), labeled_data['Anomaly']
X_unlabeled = unlabeled_data.drop(columns=['Anomaly'])

# Initial Unsupervised Anomaly Detection (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X_unlabeled)
gmm_scores = -gmm.score_samples(X_unlabeled)  # Higher scores indicate anomalies

# Convert GMM Scores into Initial Anomaly Predictions
threshold = np.percentile(gmm_scores, 95)  # Top 5% as anomalies
y_unlabeled_pred = (gmm_scores >= threshold).astype(int)


# Active Learning: Selecting Samples for Labeling (Entropy-Based)
def select_samples_for_labeling(X_unlabeled, y_pred, num_samples=50):
    uncertainty = entropy(np.vstack([y_pred, 1 - y_pred]), axis=0)
    uncertain_samples = np.argsort(-uncertainty)[:num_samples]
    return X_unlabeled.iloc[uncertain_samples]


# Simulating Active Learning Iterations
for i in range(3):  # 3 Active Learning Cycles
    X_query = select_samples_for_labeling(X_unlabeled, y_unlabeled_pred, num_samples=50)

    # Simulate Expert Labeling (Using True Labels)
    y_query = data.loc[X_query.index, 'Anomaly']

    # Add Labeled Data to Training Set
    X_labeled = pd.concat([X_labeled, X_query])
    y_labeled = pd.concat([y_labeled, y_query])

    # Train a New Isolation Forest on the Growing Labeled Set
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_labeled, y_labeled)

    # Predict on Unlabeled Data
    y_unlabeled_pred = model.predict(X_unlabeled)
    y_unlabeled_pred = np.where(y_unlabeled_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1

# Final Evaluation on Test Set
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
final_model = IsolationForest(contamination=0.05, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
