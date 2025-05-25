import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

# load the cleaned dataset
df = pd.read_csv('data/data_cleaned.csv')

# convert valence values into mood categories: low, medium, high
def valence_to_mood(val):
    if val <= 0.33:
        return 'low'
    elif val <= 0.66:
        return 'medium'
    else:
        return 'high'

# apply mood conversion
df['mood'] = df['valence'].apply(valence_to_mood)

# define feature columns to use for prediction
features = ['danceability', 'energy', 'acousticness', 'speechiness', 'instrumentalness',
            'loudness', 'tempo', 'duration_ms', 'popularity']

# separate features and target
X = df[features]
y = df['mood']

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define different classification models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'MLP Neural Net': MLPClassifier(max_iter=1000, random_state=42)
}

# dictionary to store accuracy scores
accuracies = {}

# train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # train the model
    y_pred = model.predict(X_test_scaled)  # make predictions
    acc = accuracy_score(y_test, y_pred)  # calculate accuracy
    accuracies[name] = acc

    # save the trained model
    model_filename = f'ml_pickles/{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, model_filename)

# save the scaler for future use
joblib.dump(scaler, 'ml_pickles/scaler.pkl')

# save accuracy results and test data for later analysis
joblib.dump(accuracies, 'ml_pickles/accuracies.pkl')
joblib.dump((X_test_scaled, y_test), 'ml_pickles/test_data.pkl')