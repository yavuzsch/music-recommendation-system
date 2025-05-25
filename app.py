import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

def valence_to_mood(val):
    # convert valence value to mood category
    if val <= 0.33:
        return 'low'
    elif val <= 0.66:
        return 'medium'
    else:
        return 'high'

# load the cleaned dataset
df = pd.read_csv('data/data_cleaned.csv')

# add mood column if missing, convert to lowercase if exists
if 'mood' not in df.columns:
    df['mood'] = df['valence'].apply(valence_to_mood)
else:
    df['mood'] = df['mood'].str.lower()

features = ['danceability', 'energy', 'acousticness', 'speechiness', 'instrumentalness',
            'loudness', 'tempo', 'duration_ms', 'popularity']

music_names = df['name'].tolist()  # list of all music names

model_files = {
    'random forest': 'ml_pickles/random_forest_model.pkl',
    'knn': 'ml_pickles/knn_model.pkl',
    'logistic regression': 'ml_pickles/logistic_regression_model.pkl',
    'mlp neural net': 'ml_pickles/mlp_neural_net_model.pkl'
}

scaler = joblib.load('ml_pickles/scaler.pkl')  # load feature scaler

root = tk.Tk()
root.title('music mood prediction and recommendation system')

tk.Label(root, text='select model:').grid(row=0, column=0, padx=5, pady=5, sticky='w')
model_combobox = ttk.Combobox(root, values=list(model_files.keys()), state='readonly', width=25)
model_combobox.grid(row=0, column=1, padx=5, pady=5)
model_combobox.current(0)

comboboxes = []
for i in range(4):
    tk.Label(root, text=f'music {i+1}').grid(row=i+1, column=0, padx=5, pady=5, sticky='w')
    cb = ttk.Combobox(root, values=music_names, width=50, state='readonly')
    cb.grid(row=i+1, column=1, padx=5, pady=5)
    comboboxes.append(cb)

def predict_and_recommend():
    try:
        selected_model_name = model_combobox.get()
        model_path = model_files.get(selected_model_name)

        if not model_path:
            messagebox.showerror('error', 'model file not found!')
            return

        model = joblib.load(model_path)  # load selected model

        selected_musics = [cb.get() for cb in comboboxes]
        if len(set(selected_musics)) < 4 or '' in selected_musics:
            messagebox.showerror('error', 'please select 4 different musics!')
            return

        # calculate average features of selected musics
        selected_features = df[df['name'].isin(selected_musics)][features]
        avg_features = selected_features.mean().values.reshape(1, -1)

        avg_scaled = scaler.transform(avg_features)  # scale features

        predicted_mood = model.predict(avg_scaled)[0].lower()  # predict mood

        # get prediction confidence if available
        if hasattr(model, 'predict_proba'):
            mood_proba = model.predict_proba(avg_scaled)[0]
            mood_index = list(map(str.lower, model.classes_)).index(predicted_mood)
            mood_confidence = mood_proba[mood_index]
            if mood_confidence == 0:
                mood_confidence = 1e-6  # avoid division by zero
        else:
            mood_confidence = 1.0

        # filter candidate musics with same mood excluding selected ones
        candidates = df[(df['mood'] == predicted_mood) & (~df['name'].isin(selected_musics))]
        if candidates.empty:
            messagebox.showinfo('result',
                                f'predicted mood: {predicted_mood}\n'
                                f'no other recommendations for this mood.')
            return

        candidate_features = candidates[features].values
        candidate_scaled = scaler.transform(candidate_features)

        # compute euclidean distances from average features
        distances = np.array([euclidean(avg_scaled[0], cs) for cs in candidate_scaled])
        weighted_distances = distances / mood_confidence  # weight distances by confidence

        min_idx = np.argmin(weighted_distances)  # index of closest music
        recommended_music = candidates.iloc[min_idx]['name']
        rec_valence = candidates.iloc[min_idx]['valence']

        messagebox.showinfo('result',
                            f'predicted mood: {predicted_mood} (confidence: {mood_confidence:.2f})\n'
                            f'recommended music: {recommended_music}\n'
                            f'recommended music valence: {rec_valence:.2f}')

    except Exception as e:
        print(f"error: {e}")
        messagebox.showerror('error', f'an error occurred: {e}')

btn = tk.Button(root, text='predict mood and recommend music', command=predict_and_recommend)
btn.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()