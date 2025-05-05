import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------------------- Load environment variables -------------------
load_dotenv()
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# ------------------- Spotify Setup -------------------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# ------------------- Load Models and Utilities -------------------
model = load_model("best_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------- Feature Extraction -------------------
def extract_features(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    features = np.mean(log_mel_spec, axis=1).tolist() + np.std(log_mel_spec, axis=1).tolist()
    return np.array(features).reshape(1, -1)

# ------------------- Recommend Music -------------------
def recommend_music(emotion):
    mood_map = {
        'HAP': 'pop',
        'SAD': 'classical',
        'ANG': 'rock',
        'NEU': 'chill',
        'FEA': 'metal',
        'DIS': 'emo'
    }
    genre = mood_map.get(emotion, 'pop')
    results = sp.search(q=f"genre:{genre}", limit=5)

    # Return list of embed URLs
    embed_links = []
    for track in results['tracks']['items']:
        track_id = track['id']
        embed_url = f"https://open.spotify.com/embed/track/{track_id}"
        embed_links.append(embed_url)

    return embed_links

# ------------------- Streamlit UI -------------------
st.title("🎧 Moodify")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp.wav", format='audio/wav')
    
    feats = extract_features("temp.wav")
    feats_scaled = scaler.transform(feats).reshape(1, feats.shape[1], 1)
    
    pred = model.predict(feats_scaled)
    pred_label = np.argmax(pred)
    emotion = label_encoder.inverse_transform([pred_label])[0]
    
    st.subheader("🧠 Predicted Emotion:")
    st.success(emotion)
    
    st.subheader("🎵 Recommended Music:")
    for url in recommend_music(emotion):
        st.markdown(f'<iframe src="{url}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
