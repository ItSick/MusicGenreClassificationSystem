"""

Required installations:
pip install streamlit, librosa, numpy, scikit-learn, pandas, joblib,os

"""

# ==================== IMPORTS ====================

import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import tempfile
from pathlib import Path

# ==================== CONFIGURATION ====================

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

MODEL_PATH = 'genre_classifier_model.pkl'

ENCODER_PATH = 'label_encoder.pkl'


# ==================== FEATURE EXTRACTION ====================

def extract_features(file_path, duration=30):
    try:
        audio, sr = librosa.load(file_path, duration=duration)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)


        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)

        tempo, beat_frame = librosa.beat.beat_track(y=audio, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        features = np.concatenate([
            np.atleast_1d(mfccs_mean).flatten(),
            np.atleast_1d(mfccs_std).flatten(),
            np.atleast_1d(chroma_mean).flatten(),
            np.atleast_1d(chroma_std).flatten(),
            np.atleast_1d(spectral_centroid_mean).flatten(),
            np.atleast_1d(spectral_centroid_std).flatten(),
            np.atleast_1d(spectral_rolloff_mean).flatten(),
            np.atleast_1d(spectral_rolloff_std).flatten(),
            np.atleast_1d(spectral_bandwidth_mean).flatten(),
            np.atleast_1d(spectral_bandwidth_std).flatten(),
            np.atleast_1d(zcr_mean).flatten(),
            np.atleast_1d(zcr_std).flatten(),
            np.atleast_1d(tempo).flatten(),
        ])

        if features.shape[0] != 59:
            st.warning(f"Expected 59 features but got {features.shape[0]} from {file_path}")

        return features

    except Exception as e:
        st.error(f"Error extracting features from {file_path}: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None


# ==================== MODEL TRAINING ====================

def train_model(data_folder):

    features_list = []
    labels_list = []

    st.write("### Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    genre_folders = [f for f in os.listdir(data_folder)
                     if os.path.isdir(os.path.join(data_folder, f))]

    total_files = 0
    processed_files = 0

    for genre in genre_folders:
        genre_path = os.path.join(data_folder, genre)
        files = [f for f in os.listdir(genre_path)
                 if f.endswith(('.mp3', '.wav', '.ogg'))]
        total_files += len(files)

    for genre_idx, genre in enumerate(genre_folders):
        genre_path = os.path.join(data_folder, genre)

        audio_files = [f for f in os.listdir(genre_path)
                       if f.endswith(('.mp3', '.wav', '.ogg'))]

        status_text.text(f"Processing {genre}... ({len(audio_files)} files)")

        for file_idx, audio_file in enumerate(audio_files):
            file_path = os.path.join(genre_path, audio_file)

            features = extract_features(file_path)

            if features is not None:
                features_list.append(features)
                labels_list.append(genre)

            processed_files += 1
            progress = processed_files / total_files
            progress_bar.progress(progress)

    X = np.array(features_list)
    y = np.array(labels_list)

    status_text.text(f"Extracted features from {len(X)} songs")

    if len(X) < 10:
        st.error(f"⚠️ Not enough data! Found only {len(X)} songs. You need at least 10 songs total (preferably 20+ per genre).")
        return None, None, None

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    unique, counts = np.unique(y, return_counts=True)
    genre_counts = dict(zip(unique, counts))

    st.write("#### Dataset Distribution:")
    for genre, count in genre_counts.items():
        st.write(f"- **{genre}**: {count} songs")

    min_samples = min(counts)
    if min_samples < 2:
        st.warning("⚠️ Some genres have less than 2 songs. Stratification disabled.")
        use_stratify = False
    else:
        use_stratify = True

    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    status_text.text("Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    status_text.text("Training complete!")
    progress_bar.progress(1.0)

    st.success(f"Model trained successfully with {accuracy*100:.2f}% accuracy!")

    st.write("#### Detailed Performance Metrics")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    st.info(f"Model saved to {MODEL_PATH}")

    return model, label_encoder, accuracy


# ==================== PREDICTION ====================

def predict_genre(file_path, model, label_encoder):

    features = extract_features(file_path)

    if features is None:
        return None, None

    features = features.reshape(1, -1)

    probabilities = model.predict_proba(features)[0]

    prediction = model.predict(features)[0]

    predicted_genre = label_encoder.inverse_transform([prediction])[0]

    prob_dict = {
        label_encoder.classes_[i]: probabilities[i]
        for i in range(len(probabilities))
    }

    return predicted_genre, prob_dict


# ==================== STREAMLIT UI ====================

def main():

    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 Music Genre Classification System")
    st.markdown("""
    This application uses Machine Learning to automatically identify the music genre 
    Upload an MP3 file and get instant predictions!
    """)

    st.sidebar.title("Model MGMT")

    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)

    if model_exists:
        st.sidebar.success("✅ Trained model found!")
    else:
        st.sidebar.warning("⚠️ No trained model found. Please train a model first")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Train New Model")

    data_folder = st.sidebar.text_input(
        "Dataset Folder Path",
        value="./dataset",
        help="Path to folder containing genre subfolders with audio files"
    )

    # if st.sidebar.button("Train Model"):
    #     if os.path.exists(data_folder):
    #         with st.spinner("Training in progress... This may take several minutes."):
    #             result = train_model(data_folder)
    #             if result[0] is None:
    #                 st.error("Training failed. Please check your dataset and try again.")
    #     else:
    #         st.sidebar.error(f"Folder '{data_folder}' not found!")

    st.markdown("---")
    st.header("🎼 Upload Music for Genre Prediction")

    uploaded_file = st.file_uploader(
        "Choose an audio file (MP3, WAV)",
        type=['mp3', 'wav'],
        help="Upload a music file to predict its genre"
    )

    if uploaded_file is not None:

        if not model_exists:
            st.error("Please train a model first using the sidebar!")
            return

        st.success(f"File uploaded: {uploaded_file.name}")

        st.audio(uploaded_file, format='audio/mp3')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)

            with st.spinner("Analyzing audio and predicting genre..."):
                predicted_genre, probabilities = predict_genre(
                    tmp_file_path,
                    model,
                    label_encoder
                )

            if predicted_genre is not None:
                st.markdown("---")
                st.subheader("🎯 Prediction Results")

                st.markdown(f"### Predicted Genre: **{predicted_genre.upper()}**")

                sorted_probs = sorted(
                    probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                st.write("#### Confidence Scores:")

                col1, col2 = st.columns(2)

                for idx, (genre, prob) in enumerate(sorted_probs):
                    col = col1 if idx % 2 == 0 else col2

                    with col:
                        st.write(f"**{genre.capitalize()}**")
                        st.progress(float(prob))
                        st.write(f"{prob*100:.2f}%")
                        st.write("")

            else:
                st.error("Failed to analyze the audio file. Please try another file.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>2025 ML איציק אדרי - עבודת הגשה קורס </div>",
        unsafe_allow_html=True
    )


# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()