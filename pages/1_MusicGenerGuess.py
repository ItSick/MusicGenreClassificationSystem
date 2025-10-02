"""
Music Genre Classification System
A complete Streamlit application for training and predicting music genres from MP3 files.

Required installations:
pip install streamlit librosa numpy scikit-learn pandas joblib

To run:
streamlit run app.py
"""

# ==================== IMPORTS ====================

import streamlit as st  # Web application framework for creating the UI
import librosa  # Audio processing library for feature extraction
import numpy as np  # Numerical operations and array handling
import pandas as pd  # Data manipulation and analysis
from sklearn.ensemble import RandomForestClassifier  # Machine learning classifier
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.preprocessing import LabelEncoder  # Convert genre labels to numbers
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation
import joblib  # Save and load trained models
import os  # Operating system operations for file handling
import tempfile  # Create temporary files for uploaded audio
from pathlib import Path  # Handle file paths in a cross-platform way

# ==================== CONFIGURATION ====================

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Model file path - where we save/load the trained model
MODEL_PATH = 'genre_classifier_model.pkl'

# Label encoder path - saves the mapping between genre names and numbers
ENCODER_PATH = 'label_encoder.pkl'


# ==================== FEATURE EXTRACTION ====================

def extract_features(file_path, duration=30):
    """
    Extract audio features from a music file using librosa.
    
    This function analyzes the audio and extracts various characteristics
    that help identify the genre, such as rhythm, melody, and timbre.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file (MP3, WAV, etc.)
    duration : int
        How many seconds of audio to analyze (default: 30 seconds)
        
    Returns:
    --------
    numpy.ndarray
        A 1D array containing all extracted features concatenated together
        Returns None if there's an error processing the file
    """
    try:
        # Load the audio file
        # y: audio time series (the actual sound wave as numbers)
        # sr: sampling rate (how many samples per second, usually 22050 Hz)
        # duration: only load the first 30 seconds to keep processing fast
        audio, sr = librosa.load(file_path, duration=duration)
        
        # ===== MFCC Features (Mel-Frequency Cepstral Coefficients) =====
        # MFCCs represent the shape of the vocal tract and are excellent for 
        # distinguishing different timbres (e.g., guitar vs piano)
        # n_mfcc=13: Extract 13 MFCC coefficients (standard in music analysis)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Calculate statistics across time for each MFCC coefficient
        # This gives us 13 mean values and 13 standard deviation values
        mfccs_mean = np.mean(mfccs, axis=1)  # Average value over time
        mfccs_std = np.std(mfccs, axis=1)    # Variation over time
        
        # ===== Chroma Features =====
        # Chroma features represent the 12 different pitch classes (C, C#, D, etc.)
        # Useful for identifying harmonic and melodic characteristics
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # 12 values, one per pitch class
        chroma_std = np.std(chroma, axis=1)
        
        # ===== Spectral Features =====
        # These describe the frequency content and texture of the sound
        
        # Spectral Centroid: The "center of mass" of the spectrum
        # Higher values = brighter sounds (e.g., cymbals)
        # Lower values = darker sounds (e.g., bass guitar)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        
        # Spectral Rolloff: Frequency below which 85% of energy is contained
        # Helps distinguish between harmonic and noisy sounds
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        
        # Spectral Bandwidth: Range of frequencies present
        # Wide bandwidth = many frequencies (e.g., rock music)
        # Narrow bandwidth = few frequencies (e.g., pure tones)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # ===== Zero Crossing Rate =====
        # How often the signal changes from positive to negative
        # High values indicate percussive or noisy sounds
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)
        
        # ===== Rhythm Features =====
        # Tempo: Speed of the music in beats per minute (BPM)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Ensure tempo is a scalar value (sometimes it returns an array)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        
        # Concatenate all features into a single feature vector
        # This creates one long array with all the information about the song
        # We use np.atleast_1d to ensure all scalar values become 1D arrays
        # and flatten() to ensure no 2D arrays slip through
        features = np.concatenate([
            np.atleast_1d(mfccs_mean).flatten(),                    # 13 features
            np.atleast_1d(mfccs_std).flatten(),                     # 13 features
            np.atleast_1d(chroma_mean).flatten(),                   # 12 features
            np.atleast_1d(chroma_std).flatten(),                    # 12 features
            np.atleast_1d(spectral_centroid_mean).flatten(),        # 1 feature
            np.atleast_1d(spectral_centroid_std).flatten(),         # 1 feature
            np.atleast_1d(spectral_rolloff_mean).flatten(),         # 1 feature
            np.atleast_1d(spectral_rolloff_std).flatten(),          # 1 feature
            np.atleast_1d(spectral_bandwidth_mean).flatten(),       # 1 feature
            np.atleast_1d(spectral_bandwidth_std).flatten(),        # 1 feature
            np.atleast_1d(zcr_mean).flatten(),                      # 1 feature
            np.atleast_1d(zcr_std).flatten(),                       # 1 feature
            np.atleast_1d(tempo).flatten()                          # 1 feature
        ])
        
        # Validate feature vector shape
        # Should have exactly 59 features (13+13+12+12+1+1+1+1+1+1+1+1+1 = 59)
        if features.shape[0] != 59:
            st.warning(f"Expected 59 features but got {features.shape[0]} from {file_path}")
        
        # Total: 59 features describing the audio
        return features
        
    except Exception as e:
        # If anything goes wrong (corrupted file, unsupported format, etc.)
        # print the error and return None
        st.error(f"Error extracting features from {file_path}: {str(e)}")
        # Print more detailed error information for debugging
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None


# ==================== MODEL TRAINING ====================

def train_model(data_folder):
    """
    Train a Random Forest classifier on a dataset of music files.
    
    This function expects a folder structure like:
    data_folder/
        blues/
            song1.mp3
            song2.mp3
        jazz/
            song1.mp3
            song2.mp3
        ... (one folder per genre)
    
    Parameters:
    -----------
    data_folder : str
        Path to the folder containing genre subfolders with audio files
        
    Returns:
    --------
    tuple
        (model, label_encoder, accuracy) - The trained model, encoder, and accuracy score
    """
    
    # Lists to store our training data
    features_list = []  # Will store the extracted features for each song
    labels_list = []    # Will store the genre label for each song
    
    # Progress tracking
    st.write("### Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get all genre folders
    genre_folders = [f for f in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, f))]
    
    total_files = 0
    processed_files = 0
    
    # First, count total files for progress tracking
    for genre in genre_folders:
        genre_path = os.path.join(data_folder, genre)
        files = [f for f in os.listdir(genre_path) 
                if f.endswith(('.mp3', '.wav', '.ogg'))]
        total_files += len(files)
    
    # Process each genre folder
    for genre_idx, genre in enumerate(genre_folders):
        genre_path = os.path.join(data_folder, genre)
        
        # Get all audio files in this genre folder
        audio_files = [f for f in os.listdir(genre_path) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        status_text.text(f"Processing {genre}... ({len(audio_files)} files)")
        
        # Process each audio file in the genre
        for file_idx, audio_file in enumerate(audio_files):
            file_path = os.path.join(genre_path, audio_file)
            
            # Extract features from the audio file
            features = extract_features(file_path)
            
            # Only add if feature extraction was successful
            if features is not None:
                features_list.append(features)  # Add features to our dataset
                labels_list.append(genre)        # Add corresponding genre label
            
            # Update progress bar
            processed_files += 1
            progress = processed_files / total_files
            progress_bar.progress(progress)
    
    # Convert lists to numpy arrays for machine learning
    X = np.array(features_list)  # Features matrix (each row is a song)
    y = np.array(labels_list)    # Labels vector (genre for each song)
    
    status_text.text(f"Extracted features from {len(X)} songs")
    
    # ===== Validate Dataset Size =====
    # Check if we have enough data to train
    if len(X) < 10:
        st.error(f"⚠️ Not enough data! Found only {len(X)} songs. You need at least 10 songs total (preferably 20+ per genre).")
        return None, None, None
    
    # ===== Encode Labels =====
    # Convert genre names (strings) to numbers for the ML algorithm
    # e.g., 'rock' -> 0, 'jazz' -> 1, 'blues' -> 2, etc.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Check samples per genre
    unique, counts = np.unique(y, return_counts=True)
    genre_counts = dict(zip(unique, counts))
    
    st.write("#### Dataset Distribution:")
    for genre, count in genre_counts.items():
        st.write(f"- **{genre}**: {count} songs")
    
    # Check if any genre has less than 2 samples (needed for stratification)
    min_samples = min(counts)
    if min_samples < 2:
        st.warning("⚠️ Some genres have less than 2 songs. Stratification disabled.")
        use_stratify = False
    else:
        use_stratify = True
    
    # ===== Split Data =====
    # Divide data into training set (80%) and testing set (20%)
    # This helps us evaluate how well the model works on unseen data
    # random_state=42: Makes the split reproducible (same split every time)
    # stratify: Ensures each split has the same proportion of genres (only if enough samples)
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    status_text.text("Training model...")
    
    # ===== Train Random Forest Classifier =====
    # Random Forest creates many decision trees and combines their predictions
    # n_estimators=100: Create 100 decision trees
    # random_state=42: Reproducible results
    # n_jobs=-1: Use all CPU cores for faster training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # ===== Evaluate Model =====
    # Test the model on data it hasn't seen before
    y_pred = model.predict(X_test)  # Make predictions on test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    
    # Generate detailed classification report
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    status_text.text("Training complete!")
    progress_bar.progress(1.0)
    
    # Display results
    st.success(f"Model trained successfully with {accuracy*100:.2f}% accuracy!")
    
    # Show detailed metrics in a table
    st.write("#### Detailed Performance Metrics")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # ===== Save Model and Encoder =====
    # Save to disk so we can use them later without retraining
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    st.info(f"Model saved to {MODEL_PATH}")
    
    return model, label_encoder, accuracy


# ==================== PREDICTION ====================

def predict_genre(file_path, model, label_encoder):
    """
    Predict the genre of a music file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file to classify
    model : sklearn model
        The trained classifier model
    label_encoder : LabelEncoder
        The encoder to convert predictions back to genre names
        
    Returns:
    --------
    tuple
        (predicted_genre, probabilities) - Genre name and confidence for all genres
    """
    
    # Extract features from the uploaded file
    features = extract_features(file_path)
    
    if features is None:
        return None, None
    
    # Reshape features to 2D array (sklearn expects 2D input)
    # Shape changes from (59,) to (1, 59) - 1 sample with 59 features
    features = features.reshape(1, -1)
    
    # Get prediction probabilities for each genre
    # This tells us how confident the model is about each possible genre
    probabilities = model.predict_proba(features)[0]
    
    # Get the predicted class (the genre with highest probability)
    prediction = model.predict(features)[0]
    
    # Convert the numeric prediction back to genre name
    predicted_genre = label_encoder.inverse_transform([prediction])[0]
    
    # Create a dictionary mapping genres to their probabilities
    prob_dict = {
        label_encoder.classes_[i]: probabilities[i] 
        for i in range(len(probabilities))
    }
    
    return predicted_genre, prob_dict


# ==================== STREAMLIT UI ====================

def main():
    """
    Main function that creates the Streamlit web interface.
    """
    
    # ===== Page Configuration =====
    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="wide"
    )
    
    # ===== Title and Description =====
    st.title("🎵 Music Genre Classification System")
    st.markdown("""
    This application uses Machine Learning to automatically identify the genre of music.
    Upload an MP3 file and get instant predictions!
    """)
    
    # ===== Sidebar for Model Management =====
    st.sidebar.title("Model Management")
    
    # Check if a trained model exists
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)
    
    if model_exists:
        st.sidebar.success("✅ Trained model found!")
    else:
        st.sidebar.warning("⚠️ No trained model found. Please train a model first.")
    
    # ===== Training Section =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("Train New Model")
    # st.sidebar.markdown("""
    # To train a model, organize your music files like this:
    # ```
    # dataset/
    #     blues/
    #         song1.mp3
    #         song2.mp3
    #     jazz/
    #         song1.mp3
    #     ...
    # ```
    # """)
    
    # Input field for dataset folder path
    data_folder = st.sidebar.text_input(
        "Dataset Folder Path",
        value="./dataset",
        help="Path to folder containing genre subfolders with audio files"
    )
    
    # Training button
    # if st.sidebar.button("Train Model"):
    #     if os.path.exists(data_folder):
    #         with st.spinner("Training in progress... This may take several minutes."):
    #             result = train_model(data_folder)
    #             if result[0] is None:
    #                 st.error("Training failed. Please check your dataset and try again.")
    #     else:
    #         st.sidebar.error(f"Folder '{data_folder}' not found!")
    
    # ===== Main Prediction Section =====
    st.markdown("---")
    st.header("🎼 Upload Music for Genre Prediction")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose an audio file (MP3, WAV)",
        type=['mp3', 'wav'],
        help="Upload a music file to predict its genre"
    )
    
    # If user uploaded a file
    if uploaded_file is not None:
        
        # Check if model exists
        if not model_exists:
            st.error("Please train a model first using the sidebar!")
            return
        
        # Display file information
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Create audio player so user can listen
        st.audio(uploaded_file, format='audio/mp3')
        
        # Create a temporary file to save the uploaded audio
        # This is necessary because librosa needs a file path, not a file buffer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            # Write uploaded file content to temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load the trained model and encoder from disk
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            
            # Show loading spinner while processing
            with st.spinner("Analyzing audio and predicting genre..."):
                # Make prediction
                predicted_genre, probabilities = predict_genre(
                    tmp_file_path, 
                    model, 
                    label_encoder
                )
            
            # Display results if prediction was successful
            if predicted_genre is not None:
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Show predicted genre in large text
                st.markdown(f"### Predicted Genre: **{predicted_genre.upper()}**")
                
                # Sort probabilities from highest to lowest
                sorted_probs = sorted(
                    probabilities.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Display confidence for each genre
                st.write("#### Confidence Scores:")
                
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                
                # Display each genre with its confidence score and progress bar
                for idx, (genre, prob) in enumerate(sorted_probs):
                    # Alternate between columns for nice layout
                    col = col1 if idx % 2 == 0 else col2
                    
                    with col:
                        # Show genre name and percentage
                        st.write(f"**{genre.capitalize()}**")
                        # Progress bar showing confidence level
                        st.progress(float(prob))
                        # Exact percentage
                        st.write(f"{prob*100:.2f}%")
                        st.write("")  # Add spacing
                
            else:
                st.error("Failed to analyze the audio file. Please try another file.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            
        finally:
            # Clean up: delete the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    # ===== Information Section =====
    # st.markdown("---")
    # st.subheader("ℹ️ How It Works")
    
    # with st.expander("Click to learn more"):
    #     st.markdown("""
    #     **Feature Extraction:** 
    #     - The system analyzes various audio characteristics including:
    #       - MFCCs (Mel-Frequency Cepstral Coefficients) - captures timbre
    #       - Chroma features - captures harmony and melody
    #       - Spectral features - captures frequency content
    #       - Rhythm features - captures tempo and beat
        
    #     **Machine Learning Model:**
    #     - Uses Random Forest classifier with 100 decision trees
    #     - Trained on multiple examples from each genre
    #     - Makes predictions based on learned patterns
        
    #     **Prediction:**
    #     - Extracts 59 different features from your audio file
    #     - Compares these features to learned patterns
    #     - Provides confidence scores for each possible genre
    #     """)
    
    # ===== Footer =====
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>2025 ML איציק אדרי - עבודת הגשה קורס </div>",
        unsafe_allow_html=True
    )


# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    # This runs when you execute: streamlit run app.py
    main()