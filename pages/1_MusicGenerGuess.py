import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

MODEL_PATH = 'genre_classifier_model.pkl'

ENCODER_PATH = 'label_encoder.pkl'

PYTORCH_MODEL_PATH = 'pytorch_genre_model.pth'

PYTORCH_ENCODER_PATH = 'pytorch_label_encoder.pkl'


class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GenreClassifierNN(nn.Module):
    def __init__(self, input_size=89, num_classes=10):
        super(GenreClassifierNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x


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
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        spectral_contrast_std = np.std(spectral_contrast, axis=1)
        
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        spectral_flatness_mean = np.mean(spectral_flatness)
        spectral_flatness_std = np.std(spectral_flatness)
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
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
            np.atleast_1d(spectral_contrast_mean).flatten(),
            np.atleast_1d(spectral_contrast_std).flatten(),
            np.atleast_1d(tonnetz_mean).flatten(),
            np.atleast_1d(tonnetz_std).flatten(),
            np.atleast_1d(rms_mean).flatten(),
            np.atleast_1d(rms_std).flatten(),
            np.atleast_1d(spectral_flatness_mean).flatten(),
            np.atleast_1d(spectral_flatness_std).flatten(),
            np.atleast_1d(tempo).flatten()
        ])
        
        expected_features = 89
        if features.shape[0] != expected_features:
            st.warning(f"Expected {expected_features} features but got {features.shape[0]} from {file_path}")
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features from {file_path}: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None


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
                if f.endswith(('.mp3', '.wav'))]
        total_files += len(files)
    
    for genre_idx, genre in enumerate(genre_folders):
        genre_path = os.path.join(data_folder, genre)
        
        audio_files = [f for f in os.listdir(genre_path) 
                      if f.endswith(('.mp3', '.wav'))]
        
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


def train_pytorch_model(data_folder, epochs=50, batch_size=32, learning_rate=0.001):
    features_list = []
    labels_list = []
    
    st.write("### PyTorch Neural Network Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    genre_folders = [f for f in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, f))]
    
    total_files = 0
    processed_files = 0
    
    for genre in genre_folders:
        genre_path = os.path.join(data_folder, genre)
        files = [f for f in os.listdir(genre_path) 
                if f.endswith(('.mp3', '.wav'))]
        total_files += len(files)
    
    status_text.text("Extracting features from audio files...")
    
    for genre_idx, genre in enumerate(genre_folders):
        genre_path = os.path.join(data_folder, genre)
        
        audio_files = [f for f in os.listdir(genre_path) 
                      if f.endswith(('.mp3', '.wav'))]
        
        status_text.text(f"Processing {genre}... ({len(audio_files)} files)")
        
        for file_idx, audio_file in enumerate(audio_files):
            file_path = os.path.join(genre_path, audio_file)
            
            features = extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(genre)
            
            processed_files += 1
            progress = processed_files / total_files * 0.3
            progress_bar.progress(progress)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    status_text.text(f"Extracted features from {len(X)} songs")
    
    if len(X) < 10:
        st.error(f"⚠️ Not enough data! Found only {len(X)} songs. You need at least 10 songs total.")
        return None, None, None
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    unique, counts = np.unique(y, return_counts=True)
    genre_counts = dict(zip(unique, counts))
    
    st.write("#### Dataset Distribution:")
    for genre, count in genre_counts.items():
        st.write(f"- **{genre}**: {count} songs")
    
    min_samples = min(counts)
    use_stratify = min_samples >= 2
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    status_text.text("Preparing neural network...")
    
    train_dataset = MusicDataset(X_train, y_train)
    test_dataset = MusicDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    model = GenreClassifierNN(input_size=input_size, num_classes=num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if device.type == 'cuda':
        st.info("🚀 Training on GPU for faster performance!")
    else:
        st.info("💻 Training on CPU")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    status_text.text("Training neural network...")
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        scheduler.step(avg_loss)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, PYTORCH_MODEL_PATH)
        
        progress = 0.3 + (0.7 * (epoch + 1) / epochs)
        progress_bar.progress(progress)
        
        if (epoch + 1) % 5 == 0:
            status_text.text(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                f"Accuracy: {accuracy:.2f}% - Best: {best_accuracy:.2f}%"
            )
    
    status_text.text("Training complete!")
    progress_bar.progress(1.0)
    
    checkpoint = torch.load(PYTORCH_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    final_accuracy = accuracy_score(all_labels, all_preds)
    
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    st.success(f"Neural Network trained successfully with {final_accuracy*100:.2f}% accuracy!")
    st.info(f"Best accuracy during training: {best_accuracy:.2f}%")
    
    st.write("#### Detailed Performance Metrics")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    st.write("#### Training History")
    history_df = pd.DataFrame({
        'Epoch': list(range(1, epochs + 1)),
        'Loss': train_losses,
        'Accuracy': test_accuracies
    })
    st.line_chart(history_df.set_index('Epoch'))
    
    joblib.dump(label_encoder, PYTORCH_ENCODER_PATH)
    st.info(f"Model saved to {PYTORCH_MODEL_PATH}")
    
    return model, label_encoder, final_accuracy


def predict_genre(file_path, model, label_encoder, model_type='random_forest'):
    features = extract_features(file_path)
    
    if features is None:
        return None, None
    
    if model_type == 'neural_network':
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        features_tensor = features_tensor.to(device)
        
        model.eval()
        
        with torch.no_grad():
            outputs = model(features_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            probabilities = probabilities.cpu().numpy()
            
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()
        
        predicted_genre = label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {
            label_encoder.classes_[i]: float(probabilities[i]) 
            for i in range(len(probabilities))
        }
        
    else:
        features = features.reshape(1, -1)
        
        probabilities = model.predict_proba(features)[0]
        
        prediction = model.predict(features)[0]
        
        predicted_genre = label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {
            label_encoder.classes_[i]: probabilities[i] 
            for i in range(len(probabilities))
        }
    
    return predicted_genre, prob_dict


def main():
    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Music Genre Classification System")
    st.markdown("""
    This application uses Machine Learning to automatically identify the genre of music.
    Upload an MP3 file and get instant predictions!
    """)
    
    st.sidebar.title("Model Management")
    
    st.sidebar.subheader("Select Model Type")
    model_type = st.sidebar.radio(
        "Choose the model to use:",
        options=['Random Forest', 'Neural Network'],
        help="Random Forest: Fast training, good baseline. Neural Network: Potentially higher accuracy, requires more data."
    )
    
    if model_type == 'Random Forest':
        model_path = MODEL_PATH
        encoder_path = ENCODER_PATH
        selected_model_type = 'random_forest'
    else:
        model_path = PYTORCH_MODEL_PATH
        encoder_path = PYTORCH_ENCODER_PATH
        selected_model_type = 'neural_network'
    
    model_exists = os.path.exists(model_path) and os.path.exists(encoder_path)
    
    if model_exists:
        st.sidebar.success(f"✅ {model_type} model found!")
    else:
        st.sidebar.warning(f"⚠️ No {model_type} model found. Please train a model first.")
    
    # st.sidebar.markdown("---")
    # st.sidebar.subheader("Train New Model")
    
    # if model_type == 'Neural Network':
    #     with st.sidebar.expander("⚙️ Training Parameters"):
    #         epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50, step=10,
    #                                 help="Number of training iterations")
    #         batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8,
    #                                     help="Number of samples per batch")
    #         learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, 
    #                                        value=0.001, step=0.0001, format="%.4f",
    #                                        help="How fast the model learns")
    
    # data_folder = st.sidebar.text_input(
    #     "Dataset Folder Path",
    #     value="./dataset",
    #     help="Path to folder containing genre subfolders with audio files"
    # )
    
    # if st.sidebar.button(f"Train {model_type} Model"):
    #     if os.path.exists(data_folder):
    #         with st.spinner(f"Training {model_type} model..."):
    #             if model_type == 'Random Forest':
    #                 result = train_model(data_folder)
    #             else:
    #                 result = train_pytorch_model(
    #                     data_folder, 
    #                     epochs=epochs, 
    #                     batch_size=batch_size, 
    #                     learning_rate=learning_rate
    #                 )
                
    #             if result[0] is None:
    #                 st.error("Training failed. Please check your dataset and try again.")
    #     else:
    #         st.sidebar.error(f"Folder '{data_folder}' not found!")
    
    st.markdown("---")
    st.header("🎼 Upload Music for Genre Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (MP3)",
        type=['mp3', 'wav'],
        help="Upload a music file to predict its genre"
    )
    
    if uploaded_file is not None:
        
        if not model_exists:
            st.error("train a model first!")
            return
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        st.audio(uploaded_file, format='audio/mp3')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if model_type == 'Random Forest':
                model = joblib.load(MODEL_PATH)
                label_encoder = joblib.load(ENCODER_PATH)
            else:
                label_encoder = joblib.load(PYTORCH_ENCODER_PATH)
                
                num_classes = len(label_encoder.classes_)
                
                model = GenreClassifierNN(input_size=89, num_classes=num_classes)
                
                checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            
            with st.spinner("Analyzing audio and predicting genre..."):
                predicted_genre, probabilities = predict_genre(
                    tmp_file_path, 
                    model, 
                    label_encoder,
                    model_type=selected_model_type
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
    st.markdown(
        "<div style='text-align: center; min-height: 250px'></div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>2025 ML איציק אדרי - עבודת הגשה קורס </div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()