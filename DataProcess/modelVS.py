import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json

def svm_classification(norm_logs_path: str, labels_path: str):
    # Load the normalized logs
    with open(norm_logs_path, 'r') as f:
        norm_logs = json.load(f)

    # Load the labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    # Prepare the features and labels
    X = []
    y = []

    for subj, slices in norm_logs.items():
        for slice_id, frames in slices.items():
            # Extract features from the frames
            slice_features = []
            for frame in frames.values():
                # Convert position string to tuple of floats
                pos = eval(frame['pos'])
                slice_features.extend([
                    float(frame['time']),
                    *pos,
                    float(frame['speed']),
                    float(frame['acceleration']),
                    float(frame['rotationSpeed'])
                ])
            
            # Calculate average features for the slice
            avg_features = np.mean(np.array(slice_features).reshape(-1, 7), axis=0)
            X.append(avg_features)

            # Get the corresponding label
            y.append(labels[subj][slice_id])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = svm_classifier.predict(X_test_scaled)

    # Print the results
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return svm_classifier, scaler

# Usage
svm_model, scaler = svm_classification('norm_logs.json', 'labels.json')