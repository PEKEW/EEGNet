import numpy as np
import ast
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
import random


class SVMClassifier:
    def __init__(self):
        self.data = None
        self.labels = None
        self.model = None
        self.scaler = None

    def loadData(self, data_path:str, label_path:str):
        with open(data_path, 'r') as f:
            norm_logs = json.load(f)
        with open(label_path, 'r') as f:
            labels = json.load(f)
        x, y = [], []
        for subj, slices in norm_logs.items():
            for slice_id, frames in slices.items():
                slice_features = []
                for frame in frames.values():
                    pos = ast.literal_eval(frame['pos'])
                    slice_features.extend(
                        float(pos[0]), float(pos[1]), float(pos[2]),
                        float(frame['time']),
                        float(frame['speed']),
                        float(frame['acceleration']),
                        float(frame['rotationSpeed'])
                    )
                avg_features = np.mean(np.array(slice_features).reshape(-1, ), axis=0)
                x.append(avg_features)
                y.append(labels[subj][slice_id])
        self.data = np.array(x)
        self.labels = np.array(y)
    def train_fit(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        svm_classifier = SVC(kernel='rbf', random_state=random_state)
        svm_classifier.fit(X_train_scaled, y_train)
        self.scaler = scaler
        self.model = svm_classifier
        y_pred = svm_classifier.predict(X_test_scaled)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))



if __name__ == '__main__':
    data_path = './DataProcess/norm_logs.json'
    labels_path = './DataProcess/labels.json'
    random_seed = random.randint(0, 100)
    svm = SVMClassifier()
    svm.loadData(data_path, labels_path)
    svm.train_fit(random_state=random_seed)