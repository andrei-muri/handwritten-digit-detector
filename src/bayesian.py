# bayesian.py
import numpy as np
import pickle
import os
from typing import List, Tuple

class Bayesian:
    def __init__(self, nrclasses: int = 10):
        self.nrclasses = nrclasses
        self.likelihood = None
        self.prior = None
        self.trained = False
        self.metrics_cache_dir = "../cache/bayesian_metrics"
        os.makedirs(self.metrics_cache_dir, exist_ok=True)

    def fit(self, training_data: List[Tuple[np.ndarray, int]]):
        print("Training Naive Bayes...")
        n = len(training_data)
        counts = np.zeros(self.nrclasses)
        pixel_counts = np.zeros((self.nrclasses, 784))

        for x, y in training_data:
            label = y if isinstance(y, (int, np.integer)) else int(np.argmax(y))
            counts[label] += 1
            x_bin = (x.squeeze() >= 0.5).astype(int)
            pixel_counts[label] += x_bin

        self.likelihood = (pixel_counts + 1) / (counts[:, np.newaxis] + 2)
        self.prior = counts / n
        self.trained = True
        print("Naive Bayes trained!")

    def predict(self, sample: np.ndarray) -> int:
        if not self.trained:
            return 0
        x = (sample.squeeze() >= 0.5).astype(int)
        ones = np.where(x == 1)[0]
        zeros = np.where(x == 0)[0]

        if len(ones) == 0: 
            return int(np.argmax(self.prior))

        log_probs = np.log(self.prior + 1e-15)
        log_probs += np.log(self.likelihood[:, ones]).sum(axis=1)
        log_probs += np.log(1.0 - self.likelihood[:, zeros]).sum(axis=1)

        return int(np.argmax(log_probs))

    def _metrics_cache_path(self, n_samples: int):
        return os.path.join(self.metrics_cache_dir, f"bayes_test{n_samples}.pkl")

    def print_metrics_report(self, test_data: List[Tuple[np.ndarray, int]]):
        from metrics import Metrics

        cache_path = self._metrics_cache_path(len(test_data))
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"\nLoaded cached Bayesian metrics (n={len(test_data)})")
            Metrics.print_metrics_report(data['predictions'], data['actuals'])
            return

        print(f"\nEvaluating Bayesian on {len(test_data)} test samples...")
        predictions = []
        actuals = []
        for x, y in test_data:
            label = y
            pred = self.predict(x)
            predictions.append(pred)
            actuals.append(label)

        os.makedirs(self.metrics_cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'predictions': predictions, 'actuals': actuals}, f)

        print("Bayesian evaluation complete!")
        Metrics.print_metrics_report(predictions, actuals)

    def save_model(self, path="../cache/bayesian_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"likelihood": self.likelihood, "prior": self.prior}, f)
        print(f"Bayesian model saved → {path}")

    def load_model(self, path="../cache/bayesian_model.pkl") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.likelihood = data["likelihood"]
            self.prior = data["prior"]
            self.trained = True
        print(f"Bayesian model loaded → {path}")
        return True