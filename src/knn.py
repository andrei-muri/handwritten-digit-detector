import numpy as np
from collections import Counter
from typing import List, Tuple
import pickle
import os

class KNN:
    """
    K-Nearest Neighbors classifier that uses histogram-based features
    for image comparison.
    """
    
    def __init__(self, k: int = 5, num_bins: int = 16):
        """
        Initialize KNN classifier.
        
        Args:
            k: Number of nearest neighbors to consider
            num_bins: Number of bins for the histogram features
        """
        self.k = k
        self.num_bins = num_bins
        self.training_features = None
        self.training_labels = None
        self.cached = False
        self.metrics_cache_dir = "../cache/knn_metrics"
    
    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute histogram features from an image.
        
        The histogram represents the distribution of pixel intensities,
        which captures the overall brightness and contrast patterns of the digit.
        
        Args:
            image: Input image as numpy array (784x1 or flattened)
            
        Returns:
            Histogram feature vector (normalized)
        """
        # Flatten if needed
        if image.shape == (784, 1):
            image = image.flatten()
        
        # Compute histogram with specified number of bins
        hist, _ = np.histogram(image, bins=self.num_bins, range=(0.0, 1.0))
        
        # Normalize the histogram
        hist = hist.astype(float)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        
        return hist
    
    def compute_spatial_histogram(self, image: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """
        Compute spatial histogram features by dividing image into grid cells.
        
        This captures both intensity distribution and spatial information,
        making it more discriminative than a global histogram.
        
        Args:
            image: Input image as numpy array (784x1)
            grid_size: Divide image into grid_size x grid_size cells
            
        Returns:
            Concatenated histogram features from all grid cells
        """
        # Reshape to 28x28 if needed
        if image.shape == (784, 1):
            image = image.reshape(28, 28)
        elif image.shape == (784,):
            image = image.reshape(28, 28)
        
        cell_size = 28 // grid_size
        features = []
        
        # Compute histogram for each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                row_start = i * cell_size
                row_end = (i + 1) * cell_size
                col_start = j * cell_size
                col_end = (j + 1) * cell_size
                
                cell = image[row_start:row_end, col_start:col_end]
                cell_hist = self.compute_histogram(cell.flatten())
                features.append(cell_hist)
        
        # Concatenate all cell histograms
        spatial_features = np.concatenate(features)
        return spatial_features
    
    def euclidean_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        return np.sqrt(np.sum((hist1 - hist2) ** 2))
    
    def chi_square_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        epsilon = 1e-10
        distance = np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon))
        return distance * 0.5
    
    def fit(self, training_data: List[Tuple[np.ndarray, int]], use_spatial: bool = True):
        """
        Fit the KNN model by computing histogram features for all training samples.
        
        Args:
            training_data: List of tuples (image, label)
            use_spatial: Whether to use spatial histograms (recommended)
        """
        print(f"Computing histogram features for {len(training_data)} training samples...")
        
        self.training_features = []
        self.training_labels = []
        
        for i, (image, label) in enumerate(training_data):
            if i % 5000 == 0:
                print(f"Processing sample {i}/{len(training_data)}...")
            
            # Compute features
            if use_spatial:
                features = self.compute_spatial_histogram(image)
            else:
                features = self.compute_histogram(image)
            
            self.training_features.append(features)
            self.training_labels.append(label)
        
        self.training_features = np.array(self.training_features)
        self.training_labels = np.array(self.training_labels)
        
        print("Feature computation complete!")
    
    def predict(self, image: np.ndarray, use_spatial: bool = True, 
                distance_metric: str = 'chi_square') -> int:
        """
        Predict the label for a single image.
        
        Args:
            image: Input image as numpy array
            use_spatial: Whether to use spatial histograms
            distance_metric: 'euclidean' or 'chi_square'
            
        Returns:
            Predicted label
        """
        # Compute features for query image
        if use_spatial:
            query_features = self.compute_spatial_histogram(image)
        else:
            query_features = self.compute_histogram(image)
        
        # Compute distances to all training samples
        distances = []
        for train_features in self.training_features:
            if distance_metric == 'chi_square':
                dist = self.chi_square_distance(query_features, train_features)
            else:
                dist = self.euclidean_distance(query_features, train_features)
            if dist < 20:
                distances.append(dist)
        
        # Find k nearest neighbors
        distances = np.array(distances)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.training_labels[k_indices]
        
        # Vote for the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _get_metrics_cache_filename(self, test_data_size: int, use_spatial: bool, 
                                   distance_metric: str) -> str:
        """
        Generate cache filename for metrics.
        
        Args:
            test_data_size: Number of test samples
            use_spatial: Whether spatial histograms were used
            distance_metric: Distance metric used
            
        Returns:
            Cache filename
        """
        spatial_str = "spatial" if use_spatial else "global"
        filename = f"metrics_k{self.k}_size{test_data_size}_{spatial_str}_{distance_metric}.pkl"
        return os.path.join(self.metrics_cache_dir, filename)
    
    def _load_cached_metrics(self, test_data_size: int, use_spatial: bool, 
                            distance_metric: str) -> dict:
        """
        Load cached metrics from file if available.
        
        Returns:
            Cached metrics dictionary or None if not found
        """
        cache_file = self._get_metrics_cache_filename(test_data_size, use_spatial, distance_metric)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded cached metrics from {os.path.basename(cache_file)}")
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}")
                return None
        return None
    
    def _save_metrics_to_cache(self, predictions: List[int], actuals: List[int],
                               test_data_size: int, use_spatial: bool, 
                               distance_metric: str):
        """
        Save predictions and actuals to cache file.
        
        Args:
            predictions: List of predicted labels
            actuals: List of actual labels
            test_data_size: Number of test samples
            use_spatial: Whether spatial histograms were used
            distance_metric: Distance metric used
        """
        os.makedirs(self.metrics_cache_dir, exist_ok=True)
        cache_file = self._get_metrics_cache_filename(test_data_size, use_spatial, distance_metric)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'predictions': predictions,
                    'actuals': actuals,
                    'test_data_size': test_data_size,
                    'use_spatial': use_spatial,
                    'distance_metric': distance_metric,
                    'k': self.k
                }, f)
            print(f"Cached metrics saved to {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def predict_batch(self, test_data: List[Tuple[np.ndarray, int]], 
                     use_spatial: bool = True, distance_metric: str = 'chi_square',
                     use_cache: bool = True) -> Tuple[List[int], List[int]]:
        """
        Predict labels for a batch of test samples.
        
        Args:
            test_data: List of tuples (image, actual_label)
            use_spatial: Whether to use spatial histograms
            distance_metric: 'euclidean' or 'chi_square'
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (predictions, actual_labels)
        """
        test_data_size = len(test_data)
        
        # Try to load from cache
        if use_cache:
            cached_data = self._load_cached_metrics(test_data_size, use_spatial, distance_metric)
            if cached_data is not None:
                return cached_data['predictions'], cached_data['actuals']
        
        # Compute predictions
        predictions = []
        actuals = []
        
        print(f"Predicting {test_data_size} test samples...")
        
        for i, (image, label) in enumerate(test_data):
            if i % 500 == 0:
                print(f"Predicting sample {i}/{test_data_size}...")
            
            pred = self.predict(image, use_spatial, distance_metric)
            predictions.append(pred)
            actuals.append(label)
        
        print("Prediction complete!")
        
        # Save to cache
        if use_cache:
            self._save_metrics_to_cache(predictions, actuals, test_data_size, 
                                       use_spatial, distance_metric)
        
        return predictions, actuals
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, int]], 
                use_spatial: bool = True, distance_metric: str = 'chi_square',
                use_cache: bool = True) -> float:
        """
        Evaluate the KNN classifier on test data.
        
        Args:
            test_data: List of tuples (image, label)
            use_spatial: Whether to use spatial histograms
            distance_metric: 'euclidean' or 'chi_square'
            use_cache: Whether to use cached results if available
            
        Returns:
            Accuracy score
        """
        predictions, actuals = self.predict_batch(test_data, use_spatial, distance_metric, use_cache)
        correct = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
        accuracy = correct / len(actuals)
        return accuracy
    
    def get_metrics(self, test_data: List[Tuple[np.ndarray, int]], 
                   use_spatial: bool = True, distance_metric: str = 'chi_square',
                   use_cache: bool = True) -> dict:
        """
        Get comprehensive metrics for the KNN classifier.
        
        Args:
            test_data: List of tuples (image, label)
            use_spatial: Whether to use spatial histograms
            distance_metric: 'euclidean' or 'chi_square'
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary containing all metrics
        """
        from metrics import Metrics
        
        predictions, actuals = self.predict_batch(test_data, use_spatial, distance_metric, use_cache)
        
        accuracy = Metrics.accuracy(predictions, actuals)
        per_class_metrics = Metrics.precision_recall_f1(predictions, actuals)
        confusion_matrix = Metrics.confusion_matrix(predictions, actuals)
        
        return {
            'accuracy': accuracy,
            'precision': per_class_metrics['precision'],
            'recall': per_class_metrics['recall'],
            'f1': per_class_metrics['f1'],
            'confusion_matrix': confusion_matrix,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def print_metrics_report(self, test_data: List[Tuple[np.ndarray, int]], 
                           use_spatial: bool = True, distance_metric: str = 'chi_square',
                           use_cache: bool = True):
        """
        Print comprehensive metrics report for KNN classifier.
        
        Args:
            test_data: List of tuples (image, label)
            use_spatial: Whether to use spatial histograms
            distance_metric: 'euclidean' or 'chi_square'
            use_cache: Whether to use cached results if available
        """
        from metrics import Metrics
        
        predictions, actuals = self.predict_batch(test_data, use_spatial, distance_metric, use_cache)
        Metrics.print_metrics_report(predictions, actuals)
    
    def save_model(self, filepath: str = "../cache/knn_model.pkl"):
        """
        Save the trained KNN model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'k': self.k,
                'num_bins': self.num_bins,
                'training_features': self.training_features,
                'training_labels': self.training_labels
            }, f)
        print(f"KNN model saved to {filepath}")
    
    def load_model(self, filepath: str = "../cache/knn_model.pkl"):
        """
        Load a trained KNN model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.k = data['k']
                self.num_bins = data['num_bins']
                self.training_features = data['training_features']
                self.training_labels = data['training_labels']
                self.cached = True
            print(f"KNN model loaded from {filepath}")
            return True
        return False
    
    def is_cached(self):
        """Check if model is loaded from cache."""
        return self.cached
    
    def clear_metrics_cache(self):
        """Clear all cached metrics files."""
        if os.path.exists(self.metrics_cache_dir):
            import shutil
            shutil.rmtree(self.metrics_cache_dir)
            os.makedirs(self.metrics_cache_dir, exist_ok=True)
            print("Metrics cache cleared")
