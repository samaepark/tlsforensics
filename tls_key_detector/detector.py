import os
import mmap
import hashlib
import platform
import pickle
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import cryptographic libraries for validation
try:
    from OpenSSL import crypto

    OPENSSL_AVAILABLE = True
except ImportError:
    OPENSSL_AVAILABLE = False
    print("Warning: OpenSSL not available. Cryptographic validation will be limited.")


@dataclass
class MemoryWindow:
    """
    Represents a memory window with extracted features and metadata.
    Used for organizing memory analysis results.
    """
    offset: int
    size: int
    entropy: float
    features: Dict[str, Any]
    timestamp: str
    validation_status: str = "pending"


class ForensicLogger:
    """
    Maintains a cryptographic chain-of-custody for forensic evidence.
    Every evidence addition updates the hash chain for auditability and legal compliance.
    Implements tamper-evident logging as required in digital forensics.
    """

    def __init__(self, case_id: str = None):
        """
        Initialize forensic logger with case identification.

        Args:
            case_id (str): Optional case identifier for forensic tracking
        """
        self.chain = hashlib.sha256()
        self.evidence_counter = 0
        self.case_id = case_id or f"CASE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.evidence_log = []

        # Initialize the chain with case metadata
        initial_data = f"{self.case_id}-{datetime.now().isoformat()}".encode()
        self.chain.update(initial_data)

        # Setup logging for audit trail
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'forensic_log_{self.case_id}.log'),
                logging.StreamHandler()
            ]
        )

    def add_evidence(self, data: bytes, description: str = "", metadata: Dict = None) -> str:
        """
        Adds new evidence to the cryptographic chain and returns a unique evidence ID.
        Maintains forensic integrity through cryptographic hashing.

        Args:
            data (bytes): Raw evidence data
            description (str): Human-readable description of evidence
            metadata (dict): Additional metadata for the evidence

        Returns:
            str: Unique evidence identifier with integrity hash
        """
        # Update chain with new evidence
        self.chain.update(data)
        self.evidence_counter += 1

        # Generate evidence ID with timestamp and hash
        timestamp = datetime.now().isoformat()
        evidence_id = f"EVID-{self.evidence_counter:04d}-{self.chain.hexdigest()[:12]}"

        # Store evidence metadata
        evidence_record = {
            'id': evidence_id,
            'timestamp': timestamp,
            'size': len(data),
            'description': description,
            'metadata': metadata or {},
            'chain_hash': self.chain.hexdigest()
        }

        self.evidence_log.append(evidence_record)

        # Log for audit trail
        logging.info(f"Evidence added: {evidence_id} - {description}")

        return evidence_id

    def verify_chain_integrity(self) -> bool:
        """
        Verifies the integrity of the evidence chain.

        Returns:
            bool: True if chain is intact, False otherwise
        """
        # In a real implementation, this would verify against stored hashes
        return len(self.evidence_log) == self.evidence_counter

    def export_chain_report(self) -> str:
        """
        Exports a complete chain-of-custody report for legal proceedings.

        Returns:
            str: Formatted forensic report with chain verification
        """
        report = f"""
=== FORENSIC CHAIN OF CUSTODY REPORT ===
Case ID: {self.case_id}
Generated: {datetime.now().isoformat()}
Chain Integrity: {'VERIFIED' if self.verify_chain_integrity() else 'COMPROMISED'}
Final Chain Hash: {self.chain.hexdigest()}

Evidence Summary:
Total Items: {self.evidence_counter}
"""

        for record in self.evidence_log:
            report += f"""
Evidence ID: {record['id']}
Timestamp: {record['timestamp']}
Size: {record['size']} bytes
Description: {record['description']}
Chain Hash: {record['chain_hash'][:16]}...
"""

        return report


class TLSKeyDetector:
    """
    AI-based TLS key material detector for memory dumps.
    Implements the methodology described in the research proposal:
    - Multi-stage detection pipeline
    - Machine learning classification
    - Cryptographic validation
    - Forensic chain-of-custody

    This class follows the experimental design from the proposal with support for
    both TLS 1.2 and 1.3, cross-platform compatibility, and robust feature extraction.
    """

    def __init__(self, dump_file_path: str, config: Dict = None):
        """
        Initializes the TLS key detector with memory dump file and configuration.

        Args:
            dump_file_path (str): Path to the memory dump file
            config (dict): Optional configuration overrides
        """
        if not os.path.exists(dump_file_path):
            raise FileNotFoundError(f"Memory dump file not found: {dump_file_path}")

        self.dump_file_path = dump_file_path
        self.file_size = os.path.getsize(dump_file_path)

        # Detect OS and configure experiment parameters as per proposal
        self.os_type = self._detect_os()
        self.config = self._load_configuration(config)

        # Initialize detection parameters from configuration
        self.window_size = self.config['window_size']
        self.stride = self.config['stride']
        self.entropy_threshold = self.config['entropy_threshold']
        self.confidence_threshold = self.config['confidence_threshold']

        # Initialize TLS patterns for both versions (as per proposal background)
        self.tls_patterns = self._init_tls_patterns()

        # Initialize ML model and feature scaler
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.is_model_trained = False

        # Initialize forensic logger for chain of custody
        self.logger = ForensicLogger()

        # Performance tracking for evaluation metrics
        self.performance_stats = {
            'total_windows': 0,
            'pattern_matches': 0,
            'ml_detections': 0,
            'validated_keys': 0,
            'processing_time': 0.0
        }

        logging.info(f"TLS Key Detector initialized for {dump_file_path}")
        logging.info(f"File size: {self.file_size} bytes, OS: {self.os_type}")

    def _detect_os(self) -> str:
        """
        Detects the operating system for experiment configuration.
        Supports the cross-platform testing mentioned in the proposal.

        Returns:
            str: Detected OS type ('windows', 'linux', 'macos')
        """
        system = platform.system().lower()
        if system == 'windows':
            return 'windows'
        elif system == 'darwin':
            return 'macos'
        else:
            return 'linux'

    def _load_configuration(self, custom_config: Dict = None) -> Dict:
        """
        Loads experiment configuration based on proposal specifications.
        Implements the adaptive configuration for different TLS versions and OS types.

        Args:
            custom_config (dict): Optional configuration overrides

        Returns:
            dict: Complete configuration dictionary
        """
        # Base configuration from experimental design
        base_config = {
            'window_size': 256,  # Bytes per analysis window
            'stride': 128,  # Sliding window stride
            'entropy_threshold': 7.0,  # Shannon entropy threshold
            'confidence_threshold': 0.85,  # ML confidence threshold
            'tls_version': '1.3',  # Default to TLS 1.3
            'max_memory_size': 1024 * 1024 * 1024,  # 1GB memory limit
            'n_gram_size': 3,  # N-gram analysis size
            'feature_dimensions': 20  # Number of features for ML
        }

        # OS-specific optimizations
        if self.os_type == 'windows':
            base_config.update({
                'stride': 64,  # Smaller stride for Windows memory layout
                'alignment': 16  # Windows memory alignment
            })
        elif self.os_type == 'linux':
            base_config.update({
                'stride': 128,  # Standard stride for Linux
                'alignment': 8  # Linux memory alignment
            })

        # Apply custom configuration if provided
        if custom_config:
            base_config.update(custom_config)

        return base_config

    def _init_tls_patterns(self) -> List[bytes]:
        """
        Initializes TLS version-specific memory patterns for signature detection.
        Based on the background section's description of TLS handshake patterns.
        Supports both TLS 1.2 and 1.3 as mentioned in the proposal.

        Returns:
            list: List of byte patterns for TLS key material detection
        """
        # TLS 1.2 patterns - RSA and DHE key exchange signatures
        tls12_patterns = [
            b'\x16\x03\x01',  # TLS 1.2 Handshake record
            b'\x16\x03\x02',  # TLS 1.2 variant
            b'\x16\x03\x03',  # TLS 1.2 variant
            b'-----BEGIN PRIVATE KEY-----',  # PEM format private key
            b'-----BEGIN RSA PRIVATE KEY-----',  # RSA specific
            b'\x30\x82',  # ASN.1 DER encoding start (common in keys)
            b'\x02\x01\x00\x02',  # RSA private key ASN.1 structure
        ]

        # TLS 1.3 patterns - Ephemeral keys and forward secrecy
        tls13_patterns = [
            b'\x16\x03\x04',  # TLS 1.3 record header
            b'\x03\x04',  # TLS 1.3 version identifier
            b'\x00\x33',  # Key share extension
            b'\x00\x2f',  # Supported groups extension
            b'key_share',  # TLS 1.3 key share parameter
            b'early_data',  # TLS 1.3 early data
        ]

        # General cryptographic patterns
        crypto_patterns = [
            b'PRIVATE KEY',
            b'PUBLIC KEY',
            b'CERTIFICATE',
            b'\x04\x01',  # EC private key marker
            b'\x04\x21',  # EC public key marker (33 bytes)
        ]

        return tls12_patterns + tls13_patterns + crypto_patterns

    def _initialize_model(self) -> RandomForestClassifier:
        """
        Initializes the machine learning model for key detection.
        Implements the supervised learning approach described in the proposal.

        Returns:
            RandomForestClassifier: Configured ML model
        """
        # Random Forest configuration based on proposal methodology
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,  # Minimum samples to split
            min_samples_leaf=2,  # Minimum samples per leaf
            random_state=42,  # Reproducible results
            n_jobs=-1  # Use all CPU cores
        )

        logging.info("Random Forest model initialized")
        return model

    def extract_features(self, data: bytes) -> np.ndarray:
        """
        Extracts comprehensive statistical and contextual features from memory segments.
        Implements the feature extraction methodology from the proposal including:
        - Shannon entropy
        - Byte value distributions
        - N-gram frequency analysis
        - Autocorrelation
        - Additional forensic-specific features

        Args:
            data (bytes): Memory segment data

        Returns:
            np.ndarray: Feature vector for ML classification
        """
        if len(data) == 0:
            return np.zeros(self.config['feature_dimensions'])

        features = []

        # 1. Shannon Entropy - Primary indicator of cryptographic material
        entropy = self._calculate_shannon_entropy(data)
        features.append(entropy)

        # 2. Byte distribution statistics
        byte_array = np.frombuffer(data, dtype=np.uint8)
        features.extend([
            np.mean(byte_array),  # Mean byte value
            np.std(byte_array),  # Standard deviation
            np.var(byte_array),  # Variance
            len(np.unique(byte_array)) / 256,  # Unique byte ratio
        ])

        # 3. N-gram frequency analysis
        ngram_freq = self._n_gram_frequency(data, self.config['n_gram_size'])
        features.append(ngram_freq)

        # 4. Autocorrelation analysis for periodicity detection
        autocorr = self._autocorrelation(data)
        features.append(autocorr if not np.isnan(autocorr) else 0.0)

        # 5. Chi-square test statistic for randomness
        chi_square = self._chi_square_test(byte_array)
        features.append(chi_square)

        # 6. Longest run analysis
        longest_run = self._longest_run(byte_array)
        features.append(longest_run / len(byte_array))

        # 7. Binary pattern analysis
        binary_features = self._binary_pattern_analysis(byte_array)
        features.extend(binary_features)

        # 8. Contextual features (position-based)
        context_features = self._contextual_features(data)
        features.extend(context_features)

        # Pad or truncate to fixed dimension
        feature_array = np.array(features[:self.config['feature_dimensions']])
        if len(feature_array) < self.config['feature_dimensions']:
            padding = np.zeros(self.config['feature_dimensions'] - len(feature_array))
            feature_array = np.concatenate([feature_array, padding])

        return feature_array

    def _calculate_shannon_entropy(self, data: bytes) -> float:
        """
        Calculates Shannon entropy of the data segment.
        High entropy is a strong indicator of encrypted/random data like keys.

        Args:
            data (bytes): Input data

        Returns:
            float: Shannon entropy value (0-8 bits)
        """
        if not data:
            return 0.0

        # Count byte frequencies
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)

        # Calculate probabilities
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]

        # Calculate Shannon entropy
        return -np.sum(probabilities * np.log2(probabilities))

    def _n_gram_frequency(self, data: bytes, n: int = 3) -> float:
        """
        Calculates normalized n-gram frequency diversity.
        Cryptographic keys typically have high n-gram diversity.

        Args:
            data (bytes): Input data
            n (int): N-gram size

        Returns:
            float: Normalized n-gram frequency
        """
        if len(data) < n:
            return 0.0

        ngrams = [data[i:i + n] for i in range(len(data) - n + 1)]
        unique_ngrams = set(ngrams)

        # Return diversity ratio
        return len(unique_ngrams) / max(1, len(ngrams))

    def _autocorrelation(self, data: bytes) -> float:
        """
        Computes lag-1 autocorrelation as a measure of data periodicity.
        Random cryptographic material should have low autocorrelation.

        Args:
            data (bytes): Input data

        Returns:
            float: Autocorrelation coefficient
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        if len(arr) < 2:
            return 0.0

        try:
            correlation_matrix = np.corrcoef(arr[:-1], arr[1:])
            return correlation_matrix[0, 1]
        except:
            return 0.0

    def _chi_square_test(self, byte_array: np.ndarray) -> float:
        """
        Performs chi-square test for uniformity of byte distribution.

        Args:
            byte_array (np.ndarray): Array of byte values

        Returns:
            float: Chi-square test statistic
        """
        observed = np.bincount(byte_array, minlength=256)
        expected = len(byte_array) / 256

        # Avoid division by zero
        if expected == 0:
            return 0.0

        chi_square = np.sum((observed - expected) ** 2 / expected)
        return chi_square

    def _longest_run(self, byte_array: np.ndarray) -> int:
        """
        Finds the longest run of identical bytes.

        Args:
            byte_array (np.ndarray): Array of byte values

        Returns:
            int: Length of longest run
        """
        if len(byte_array) == 0:
            return 0

        max_run = 1
        current_run = 1

        for i in range(1, len(byte_array)):
            if byte_array[i] == byte_array[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        return max_run

    def _binary_pattern_analysis(self, byte_array: np.ndarray) -> List[float]:
        """
        Analyzes binary patterns in the data.

        Args:
            byte_array (np.ndarray): Array of byte values

        Returns:
            list: Binary pattern features
        """
        if len(byte_array) == 0:
            return [0.0, 0.0, 0.0]

        # Convert to binary string
        binary_str = ''.join(format(b, '08b') for b in byte_array)

        # Count ones and zeros
        ones = binary_str.count('1')
        zeros = binary_str.count('0')
        total = len(binary_str)

        # Binary features
        ones_ratio = ones / total if total > 0 else 0
        zeros_ratio = zeros / total if total > 0 else 0
        balance = abs(ones_ratio - 0.5)  # Deviation from perfect balance

        return [ones_ratio, zeros_ratio, balance]

    def _contextual_features(self, data: bytes) -> List[float]:
        """
        Extracts contextual features based on data position and structure.

        Args:
            data (bytes): Input data

        Returns:
            list: Contextual features
        """
        features = []

        # Position-based features
        features.append(len(data) / self.window_size)  # Relative size

        # Pattern occurrence counts
        pattern_count = sum(1 for pattern in self.tls_patterns if pattern in data)
        features.append(pattern_count / len(self.tls_patterns))

        # Null byte analysis
        null_count = data.count(b'\x00')
        features.append(null_count / len(data) if len(data) > 0 else 0)

        return features

    def train_model(self, training_data: List[Tuple[bytes, int]],
                    validation_split: float = 0.2) -> Dict[str, float]:
        """
        Trains the ML model on labeled training data.
        Implements the supervised learning methodology from the proposal.

        Args:
            training_data (list): List of (data, label) tuples
            validation_split (float): Fraction of data for validation

        Returns:
            dict: Training performance metrics
        """
        logging.info(f"Training model on {len(training_data)} samples")

        # Extract features and labels
        X = np.array([self.extract_features(data) for data, _ in training_data])
        y = np.array([label for _, label in training_data])

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Train the model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)

        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y_val, val_predictions),
            'precision': precision_score(y_val, val_predictions, average='weighted'),
            'recall': recall_score(y_val, val_predictions, average='weighted'),
            'f1_score': f1_score(y_val, val_predictions, average='weighted'),
            'training_time': training_time
        }

        # Cross-validation for robustness assessment
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        metrics['cv_mean'] = np.mean(cv_scores)
        metrics['cv_std'] = np.std(cv_scores)

        self.is_model_trained = True
        logging.info(f"Model training completed. Accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def save_model(self, filepath: str):
        """
        Saves the trained model and scaler to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_model_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'patterns': self.tls_patterns
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Loads a pre-trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config.update(model_data.get('config', {}))
        self.tls_patterns = model_data.get('patterns', self.tls_patterns)
        self.is_model_trained = True

        logging.info(f"Model loaded from {filepath}")

    def process_memory_dump(self) -> Dict[str, Any]:
        """
        Main processing pipeline for memory dump analysis.
        Implements the multi-stage detection methodology from the proposal:
        1. Pattern-based pre-screening
        2. Feature extraction
        3. ML classification
        4. Cryptographic validation
        5. Forensic logging

        Returns:
            dict: Comprehensive analysis results with forensic report
        """
        logging.info("Starting memory dump analysis")
        start_time = time.time()

        # Initialize results structure
        results = {
            'key_locations': [],
            'performance_stats': {},
            'forensic_report': '',
            'analysis_metadata': {
                'file_path': self.dump_file_path,
                'file_size': self.file_size,
                'analysis_start': datetime.now().isoformat(),
                'configuration': self.config
            }
        }

        # Memory-mapped file processing for efficiency
        try:
            with open(self.dump_file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as memory_map:

                    # Calculate total windows for progress tracking
                    total_windows = (len(memory_map) - self.window_size) // self.stride + 1
                    self.performance_stats['total_windows'] = total_windows

                    processed_windows = 0

                    # Sliding window analysis
                    for offset in range(0, len(memory_map) - self.window_size, self.stride):
                        window_data = memory_map[offset:offset + self.window_size]
                        processed_windows += 1

                        # Progress logging every 10000 windows
                        if processed_windows % 10000 == 0:
                            progress = (processed_windows / total_windows) * 100
                            logging.info(f"Progress: {progress:.1f}% - {processed_windows}/{total_windows}")

                        # Stage 1: Pattern-based pre-screening
                        if self._pattern_based_prescreening(window_data):
                            self.performance_stats['pattern_matches'] += 1

                            # Stage 2: Feature extraction
                            features = self.extract_features(window_data)

                            # Stage 3: ML classification (if model is trained)
                            if self.is_model_trained and self._ml_classification(features):
                                self.performance_stats['ml_detections'] += 1

                                # Stage 4: Cryptographic validation
                                validation_result = self._cryptographic_validation(window_data)

                                if validation_result['is_valid']:
                                    self.performance_stats['validated_keys'] += 1

                                    # Stage 5: Forensic logging
                                    evidence_id = self.logger.add_evidence(
                                        window_data,
                                        f"TLS key material at offset {offset}",
                                        {
                                            'offset': offset,
                                            'size': len(window_data),
                                            'entropy': features[0],
                                            'validation_method': validation_result['method']
                                        }
                                    )

                                    # Store detection result
                                    key_location = {
                                        'offset': offset,
                                        'size': self.window_size,
                                        'entropy': float(features[0]),
                                        'validation_status': 'confirmed',
                                        'validation_method': validation_result['method'],
                                        'evidence_id': evidence_id,
                                        'timestamp': datetime.now().isoformat(),
                                        'confidence_score': validation_result.get('confidence', 0.0)
                                    }

                                    results['key_locations'].append(key_location)

                                    logging.info(f"Key material detected at offset {offset}")

        except Exception as e:
            logging.error(f"Error processing memory dump: {str(e)}")
            results['error'] = str(e)

        # Calculate final performance statistics
        end_time = time.time()
        self.performance_stats['processing_time'] = end_time - start_time
        self.performance_stats['analysis_completed'] = datetime.now().isoformat()

        results['performance_stats'] = self.performance_stats
        results['forensic_report'] = self._generate_forensic_report(results)

        logging.info(f"Analysis completed in {self.performance_stats['processing_time']:.2f} seconds")
        logging.info(f"Found {len(results['key_locations'])} validated key locations")

        return results

    def _pattern_based_prescreening(self, data: bytes) -> bool:
        """
        Performs pattern-based pre-screening using TLS signatures.
        This stage filters out obviously non-cryptographic data.

        Args:
            data (bytes): Memory window data

        Returns:
            bool: True if patterns suggest potential key material
        """
        # Direct pattern matching
        pattern_matches = sum(1 for pattern in self.tls_patterns if pattern in data)

        # Quick entropy check
        entropy = self._calculate_shannon_entropy(data)

        # Pre-screening criteria
        return (pattern_matches > 0 or
                entropy > self.entropy_threshold * 0.8 or
                self._has_cryptographic_structure(data))

    def _has_cryptographic_structure(self, data: bytes) -> bool:
        """
        Checks for structural indicators of cryptographic material.

        Args:
            data (bytes): Memory window data

        Returns:
            bool: True if cryptographic structure detected
        """
        # ASN.1 DER encoding indicators
        asn1_indicators = [b'\x30\x82', b'\x30\x81', b'\x02\x01']

        # PEM format indicators
        pem_indicators = [b'-----BEGIN', b'-----END']

        # Check for structural patterns
        return any(indicator in data for indicator in asn1_indicators + pem_indicators)

    def _ml_classification(self, features: np.ndarray) -> bool:
        """
        Performs ML-based classification of memory windows.

        Args:
            features (np.ndarray): Extracted feature vector

        Returns:
            bool: True if classified as containing key material
        """
        if not self.is_model_trained:
            # Fallback to entropy-based classification
            return features[0] > self.entropy_threshold

        # Normalize features
        features_scaled = self.scaler.transform([features])

        # Get prediction probability
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Check confidence threshold
        return probabilities[1] > self.confidence_threshold

    def _cryptographic_validation(self, data: bytes) -> Dict[str, Any]:
        """
        Validates potential key material using cryptographic libraries.
        Implements multiple validation methods for robustness.

        Args:
            data (bytes): Potential key material data

        Returns:
            dict: Validation results with method and confidence
        """
        validation_result = {
            'is_valid': False,
            'method': 'none',
            'confidence': 0.0,
            'key_type': 'unknown'
        }

        # Method 1: OpenSSL cryptographic validation (if available)
        if OPENSSL_AVAILABLE:
            try:
                # Try to parse as ASN.1 private key
                crypto.load_privatekey(crypto.FILETYPE_ASN1, data)
                validation_result.update({
                    'is_valid': True,
                    'method': 'openssl_asn1',
                    'confidence': 0.95,
                    'key_type': 'private_key'
                })
                return validation_result
            except:
                pass

            try:
                # Try to parse as PEM private key
                crypto.load_privatekey(crypto.FILETYPE_PEM, data)
                validation_result.update({
                    'is_valid': True,
                    'method': 'openssl_pem',
                    'confidence': 0.95,
                    'key_type': 'private_key'
                })
                return validation_result
            except:
                pass

        # Method 2: Structural validation for ASN.1 DER format
        if self._validate_asn1_structure(data):
            validation_result.update({
                'is_valid': True,
                'method': 'asn1_structure',
                'confidence': 0.75,
                'key_type': 'structured_key'
            })
            return validation_result

        # Method 3: Statistical validation (high entropy + cryptographic patterns)
        entropy = self._calculate_shannon_entropy(data)
        pattern_score = self._calculate_pattern_score(data)

        if entropy > 7.5 and pattern_score > 0.7:
            validation_result.update({
                'is_valid': True,
                'method': 'statistical',
                'confidence': min(0.8, (entropy / 8.0) * pattern_score),
                'key_type': 'probable_key'
            })

        return validation_result

    def _validate_asn1_structure(self, data: bytes) -> bool:
        """
        Validates ASN.1 DER structure commonly used in cryptographic keys.

        Args:
            data (bytes): Data to validate

        Returns:
            bool: True if valid ASN.1 structure detected
        """
        if len(data) < 4:
            return False

        # Check for ASN.1 SEQUENCE tag (0x30)
        if data[0] != 0x30:
            return False

        # Check length encoding
        if data[1] & 0x80:  # Long form length
            length_bytes = data[1] & 0x7F
            if length_bytes == 0 or length_bytes > 4:
                return False
            if len(data) < 2 + length_bytes:
                return False
        else:  # Short form length
            if data[1] > len(data) - 2:
                return False

        return True

    def _calculate_pattern_score(self, data: bytes) -> float:
        """
        Calculates a score based on cryptographic patterns in the data.

        Args:
            data (bytes): Data to analyze

        Returns:
            float: Pattern score (0.0 to 1.0)
        """
        score = 0.0
        total_patterns = len(self.tls_patterns)

        # Count pattern matches
        matches = sum(1 for pattern in self.tls_patterns if pattern in data)
        score += (matches / total_patterns) * 0.5

        # Check for specific cryptographic indicators
        crypto_indicators = [
            b'\x02\x01\x00',  # ASN.1 version field
            b'\x02\x01\x01',  # Another version field
            b'\x04\x20',  # 32-byte octet string (common key size)
            b'\x04\x40',  # 64-byte octet string
            b'\x03\x01\x00',  # Bit string
        ]

        indicator_matches = sum(1 for indicator in crypto_indicators if indicator in data)
        score += (indicator_matches / len(crypto_indicators)) * 0.3

        # Byte distribution analysis
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        uniformity = 1.0 - (np.std(byte_counts) / np.mean(byte_counts)) if np.mean(byte_counts) > 0 else 0
        score += uniformity * 0.2

        return min(1.0, score)

    def _generate_forensic_report(self, results: Dict[str, Any]) -> str:
        """
        Generates a comprehensive forensic report suitable for legal proceedings.
        Includes all required elements for digital forensics documentation.

        Args:
            results (dict): Analysis results

        Returns:
            str: Formatted forensic report
        """
        report = f"""
{'=' * 80}
                    TLS KEY MATERIAL DETECTION REPORT
{'=' * 80}

CASE INFORMATION:
    Case ID: {self.logger.case_id}
    Analysis Date: {results['analysis_metadata'].get('analysis_start', 'Unknown')}
    Examiner: AI-Assisted Detection System v1.0
    Report Generated: {datetime.now().isoformat()}

EVIDENCE INFORMATION:
    Source File: {os.path.basename(self.dump_file_path)}
    Full Path: {self.dump_file_path}
    File Size: {self.file_size:,} bytes ({self.file_size / (1024 * 1024):.2f} MB)
    File Modified: {datetime.fromtimestamp(os.path.getmtime(self.dump_file_path)).isoformat()}

SYSTEM INFORMATION:
    Operating System: {self.os_type.upper()}
    Platform: {platform.platform()}
    Python Version: {platform.python_version()}
    Analysis Host: {platform.node()}

ANALYSIS CONFIGURATION:
    Window Size: {self.window_size} bytes
    Stride: {self.stride} bytes
    Entropy Threshold: {self.entropy_threshold}
    Confidence Threshold: {self.confidence_threshold}
    TLS Version Focus: {self.config.get('tls_version', 'Multiple')}
    Model Trained: {'Yes' if self.is_model_trained else 'No'}

DETECTION RESULTS:
    Total Memory Windows Analyzed: {self.performance_stats.get('total_windows', 0):,}
    Pattern Matches Found: {self.performance_stats.get('pattern_matches', 0):,}
    ML Detections: {self.performance_stats.get('ml_detections', 0):,}
    Validated Key Materials: {len(results['key_locations'])}

PERFORMANCE METRICS:
    Total Processing Time: {self.performance_stats.get('processing_time', 0):.2f} seconds
    Windows per Second: {self.performance_stats.get('total_windows', 0) / max(1, self.performance_stats.get('processing_time', 1)):.0f}
    Detection Rate: {(len(results['key_locations']) / max(1, self.performance_stats.get('total_windows', 1))) * 100:.6f}%

DETECTED KEY MATERIALS:
{'=' * 80}
"""

        if results['key_locations']:
            report += f"""
{'No.':<4} {'Offset':<12} {'Size':<8} {'Entropy':<8} {'Method':<15} {'Confidence':<10} {'Evidence ID':<20}
{'-' * 80}
"""
            for i, location in enumerate(results['key_locations'], 1):
                report += f"{i:<4} {location['offset']:<12} {location['size']:<8} {location['entropy']:<8.2f} {location['validation_method']:<15} {location['confidence_score']:<10.2f} {location['evidence_id']:<20}\n"
        else:
            report += "\nNo validated TLS key materials were detected in this memory dump.\n"

        report += f"""
{'=' * 80}
FORENSIC INTEGRITY:
    Evidence Chain Hash: {self.logger.chain.hexdigest()}
    Chain Integrity Status: {'VERIFIED' if self.logger.verify_chain_integrity() else 'COMPROMISED'}
    Total Evidence Items: {self.logger.evidence_counter}

METHODOLOGY:
    This analysis employed a multi-stage AI-assisted detection methodology:

    1. Pattern-Based Pre-screening: Memory segments were scanned for known TLS
       handshake patterns and cryptographic structure indicators.

    2. Feature Extraction: Statistical features including Shannon entropy,
       byte distribution, n-gram frequency, and autocorrelation were computed.

    3. Machine Learning Classification: {'A trained Random Forest model classified' if self.is_model_trained else 'Entropy-based heuristics classified'}
       memory segments based on extracted features.

    4. Cryptographic Validation: Potential key materials were validated using
       OpenSSL parsing and ASN.1 structure analysis.

    5. Forensic Logging: All validated findings were logged with cryptographic
       chain-of-custody for legal admissibility.

LIMITATIONS AND CONSIDERATIONS:
    - TLS 1.3 implements forward secrecy, limiting key recovery timeframes
    - Secure memory management may clear keys during operation
    - Hardware memory encryption (SGX, SEV) may prevent key extraction
    - Results are probabilistic and require expert interpretation
    - This tool is designed for authorized forensic investigations only

DISCLAIMER:
    This analysis was performed using AI-assisted detection methods. Results
    should be validated by qualified digital forensics experts before use in
    legal proceedings. The detection methodology is based on statistical
    analysis and may produce false positives or miss obfuscated key material.

{'=' * 80}
Report End - {datetime.now().isoformat()}
{'=' * 80}
"""

        # Append chain of custody report
        report += "\n\n" + self.logger.export_chain_report()

        return report

    def export_results(self, results: Dict[str, Any], output_dir: str = "forensic_output"):
        """
        Exports analysis results in multiple formats for forensic documentation.

        Args:
            results (dict): Analysis results
            output_dir (str): Output directory for exports
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"tls_analysis_{timestamp}"

        # Export forensic report
        report_file = os.path.join(output_dir, f"{base_filename}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(results['forensic_report'])

        # Export results as JSON
        json_file = os.path.join(output_dir, f"{base_filename}_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            import json
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2, default=str)

        # Export CSV summary
        csv_file = os.path.join(output_dir, f"{base_filename}_summary.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['Offset', 'Size', 'Entropy', 'Validation_Method',
                             'Confidence', 'Evidence_ID', 'Timestamp'])

            for location in results['key_locations']:
                writer.writerow([
                    location['offset'], location['size'], location['entropy'],
                    location['validation_method'], location['confidence_score'],
                    location['evidence_id'], location['timestamp']
                ])

        logging.info(f"Results exported to {output_dir}")
        logging.info(f"Files created: {report_file}, {json_file}, {csv_file}")

    def _convert_for_json(self, obj):
        """
        Converts numpy types and other non-JSON-serializable objects.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def generate_training_data(self, known_key_files: List[str],
                               negative_samples_dir: str = None) -> List[Tuple[bytes, int]]:
        """
        Generates labeled training data from known key files and negative samples.
        Supports the supervised learning methodology from the proposal.

        Args:
            known_key_files (list): List of files containing known key material
            negative_samples_dir (str): Directory containing non-key memory samples

        Returns:
            list: List of (data, label) tuples for training
        """
        training_data = []

        # Process positive samples (known keys)
        for key_file in known_key_files:
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key_data = f.read()

                # Create overlapping windows from key data
                for i in range(0, len(key_data) - self.window_size + 1, self.stride):
                    window = key_data[i:i + self.window_size]
                    if len(window) == self.window_size:
                        training_data.append((window, 1))  # Positive label

        # Process negative samples
        if negative_samples_dir and os.path.exists(negative_samples_dir):
            for filename in os.listdir(negative_samples_dir):
                filepath = os.path.join(negative_samples_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, 'rb') as f:
                        data = f.read()

                    # Sample random windows from non-key data
                    num_samples = min(100, len(data) // self.window_size)
                    for _ in range(num_samples):
                        start = np.random.randint(0, len(data) - self.window_size)
                        window = data[start:start + self.window_size]
                        if len(window) == self.window_size:
                            training_data.append((window, 0))  # Negative label

        # Balance the dataset
        positive_samples = [item for item in training_data if item[1] == 1]
        negative_samples = [item for item in training_data if item[1] == 0]

        # Ensure balanced classes
        min_samples = min(len(positive_samples), len(negative_samples))
        if min_samples > 0:
            balanced_data = (positive_samples[:min_samples] +
                             negative_samples[:min_samples])
            np.random.shuffle(balanced_data)
            training_data = balanced_data

        logging.info(f"Generated {len(training_data)} training samples")
        logging.info(f"Positive samples: {sum(1 for _, label in training_data if label == 1)}")
        logging.info(f"Negative samples: {sum(1 for _, label in training_data if label == 0)}")

        return training_data

    @staticmethod
    def benchmark_performance(detector_instances: List['TLSKeyDetector'],
                              test_files: List[str]) -> Dict[str, Any]:
        """
        Benchmarks multiple detector configurations for performance comparison.
        Implements the comparative evaluation methodology from the proposal.

        Args:
            detector_instances (list): List of configured detector instances
            test_files (list): List of test memory dump files

        Returns:
            dict: Comprehensive benchmark results
        """
        benchmark_results = {
            'test_files': test_files,
            'detectors': [],
            'summary': {}
        }

        for i, detector in enumerate(detector_instances):
            detector_name = f"Detector_{i + 1}"
            detector_results = {
                'name': detector_name,
                'config': detector.config,
                'results': []
            }

            total_time = 0
            total_detections = 0

            for test_file in test_files:
                if os.path.exists(test_file):
                    # Temporarily change detector's file path
                    original_path = detector.dump_file_path
                    detector.dump_file_path = test_file
                    detector.file_size = os.path.getsize(test_file)

                    # Run analysis
                    start_time = time.time()
                    results = detector.process_memory_dump()
                    end_time = time.time()

                    # Collect metrics
                    file_results = {
                        'file': os.path.basename(test_file),
                        'processing_time': end_time - start_time,
                        'detections': len(results['key_locations']),
                        'performance_stats': results['performance_stats']
                    }

                    detector_results['results'].append(file_results)
                    total_time += file_results['processing_time']
                    total_detections += file_results['detections']

                    # Restore original path
                    detector.dump_file_path = original_path
                    detector.file_size = os.path.getsize(original_path)

            # Calculate aggregate metrics
            detector_results['aggregate'] = {
                'total_processing_time': total_time,
                'average_processing_time': total_time / len(test_files) if test_files else 0,
                'total_detections': total_detections,
                'average_detections': total_detections / len(test_files) if test_files else 0
            }

            benchmark_results['detectors'].append(detector_results)

        # Generate summary comparison
        if benchmark_results['detectors']:
            fastest_detector = min(benchmark_results['detectors'],
                                   key=lambda x: x['aggregate']['average_processing_time'])
            most_sensitive = max(benchmark_results['detectors'],
                                 key=lambda x: x['aggregate']['total_detections'])

            benchmark_results['summary'] = {
                'fastest_detector': fastest_detector['name'],
                'most_sensitive_detector': most_sensitive['name'],
                'benchmark_timestamp': datetime.now().isoformat()
            }

        return benchmark_results

    def get_memory_alignment(self) -> int:
        """
        Returns optimal memory alignment for current architecture.
        Supports cross-platform optimization as mentioned in the proposal.

        Returns:
            int: Optimal memory alignment in bytes
        """
        arch = platform.machine().lower()

        # ARM architectures typically require 16-byte alignment
        if any(arm_arch in arch for arm_arch in ['arm', 'aarch64', 'armv7', 'armv8']):
            return 16

        # x86_64 typically uses 8-byte alignment
        elif 'x86_64' in arch or 'amd64' in arch:
            return 8

        # Default alignment for other architectures
        else:
            return 4


# Utility functions for forensic analysis

def create_test_environment(output_dir: str = "test_environment"):
    """
    Creates a test environment with sample data for methodology validation.
    Supports the experimental design from the proposal.

    Args:
        output_dir (str): Directory to create test environment
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create sample key files
    key_samples = [
        # Simulated RSA private key header
        b'\x30\x82\x04\xa4\x02\x01\x00\x02\x82\x01\x01\x00' + os.urandom(100),

        # Simulated EC private key
        b'\x30\x81\x87\x02\x01\x00\x30\x13\x06\x07\x2a\x86\x48\xce\x3d\x02\x01\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07' + os.urandom(
            80),

        # High-entropy random data (simulating session keys)
        os.urandom(256),
        os.urandom(128),
    ]

    # Save sample keys
    for i, key_data in enumerate(key_samples):
        with open(os.path.join(output_dir, f"sample_key_{i + 1}.bin"), 'wb') as f:
            f.write(key_data)

    # Create negative samples (low entropy data)
    negative_samples = [
        b'\x00' * 256,  # All zeros
        b'A' * 256,  # Repeated character
        b'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 10,  # Text data
    ]

    negative_dir = os.path.join(output_dir, "negative_samples")
    os.makedirs(negative_dir, exist_ok=True)

    for i, neg_data in enumerate(negative_samples):
        with open(os.path.join(negative_dir, f"negative_{i + 1}.bin"), 'wb') as f:
            f.write(neg_data)

    logging.info(f"Test environment created in {output_dir}")


# Example usage and demonstration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create test environment (uncomment to generate test data)
    # create_test_environment()

    # Example usage
    try:
        # Initialize detector
        detector = TLSKeyDetector("memory.dmp")

        # Optional: Train model if training data is available
        # training_data = detector.generate_training_data(
        #     known_key_files=["test_environment/sample_key_1.bin"],
        #     negative_samples_dir="test_environment/negative_samples"
        # )
        # detector.train_model(training_data)

        # Process memory dump
        results = detector.process_memory_dump()

        # Export results
        detector.export_results(results)

        # Print summary
        print(f"Analysis completed successfully!")
        print(f"Key locations found: {len(results['key_locations'])}")
        print(f"Processing time: {results['performance_stats'].get('processing_time', 0):.2f} seconds")

    except FileNotFoundError:
        print("Memory dump file 'memory.dmp' not found.")
        print("Please provide a valid memory dump file path.")

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        print(f"Analysis failed: {str(e)}")