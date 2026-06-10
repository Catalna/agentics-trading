"""
data_pipeline/synthetic_generator.py — Synthetic Data Augmentation for ML Pipeline
Implements advanced augmentation techniques (Bootstrap, Pattern Interpolation)
to increase dataset size and robustness.
"""

import numpy as np
import pandas as pd
from typing import List

class SyntheticAugmentation:
    """
    Generate realistic synthetic data to:
    1. Increase dataset size
    2. Add controlled noise for robustness
    3. Create edge cases for stress testing
    """
    
    def generate_augmented_dataset(self, real_data: pd.DataFrame, target_size: int = 100000) -> pd.DataFrame:
        """
        Master augmentation pipeline (Time-Series Safe)
        """
        augmented_pool = []
        
        # METHOD 1: Duplicate sequence with Noise
        # Create full copies of the timeline with varying noise levels
        noise_copies = self.bootstrap_with_noise(
            real_data,
            noise_levels=[0.005, 0.01, 0.02]  # 0.5%, 1%, 2% noise
        )
        augmented_pool.append(noise_copies)
        
        # Combine all sequentially (DO NOT SHUFFLE)
        full_dataset = pd.concat([real_data] + augmented_pool).reset_index(drop=True)
        return full_dataset
    
    def bootstrap_with_noise(self, data: pd.DataFrame, noise_levels: List[float]) -> pd.DataFrame:
        """
        Create parallel alternate histories of the entire dataset with controlled noise injection.
        Preserves time-series continuity.
        """
        augmented = []
        
        # Calculate ATR if not present (simple proxy for volatility)
        if 'atr' not in data.columns:
            data['atr'] = data['high'] - data['low']
        
        for noise_level in noise_levels:
            # Clone entire sequence
            sample = data.copy()
            
            # Add noise proportional to ATR (volatility-aware)
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, 1, len(sample)) * sample['atr'] * noise_level
                sample[col] += noise
            
            # Maintain OHLC integrity
            sample['high'] = sample[['open', 'high', 'low', 'close']].max(axis=1)
            sample['low'] = sample[['open', 'high', 'low', 'close']].min(axis=1)
            
            # Add volume jitter
            if 'volume' in sample.columns:
                volume_noise = np.random.uniform(0.9, 1.1, len(sample))
                sample['volume'] *= volume_noise
            
            augmented.append(sample)
        
        return pd.concat(augmented) if augmented else pd.DataFrame()
    
    def pattern_interpolation(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Blend two real patterns to create new realistic patterns
        """
        augmented = []
        window = 100  # 100 candles per pattern
        
        if len(data) < window * 2:
            return pd.DataFrame() # Not enough data
            
        iterations = n_samples // window
        
        for _ in range(iterations):
            idx1 = np.random.randint(0, len(data) - window)
            idx2 = np.random.randint(0, len(data) - window)
            
            pattern_a = data.iloc[idx1:idx1+window].copy().reset_index(drop=True)
            pattern_b = data.iloc[idx2:idx2+window].copy().reset_index(drop=True)
            
            # Interpolation weight
            alpha = np.random.uniform(0.3, 0.7)
            
            blended = pattern_a.copy()
            blend_cols = ['open', 'high', 'low', 'close']
            if 'volume' in pattern_a.columns:
                blend_cols.append('volume')
                
            for col in blend_cols:
                blended[col] = alpha * pattern_a[col] + (1 - alpha) * pattern_b[col]
            
            # Add small noise
            noise = np.random.normal(0, 0.01, len(blended))
            for col in ['open', 'high', 'low', 'close']:
                blended[col] *= (1 + noise)
                
            # Maintain OHLC integrity
            blended['high'] = blended[['open', 'high', 'low', 'close']].max(axis=1)
            blended['low'] = blended[['open', 'high', 'low', 'close']].min(axis=1)
            
            augmented.append(blended)
        
        return pd.concat(augmented).reset_index(drop=True) if augmented else pd.DataFrame()
