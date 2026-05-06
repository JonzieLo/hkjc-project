import pandas as pd
import numpy as np
import joblib
import logging
from sqlalchemy import create_engine
from hkjc_engine.config import DB_URL
from hkjc_engine.models.feature_factory import HKJCFeatureFactory

logging.basicConfig(level=logging.INFO, format='%(message)s')

def build_copula_matrix():
    logging.info("Fetching deep history for pace copula...")
    factory = HKJCFeatureFactory(DB_URL)
    df = factory.fetch_raw_data('2015-01-01', '2024-01-01')
    df = factory.engineer_features(df)
    
    # 1. Bucket the field into 5 Pace Archetypes (0: Closer, 4: Front-Runner)
    df['pace_archetype'] = pd.qcut(df['relative_early_pace'], q=5, labels=False, duplicates='drop')
    
    # 2. Pivot to get finishing positions by archetype per race
    # If multiple horses in a race share an archetype, take their mean finish
    pivot = df.pivot_table(index='race_id', columns='pace_archetype', values='finish_position', aggfunc='mean')
    pivot = pivot.dropna()
    
    # 3. Calculate Spearman Rank Correlation between archetypes
    sigma = pivot.corr(method='spearman').values
    
    # Convert rank correlation to Pearson correlation for the Gaussian Copula 
    # using the standard transformation: Pearson = 2 * sin(pi/6 * Spearman)
    sigma_gaussian = 2 * np.sin((np.pi / 6) * sigma)
    
    # Fill diagonal with 1.0 explicitly
    np.fill_diagonal(sigma_gaussian, 1.0)
    
    logging.info("\n--- Gaussian Copula Correlation Matrix (Pace 0 to 4) ---")
    logging.info(np.round(sigma_gaussian, 3))
    
    np.save('pace_copula.npy', sigma_gaussian)
    logging.info("Saved to pace_copula.npy")

if __name__ == "__main__":
    build_copula_matrix()