import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import joblib

class CoxIndependentTrainer:
    def __init__(self, db_url):
        self.features = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'jockey_alpha', 'track_width', 'straight_length'
        ]
        
    def train(self, df):
        # 1. Prepare survival targets. 
        # Time = finish_position (lower is 'earlier' survival, meaning higher hazard/strength).
        # Event = 1 (all runners experience the 'event' of finishing).
        df['event'] = 1
        
        # 2. Map continuous pace into an archetype for the frailty term
        df['pace_archetype'] = pd.qcut(df['relative_early_pace'], q=5, labels=False)
        
        train_cols = self.features + ['race_id', 'finish_position', 'event', 'pace_archetype']
        X_train = df[train_cols].copy()
        
        # 3. Fit Stratified CoxPH with Gamma Frailty
        # strata='race_id' restricts the likelihood denominator to the specific race (exact Plackett-Luce).
        # frailty_col='pace_archetype' estimates the unobserved latent variance \theta across archetypes.
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
        cph.fit(
            X_train,
            duration_col='finish_position',
            event_col='event',
            strata=['race_id'],
            cluster_col='pace_archetype', # Injects Gamma frailty Z ~ Gamma(k, \theta)
            robust=True
        )
        
        # 4. Extract marginal probabilities
        # partial_hazard represents \exp(\beta^T X)
        df['hazard'] = cph.predict_partial_hazard(X_train)
        df['P_model'] = df['hazard'] / df.groupby('race_id')['hazard'].transform('sum')
        
        joblib.dump(cph, 'cox_ind_model.pkl')
        return df