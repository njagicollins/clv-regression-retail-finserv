"""
Advanced Feature Engineering Module for CLV Prediction

This module provides comprehensive feature engineering functions to create 
predictive features for customer lifetime value estimation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict


class FeatureEngineer:
    """
    Comprehensive feature engineering for CLV prediction.
    Includes interaction terms, ratio features, RFM-style metrics, and segmentation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer with raw data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw customer data with CLV as target
        """
        self.df = df.copy()
        self.engineered_features = []
        
    # ============ MONETARY VALUE FEATURES ============
    
    def create_monetary_features(self) -> pd.DataFrame:
        """
        Create features capturing spending intensity and value metrics.
        """
        df = self.df.copy()
        
        # Spending patterns
        df['spend_per_product'] = df['avg_monthly_spend'] / (df['num_products_owned'] + 1e-6)
        df['annual_spend'] = df['avg_monthly_spend'] * 12
        df['total_product_value'] = df['avg_monthly_spend'] * df['num_products_owned']
        
        # Transaction efficiency
        df['spend_per_transaction'] = df['avg_monthly_spend'] / (df['purchase_frequency'] + 1e-6)
        df['transaction_density'] = df['purchase_frequency'] / 12  # purchases per month
        
        # Discount sensitivity
        df['discount_impact'] = df['avg_monthly_spend'] * (df['discount_usage_rate'] / 100 + 1e-6)
        df['discounted_spend_ratio'] = df['discount_usage_rate'] / (df['avg_monthly_spend'] + 1e-6)
        
        # Value concentration
        df['product_concentration'] = df['num_products_owned'] / (df['num_products_owned'] + 1)  # 0-1 scale
        
        self.engineered_features.extend([
            'spend_per_product', 'annual_spend', 'total_product_value',
            'spend_per_transaction', 'transaction_density', 
            'discount_impact', 'discounted_spend_ratio', 'product_concentration'
        ])
        
        return df
    
    # ============ ENGAGEMENT & BEHAVIORAL FEATURES ============
    
    def create_engagement_features(self) -> pd.DataFrame:
        """
        Create features capturing customer engagement and behavior.
        """
        df = self.df.copy()
        
        # Engagement intensity
        df['engagement_x_spend'] = df['engagement_score'] * df['avg_monthly_spend']
        df['engagement_per_product'] = df['engagement_score'] / (df['num_products_owned'] + 1e-6)
        df['engagement_efficiency'] = df['engagement_score'] / (df['num_support_calls'] + 1)
        
        # Email engagement patterns
        df['email_engagement_x_discount'] = df['email_open_rate'] * df['discount_usage_rate']
        df['email_spend_alignment'] = df['email_open_rate'] * df['avg_monthly_spend'] / 10000  # scaled
        
        # Support needs vs engagement
        df['support_engagement_gap'] = df['engagement_score'] - (df['num_support_calls'] * 5)
        df['support_per_month'] = df['num_support_calls'] / (df['tenure_months'] + 1e-6)
        
        # Multi-channel behavior
        df['engagement_volatility'] = df['engagement_score'] * (1 - df['email_open_rate'] / 100)
        
        self.engineered_features.extend([
            'engagement_x_spend', 'engagement_per_product', 'engagement_efficiency',
            'email_engagement_x_discount', 'email_spend_alignment',
            'support_engagement_gap', 'support_per_month', 'engagement_volatility'
        ])
        
        return df
    
    # ============ RECENCY & TENURE FEATURES ============
    
    def create_recency_tenure_features(self) -> pd.DataFrame:
        """
        Create temporal features capturing customer lifecycle stage.
        """
        df = self.df.copy()
        
        # Recency buckets (if not already present)
        if 'recency_bucket' not in df.columns:
            df['recency_bucket'] = pd.cut(
                df['days_since_last_purchase'],
                bins=[-1, 30, 90, 180, 365, float('inf')],
                labels=['Very_Active', 'Active', 'Cooling_Off', 'At_Risk', 'Lost']
            )
        
        # Tenure stages
        df['tenure_years'] = df['tenure_months'] / 12
        df['tenure_stage'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 12, 24, 60, float('inf')],
            labels=['New', 'Emerging', 'Established', 'Loyal', 'VIP']
        )
        
        # Recency-tenure interaction
        df['recent_and_long_tenure'] = (
            (df['days_since_last_purchase'] <= 30).astype(int) * 
            (df['tenure_months'] >= 24).astype(int)
        )
        
        # Churn risk composite
        df['churn_risk_decay'] = df['churn_risk_score'] * (df['days_since_last_purchase'] / 365)
        
        self.engineered_features.extend([
            'tenure_years', 'tenure_stage', 'recency_bucket',
            'recent_and_long_tenure', 'churn_risk_decay'
        ])
        
        return df
    
    # ============ RFM-STYLE FEATURES ============
    
    def create_rfm_features(self) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) style composite features.
        """
        df = self.df.copy()
        
        # Recency score (inverse: lower days = higher score)
        df['recency_score'] = 100 - (df['days_since_last_purchase'] / df['days_since_last_purchase'].max() * 100)
        df['recency_score'] = df['recency_score'].clip(0, 100)
        
        # Frequency score (normalized purchase frequency)
        df['frequency_score'] = (df['purchase_frequency'] / df['purchase_frequency'].max() * 100)
        df['frequency_score'] = df['frequency_score'].clip(0, 100)
        
        # Monetary score (normalized average monthly spend)
        df['monetary_score'] = (df['avg_monthly_spend'] / df['avg_monthly_spend'].max() * 100)
        df['monetary_score'] = df['monetary_score'].clip(0, 100)
        
        # RFM composite score (weighted average)
        df['rfm_score'] = (
            0.3 * df['recency_score'] + 
            0.3 * df['frequency_score'] + 
            0.4 * df['monetary_score']
        )
        
        # RFM segmentation
        df['rfm_segment'] = pd.cut(
            df['rfm_score'],
            bins=[0, 33, 66, 100],
            labels=['Low_Value', 'Medium_Value', 'High_Value']
        )
        
        self.engineered_features.extend([
            'recency_score', 'frequency_score', 'monetary_score',
            'rfm_score', 'rfm_segment'
        ])
        
        return df
    
    # ============ CUSTOMER SEGMENT FEATURES ============
    
    def create_segment_features(self) -> pd.DataFrame:
        """
        Create features based on customer segmentation patterns.
        """
        df = self.df.copy()
        
        # Segment-based monetization
        if 'customer_segment' in df.columns:
            segment_spend = df.groupby('customer_segment')['avg_monthly_spend'].transform('mean')
            df['segment_spend_deviation'] = df['avg_monthly_spend'] - segment_spend
            df['segment_spend_percentile'] = df.groupby('customer_segment')['avg_monthly_spend'].transform('rank', pct=True)
            
            segment_tenure = df.groupby('customer_segment')['tenure_months'].transform('mean')
            df['segment_tenure_deviation'] = df['tenure_months'] - segment_tenure
            
            self.engineered_features.extend([
                'segment_spend_deviation', 'segment_spend_percentile', 'segment_tenure_deviation'
            ])
        
        # Age-based insights
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['Gen_Z', 'Millennial', 'Gen_X', 'Boomer', 'Silent']
            )
            df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
            
            self.engineered_features.extend(['age_group', 'age_normalized'])
        
        # Gender-based patterns (if available)
        if 'gender' in df.columns:
            gender_spend = df.groupby('gender')['avg_monthly_spend'].transform('mean')
            df['gender_spend_ratio'] = df['avg_monthly_spend'] / (gender_spend + 1e-6)
        
        return df
    
    # ============ POLYNOMIAL & INTERACTION FEATURES ============
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create meaningful interaction terms between key features.
        """
        df = self.df.copy()
        
        # High-value interactions
        df['high_engagement_high_spend'] = (
            ((df['engagement_score'] > df['engagement_score'].quantile(0.75)).astype(int)) *
            ((df['avg_monthly_spend'] > df['avg_monthly_spend'].quantile(0.75)).astype(int))
        )
        
        # Loyalty indicators
        df['tenure_engagement_product'] = (
            (df['tenure_months'] / 100) * 
            (df['engagement_score'] / 100) * 
            df['num_products_owned']
        )
        
        # Value stability
        df['consistent_value_buyer'] = (
            ((df['avg_transaction_value'] > df['avg_transaction_value'].quantile(0.5)).astype(int)) *
            ((df['purchase_frequency'] > df['purchase_frequency'].quantile(0.5)).astype(int))
        )
        
        # Churn-resistant customers
        df['at_risk_despite_engagement'] = (
            ((df['churn_risk_score'] > 50).astype(int)) *
            ((df['engagement_score'] > df['engagement_score'].quantile(0.75)).astype(int))
        )
        
        self.engineered_features.extend([
            'high_engagement_high_spend', 'tenure_engagement_product',
            'consistent_value_buyer', 'at_risk_despite_engagement'
        ])
        
        return df
    
    # ============ POLYNOMIAL FEATURES ============
    
    def create_polynomial_features(self, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for key numeric columns.
        
        Parameters:
        -----------
        degree : int
            Polynomial degree (default: 2)
        """
        df = self.df.copy()
        
        key_features = [
            'avg_monthly_spend', 'purchase_frequency', 
            'tenure_months', 'engagement_score'
        ]
        
        for col in key_features:
            if col in df.columns:
                if degree >= 2:
                    df[f'{col}_squared'] = df[col] ** 2
                if degree >= 3:
                    df[f'{col}_cubed'] = df[col] ** 3
                if degree >= 2:
                    df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        
        self.engineered_features.extend([
            col for col in df.columns if '_squared' in col or '_cubed' in col or '_sqrt' in col
        ])
        
        return df
    
    # ============ STATISTICAL FEATURES ============
    
    def create_statistical_features(self) -> pd.DataFrame:
        """
        Create statistical aggregate features.
        """
        df = self.df.copy()
        
        # Coefficient of variation (spend stability)
        df['spend_cv'] = df['avg_monthly_spend'] / (df['avg_monthly_spend'] + 1e-6)
        
        # Skewness indicators
        df['high_atv_customer'] = (df['avg_transaction_value'] > df['avg_transaction_value'].quantile(0.75)).astype(int)
        df['high_frequency_customer'] = (df['purchase_frequency'] > df['purchase_frequency'].quantile(0.75)).astype(int)
        
        # Combination patterns
        df['premium_customer_flag'] = (
            ((df['num_products_owned'] > 3).astype(int)) *
            ((df['engagement_score'] > df['engagement_score'].quantile(0.75)).astype(int)) *
            ((df['satisfaction_score'] > 3).astype(int)) if 'satisfaction_score' in df.columns else 0
        )
        
        self.engineered_features.extend([
            'spend_cv', 'high_atv_customer', 'high_frequency_customer', 'premium_customer_flag'
        ])
        
        return df
    
    # ============ GENERATE ALL FEATURES ============
    
    def generate_all_features(self) -> pd.DataFrame:
        """
        Generate all engineered features in sequence.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all engineered features
        """
        df = self.df.copy()
        
        # Apply all feature engineering methods
        df = self.create_monetary_features()
        df = self.create_engagement_features()
        df = self.create_recency_tenure_features()
        df = self.create_rfm_features()
        df = self.create_segment_features()
        df = self.create_interaction_features()
        df = self.create_polynomial_features(degree=2)
        df = self.create_statistical_features()
        
        # Update dataframe
        self.df = df
        
        return df
    
    def get_engineered_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        return self.engineered_features
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary of all engineered features.
        
        Returns:
        --------
        pd.DataFrame
            Summary with feature name, type, and count
        """
        summary_data = []
        for col in self.engineered_features:
            if col in self.df.columns:
                summary_data.append({
                    'Feature': col,
                    'Type': self.df[col].dtype,
                    'Non-Null': self.df[col].notna().sum(),
                    'Null': self.df[col].isna().sum(),
                    'Unique': self.df[col].nunique()
                })
        
        return pd.DataFrame(summary_data)


# ============ UTILITY FUNCTIONS ============

def scale_features(df: pd.DataFrame, feature_cols: List[str], 
                   method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        Columns to scale
    method : str
        'standard' or 'minmax'
    
    Returns:
    --------
    Tuple[pd.DataFrame, object]
        Scaled dataframe and scaler object
    """
    df_scaled = df.copy()
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_scaled, scaler


def get_feature_importance_ranking(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract and rank feature importance from trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ or coef_ attribute
    feature_names : List[str]
        Names of features
    
    Returns:
    --------
    pd.DataFrame
        Ranked features by importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model must have feature_importances_ or coef_ attribute")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return feature_importance
