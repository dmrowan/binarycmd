#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from . import plotutils

"""
StarHorse Training Set Query
-----------------------------
Run on Gaia Archive (https://gea.esac.esa.int/archive/)

SELECT
  sh.source_id,
  sh.dist50 AS dist,
  sh.mass50 AS mass,
  sh.teff50 AS teff,
  sh.logg50 AS logg,
  sh.met50 AS met,
  sh.av50 AS av,
  sh.ag50 AS ag,
  sh.abp50 AS abp,
  sh.arp50 AS arp,
  sh.sh_outflag,
  g.ra,
  g.dec,
  g.phot_g_mean_mag AS gmag,
  g.phot_bp_mean_mag AS bpmag,
  g.phot_rp_mean_mag AS rpmag,
  g.bp_rp,
  g.parallax,
  g.parallax_over_error,
  g.ruwe,
  g.phot_bp_rp_excess_factor,
  g.visibility_periods_used
FROM gaiaedr3_contrib.starhorse AS sh
JOIN gaiadr3.gaia_source AS g ON sh.source_id = g.source_id
WHERE
  sh.mass50 > 0.08
  AND sh.mass50 < 100
  AND sh.teff50 > 2500
  AND sh.teff50 < 50000
  AND sh.dist50 > 0
  AND sh.dist50 < 100000
  AND g.phot_g_mean_mag IS NOT NULL
  AND g.bp_rp IS NOT NULL
  AND g.parallax IS NOT NULL
  AND g.parallax_over_error >= 20
  AND g.ruwe < 1.4
  AND g.phot_bp_rp_excess_factor BETWEEN 1.0 AND 1.5
  AND g.duplicated_source = FALSE
  AND g.visibility_periods_used >= 8
  AND MOD(g.random_index, 10) = 0
"""

STARHORSE_TRAINING_PATH = '/Users/dmrowan/catalogs/sh_training_set.csv'


def load_starhorse_training_data(filepath=None):
    """
    Load StarHorse training data from CSV file

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Uses default if None.

    Returns
    -------
    df : pandas.DataFrame
        Training data with computed features
    """

    if filepath is None:
        filepath = STARHORSE_TRAINING_PATH

    df = pd.read_csv(filepath)
    df = compute_features(df)

    return df


def compute_features(df):
    """
    Compute derived features from raw StarHorse+Gaia data

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
        With additional feature columns
    """

    df = df.copy()

    # Absolute magnitude (uncorrected)
    # Distance is in kpc, so M = m - 5*log10(d_kpc) - 10
    dist_col = 'dist' if 'dist' in df.columns else 'distance'
    if 'gmag' in df.columns and dist_col in df.columns:
        df['absolute_g'] = df.gmag - 5*np.log10(df[dist_col]) - 10 - df.ag

    # Extinction-corrected color
    if 'bp_rp' in df.columns and 'abp' in df.columns and 'arp' in df.columns:
        df['bp_rp_corrected'] = df.bp_rp - df.abp + df.arp

    return df


def plot_training_cmd(df=None, n_sample=50000, savefig=None):
    """
    Plot CMD of training set colored by mass and Teff

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Training data. If None, loads default.
    n_sample : int
        Number of stars to plot (subsample for speed)
    savefig : str, optional
        Path to save figure

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """

    if df is None:
        df = load_starhorse_training_data()

    np.random.seed(42)
    n_sample = min(n_sample, len(df))
    idx = np.random.choice(len(df), size=n_sample, replace=False)
    df_plot = df.iloc[idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Mass-colored CMD
    ax = axes[0]
    sc1 = ax.scatter(df_plot['bp_rp_corrected'], df_plot['absolute_g'],
                    c=df_plot['mass'], cmap='viridis', s=1, alpha=0.5,
                    vmin=0.08, vmax=2.0)
    ax.invert_yaxis()
    ax.set_xlabel('BP - RP (extinction corrected)', fontsize=16)
    ax.set_ylabel('Absolute G (extinction corrected)', fontsize=16)
    ax.set_title('Colored by Mass', fontsize=18)
    cbar1 = plt.colorbar(sc1, ax=ax, label=r'Mass [M$_{\odot}$]')
    cbar1.set_label(r'Mass [M$_{\odot}$]', fontsize=16)
    ax = plotutils.plotparams(ax)
    cbar1 = plotutils.plotparams_cbar(cbar1)

    # Teff-colored CMD
    ax = axes[1]
    sc2 = ax.scatter(df_plot['bp_rp_corrected'], df_plot['absolute_g'],
                    c=df_plot['teff'], cmap='coolwarm', s=1, alpha=0.5,
                    vmin=3000, vmax=7000)
    ax.invert_yaxis()
    ax.set_xlabel('BP - RP (extinction corrected)', fontsize=16)
    ax.set_ylabel('Absolute G (extinction corrected)', fontsize=16)
    ax.set_title('Colored by Teff', fontsize=18)
    cbar2 = plt.colorbar(sc2, ax=ax, label='Teff [K]')
    cbar2.set_label('Teff [K]', fontsize=16)
    ax = plotutils.plotparams(ax)
    cbar2 = plotutils.plotparams_cbar(cbar2)

    plt.suptitle(f'StarHorse Training Set CMD (N={len(df):,}, showing {len(df_plot):,})',
                 fontsize=20, y=1.02)
    plt.tight_layout()

    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, axes


class KNNStellarParams:
    """
    KNN-based stellar parameter inference from CMD position
    """

    def __init__(self, target_param='mass', n_neighbors=10,
                 features=['absolute_g', 'bp_rp_corrected'], test_size=0.2, random_state=42):
        """
        Parameters
        ----------
        target_param : str
            Parameter to predict (mass, teff, logg, etc.)
        n_neighbors : int
            Number of neighbors for KNN
        features : list
            Feature columns to use
        test_size : float
            Fraction of data to use for test set
        random_state : int
            Random seed for reproducibility
        """
        self.target_param = target_param
        self.n_neighbors = n_neighbors
        self.features = features
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        self.scaler = StandardScaler()
        self.trained = False

        # Load and split training data
        full_data = load_starhorse_training_data()
        self.train_data, self.test_data = train_test_split(
            full_data, test_size=test_size, random_state=random_state
        )

    def fit(self):
        """
        Train the model on the training portion of the dataset
        """
        X = self.train_data[self.features].values
        y = self.train_data[self.target_param].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        # Standardize features for KNN
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.trained = True

        print(f"Trained KNN model for {self.target_param}")
        print(f"  Features: {self.features}")
        print(f"  Training samples: {len(X):,}")

    def predict(self, absolute_g=None, bp_rp_corrected=None,
                X=None, return_uncertainty=False):
        """
        Predict parameter for new stars

        Parameters
        ----------
        absolute_g : float or array
            Extinction-corrected absolute G
        bp_rp_corrected : float or array
            Extinction-corrected BP-RP color
        X : array, optional
            Feature array (if not using absolute_g, bp_rp_corrected)
        return_uncertainty : bool
            Return uncertainty estimate

        Returns
        -------
        predictions : array
            Predicted values
        uncertainties : array (if return_uncertainty=True)
            Uncertainty estimates (std dev of K neighbors)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        if X is None:
            if absolute_g is None or bp_rp_corrected is None:
                raise ValueError("Must provide either X or (absolute_g, bp_rp_corrected)")
            X = np.column_stack([absolute_g, bp_rp_corrected])

        X = np.atleast_2d(X)

        # Standardize features using the fitted scaler
        X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        if return_uncertainty:
            distances, indices = self.model.kneighbors(X)
            neighbor_values = self.model._y[indices]
            uncertainties = np.std(neighbor_values, axis=1)
            return predictions, uncertainties

        return predictions

    def cross_validate(self, cv=5, plot=False, savefig=None):
        """
        Evaluate model performance with cross-validation on the training set

        Parameters
        ----------
        cv : int
            Number of CV folds
        plot : bool
            Whether to create CV plots
        savefig : str, optional
            Path to save figure

        Returns
        -------
        scores : dict
            CV scores
        """
        X = self.train_data[self.features].values
        y = self.train_data[self.target_param].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        # Standardize features for cross-validation
        X = self.scaler.fit_transform(X)

        r2_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        neg_mae_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        mae_scores = -neg_mae_scores

        scores = {
            'r2_mean': r2_scores.mean(), #coefficient of determination
            'r2_std': r2_scores.std(), #lower value, more consistent across different folds
            'mae_mean': mae_scores.mean(), #mean absolute error
            'mae_std': mae_scores.std() #lower value, more consistent across different folds
        }

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # R² scores across folds
            ax = axes[0]
            folds = np.arange(1, len(r2_scores) + 1)
            ax.bar(folds, r2_scores, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axhline(y=r2_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {r2_scores.mean():.4f}')
            ax.fill_between(folds,
                           r2_scores.mean() - r2_scores.std(),
                           r2_scores.mean() + r2_scores.std(),
                           alpha=0.2, color='r', label=f'±1 Std Dev: {r2_scores.std():.4f}')
            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title('R² Across Folds', fontsize=14)
            ax.set_xticks(folds)
            ax.set_ylim([max(0, r2_scores.min() - 0.1), min(1, r2_scores.max() + 0.1)])
            ax.grid(alpha=0.3, axis='y')
            ax.legend()
            ax = plotutils.plotparams(ax)

            # MAE scores across folds
            ax = axes[1]
            ax.bar(folds, mae_scores, alpha=0.7, edgecolor='black', color='coral')
            ax.axhline(y=mae_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {mae_scores.mean():.4f}')
            ax.fill_between(folds,
                           mae_scores.mean() - mae_scores.std(),
                           mae_scores.mean() + mae_scores.std(),
                           alpha=0.2, color='r', label=f'±1 Std Dev: {mae_scores.std():.4f}')
            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel('MAE', fontsize=12)
            ax.set_title('MAE Across Folds', fontsize=14)
            ax.set_xticks(folds)
            ax.grid(alpha=0.3, axis='y')
            ax.legend()
            ax = plotutils.plotparams(ax)

            plt.suptitle(f'{cv}-Fold Cross-Validation Results', fontsize=16, y=1.02)
            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')

        return scores

    def _compute_test_scores(self):
        """
        Compute predictions and residuals on test set

        Returns
        -------
        dict with keys: 'X_unscaled', 'y', 'predictions', 'residuals', 'r2', 'mae', 'scores'
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        X_unscaled = self.test_data[self.features].values
        y = self.test_data[self.target_param].values

        mask = np.isfinite(X_unscaled).all(axis=1) & np.isfinite(y)
        X_unscaled = X_unscaled[mask]
        y = y[mask]

        # Standardize features for prediction
        X_scaled = self.scaler.transform(X_unscaled)
        predictions = self.model.predict(X_scaled)
        r2 = self.model.score(X_scaled, y)
        mae = np.mean(np.abs(predictions - y))
        residuals = y - predictions

        return {
            'X_unscaled': X_unscaled,
            'y': y,
            'predictions': predictions,
            'residuals': residuals,
            'r2': r2,
            'mae': mae,
            'scores': {'r2': r2, 'mae': mae}
        }

    def plot_test_evaluation(self, savefig=None):
        """
        Plot 3-panel evaluation: predicted vs actual, residuals vs actual, residuals histogram

        Parameters
        ----------
        savefig : str, optional
            Path to save figure
        """
        results = self._compute_test_scores()
        y = results['y']
        predictions = results['predictions']
        residuals = results['residuals']
        r2 = results['r2']
        mae = results['mae']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Predicted vs Actual
        ax = axes[0]
        ax.scatter(y, predictions, alpha=0.5, s=10)
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
        ax.set_ylabel(f'Predicted {self.target_param}', fontsize=12)
        ax.set_title('Predicted vs Actual', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax = plotutils.plotparams(ax)

        # Residuals vs Actual
        ax = axes[1]
        ax.scatter(y, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals vs Actual', fontsize=14)
        ax.grid(alpha=0.3)
        ax = plotutils.plotparams(ax)

        # Residuals histogram
        ax = axes[2]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.axvline(x=np.median(residuals)-3*np.std(residuals), color='r', linestyle=':', lw=2)
        ax.axvline(x=np.median(residuals)+3*np.std(residuals), color='r', linestyle=':', lw=2)
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14)
        ax.grid(alpha=0.3, axis='y')
        ax = plotutils.plotparams(ax)
        ax.set_yscale('log')

        plt.suptitle(f'Test Set Evaluation (R²={r2:.4f}, MAE={mae:.4f})',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')

    def plot_cmd_residuals(self, threshold=None, savefig=None):
        """
        Plot CMD colored by residual magnitude with optional threshold highlighting

        Parameters
        ----------
        threshold : float, optional
            Highlight residuals with magnitude greater than this value
        savefig : str, optional
            Path to save figure
        """
        results = self._compute_test_scores()
        X_unscaled = results['X_unscaled']
        residuals = results['residuals']

        fig, ax = plt.subplots(figsize=(10, 8))

        abs_residuals = np.abs(residuals)

        # Scatter plot colored by residual magnitude
        sc = ax.scatter(X_unscaled[:, 1], X_unscaled[:, 0],
                       c=abs_residuals, cmap='RdYlBu_r', s=20, alpha=0.6,
                       vmin=0.1, vmax=threshold*1.5)
        ax.invert_yaxis()

        # Highlight large residuals if threshold provided
        if threshold is not None:
            large_residuals = abs_residuals > threshold
            ax.scatter(X_unscaled[large_residuals, 1], X_unscaled[large_residuals, 0],
                      s=100, facecolors='none', edgecolors='black', linewidths=2,
                      label=r'abs(residual) $>{}$'.format(threshold))
            ax.legend(fontsize=12)

        ax.set_xlabel('BP - RP (extinction corrected)', fontsize=12)
        ax.set_ylabel('Absolute G (extinction corrected)', fontsize=12)
        ax.set_title(f'CMD colored by |residuals| in {self.target_param}', fontsize=14)

        cbar = plt.colorbar(sc, ax=ax, label=f'abs(Residuals) [{self.target_param}]')
        cbar = plotutils.plotparams_cbar(cbar)
        ax = plotutils.plotparams(ax)

        plt.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')

    def evaluate_test_set(self, plot=False, cmd_plot=False, threshold=None, savefig=None):
        """
        Evaluate model performance on the test set

        Parameters
        ----------
        plot : bool
            Whether to create standard evaluation plots (pred vs actual, residuals, histogram)
        cmd_plot : bool
            Whether to create CMD residual plot
        threshold : float, optional
            Threshold for highlighting large residuals in CMD plot
        savefig : str, optional
            Path to save figure

        Returns
        -------
        scores : dict
            R² and MAE scores on test set
        """
        results = self._compute_test_scores()
        scores = results['scores']

        if plot:
            self.plot_test_evaluation(savefig=savefig)

        if cmd_plot:
            self.plot_cmd_residuals(threshold=threshold, savefig=savefig)

        return scores

        if plot:
            residuals = y - predictions
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Predicted vs Actual
            ax = axes[0]
            ax.scatter(y, predictions, alpha=0.5, s=10)
            min_val = min(y.min(), predictions.min())
            max_val = max(y.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
            ax.set_ylabel(f'Predicted {self.target_param}', fontsize=12)
            ax.set_title('Predicted vs Actual', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            ax = plotutils.plotparams(ax)

            # Residuals vs Actual
            ax = axes[1]
            ax.scatter(y, residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
            ax.set_ylabel('Residuals', fontsize=12)
            ax.set_title('Residuals vs Actual', fontsize=14)
            ax.grid(alpha=0.3)
            ax = plotutils.plotparams(ax)

            # Residuals histogram
            ax = axes[2]
            ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.axvline(x=np.median(residuals)-3*np.std(residuals), color='r', linestyle=':', lw=2)
            ax.axvline(x=np.median(residuals)+3*np.std(residuals), color='r', linestyle=':', lw=2)
            ax.set_xlabel('Residuals', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Residuals', fontsize=14)
            ax.grid(alpha=0.3, axis='y')
            ax = plotutils.plotparams(ax)
            ax.set_yscale('log')

            plt.suptitle(f'Test Set Evaluation (R²={r2:.4f}, MAE={mae:.4f})',
                         fontsize=16, y=1.02)
            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')

        return scores

    def save_model(self, filepath):
        """
        Save trained model

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        """
        Load trained model

        Parameters
        ----------
        filepath : str
            Path to saved model

        Returns
        -------
        model : KNNStellarParams
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


class RFStellarParams:
    """
    Random Forest-based stellar parameter inference from CMD position
    """

    def __init__(self, target_param='mass', n_estimators=100,
                 features=['absolute_g', 'bp_rp_corrected'], test_size=0.2, random_state=42):
        """
        Parameters
        ----------
        target_param : str
            Parameter to predict (mass, teff, logg, etc.)
        n_estimators : int
            Number of trees in the forest
        features : list
            Feature columns to use
        test_size : float
            Fraction of data to use for test set
        random_state : int
            Random seed for reproducibility
        """
        self.target_param = target_param
        self.n_estimators = n_estimators
        self.features = features
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.trained = False

        # Load and split training data
        full_data = load_starhorse_training_data()
        self.train_data, self.test_data = train_test_split(
            full_data, test_size=test_size, random_state=random_state
        )

    def fit(self):
        """
        Train the model on the training portion of the dataset
        """
        X = self.train_data[self.features].values
        y = self.train_data[self.target_param].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        self.model.fit(X, y)
        self.trained = True

        print(f"Trained Random Forest model for {self.target_param}")
        print(f"  Features: {self.features}")
        print(f"  Training samples: {len(X):,}")
        print(f"  Feature importances: {dict(zip(self.features, self.model.feature_importances_.round(4)))}")

    def predict(self, absolute_g=None, bp_rp_corrected=None,
                X=None, return_uncertainty=False):
        """
        Predict parameter for new stars

        Parameters
        ----------
        absolute_g : float or array
            Extinction-corrected absolute G
        bp_rp_corrected : float or array
            Extinction-corrected BP-RP color
        X : array, optional
            Feature array (if not using absolute_g, bp_rp_corrected)
        return_uncertainty : bool
            Return uncertainty estimate

        Returns
        -------
        predictions : array
            Predicted values
        uncertainties : array (if return_uncertainty=True)
            Uncertainty estimates (std dev across trees)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        if X is None:
            if absolute_g is None or bp_rp_corrected is None:
                raise ValueError("Must provide either X or (absolute_g, bp_rp_corrected)")
            X = np.column_stack([absolute_g, bp_rp_corrected])

        X = np.atleast_2d(X)

        predictions = self.model.predict(X)

        if return_uncertainty:
            # Get predictions from each tree
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            uncertainties = np.std(tree_predictions, axis=0)
            return predictions, uncertainties

        return predictions

    def cross_validate(self, cv=5, plot=False, savefig=None):
        """
        Evaluate model performance with cross-validation on the training set

        Parameters
        ----------
        cv : int
            Number of CV folds
        plot : bool
            Whether to create CV plots
        savefig : str, optional
            Path to save figure

        Returns
        -------
        scores : dict
            CV scores
        """
        X = self.train_data[self.features].values
        y = self.train_data[self.target_param].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        r2_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        neg_mae_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        mae_scores = -neg_mae_scores

        scores = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # R² scores across folds
            ax = axes[0]
            folds = np.arange(1, len(r2_scores) + 1)
            ax.bar(folds, r2_scores, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axhline(y=r2_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {r2_scores.mean():.4f}')
            ax.fill_between(folds,
                           r2_scores.mean() - r2_scores.std(),
                           r2_scores.mean() + r2_scores.std(),
                           alpha=0.2, color='r', label=f'±1 Std Dev: {r2_scores.std():.4f}')
            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title('R² Across Folds', fontsize=14)
            ax.set_xticks(folds)
            ax.set_ylim([max(0, r2_scores.min() - 0.1), min(1, r2_scores.max() + 0.1)])
            ax.grid(alpha=0.3, axis='y')
            ax.legend()
            ax = plotutils.plotparams(ax)

            # MAE scores across folds
            ax = axes[1]
            ax.bar(folds, mae_scores, alpha=0.7, edgecolor='black', color='coral')
            ax.axhline(y=mae_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {mae_scores.mean():.4f}')
            ax.fill_between(folds,
                           mae_scores.mean() - mae_scores.std(),
                           mae_scores.mean() + mae_scores.std(),
                           alpha=0.2, color='r', label=f'±1 Std Dev: {mae_scores.std():.4f}')
            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel('MAE', fontsize=12)
            ax.set_title('MAE Across Folds', fontsize=14)
            ax.set_xticks(folds)
            ax.grid(alpha=0.3, axis='y')
            ax.legend()
            ax = plotutils.plotparams(ax)

            plt.suptitle(f'{cv}-Fold Cross-Validation Results', fontsize=16, y=1.02)
            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')

        return scores

    def _compute_test_scores(self):
        """
        Compute predictions and residuals on test set

        Returns
        -------
        dict with keys: 'X_unscaled', 'y', 'predictions', 'residuals', 'r2', 'mae', 'scores'
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        X_unscaled = self.test_data[self.features].values
        y = self.test_data[self.target_param].values

        mask = np.isfinite(X_unscaled).all(axis=1) & np.isfinite(y)
        X_unscaled = X_unscaled[mask]
        y = y[mask]

        predictions = self.model.predict(X_unscaled)
        r2 = self.model.score(X_unscaled, y)
        mae = np.mean(np.abs(predictions - y))
        residuals = y - predictions

        return {
            'X_unscaled': X_unscaled,
            'y': y,
            'predictions': predictions,
            'residuals': residuals,
            'r2': r2,
            'mae': mae,
            'scores': {'r2': r2, 'mae': mae}
        }

    def plot_test_evaluation(self, savefig=None):
        """
        Plot 3-panel evaluation: predicted vs actual, residuals vs actual, residuals histogram

        Parameters
        ----------
        savefig : str, optional
            Path to save figure
        """
        results = self._compute_test_scores()
        y = results['y']
        predictions = results['predictions']
        residuals = results['residuals']
        r2 = results['r2']
        mae = results['mae']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Predicted vs Actual
        ax = axes[0]
        ax.scatter(y, predictions, alpha=0.5, s=10)
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
        ax.set_ylabel(f'Predicted {self.target_param}', fontsize=12)
        ax.set_title('Predicted vs Actual', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax = plotutils.plotparams(ax)

        # Residuals vs Actual
        ax = axes[1]
        ax.scatter(y, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel(f'Actual {self.target_param}', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals vs Actual', fontsize=14)
        ax.grid(alpha=0.3)
        ax = plotutils.plotparams(ax)

        # Residuals histogram
        ax = axes[2]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14)
        ax.grid(alpha=0.3, axis='y')
        ax = plotutils.plotparams(ax)

        plt.suptitle(f'Test Set Evaluation (R²={r2:.4f}, MAE={mae:.4f})',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')

    def plot_cmd_residuals(self, threshold=None, savefig=None):
        """
        Plot CMD colored by residual magnitude with optional threshold highlighting

        Parameters
        ----------
        threshold : float, optional
            Highlight residuals with magnitude greater than this value
        savefig : str, optional
            Path to save figure
        """
        results = self._compute_test_scores()
        X_unscaled = results['X_unscaled']
        residuals = results['residuals']

        fig, ax = plt.subplots(figsize=(10, 8))

        abs_residuals = np.abs(residuals)

        # Scatter plot colored by residual magnitude
        sc = ax.scatter(X_unscaled[:, 1], X_unscaled[:, 0],
                       c=abs_residuals, cmap='RdYlBu_r', s=20, alpha=0.6)
        ax.invert_yaxis()

        # Highlight large residuals if threshold provided
        if threshold is not None:
            large_residuals = abs_residuals > threshold
            ax.scatter(X_unscaled[large_residuals, 1], X_unscaled[large_residuals, 0],
                      s=100, facecolors='none', edgecolors='black', linewidths=2,
                      label=f'|residual| > {threshold}')
            ax.legend(fontsize=12)

        ax.set_xlabel('BP - RP (extinction corrected)', fontsize=12)
        ax.set_ylabel('Absolute G (extinction corrected)', fontsize=12)
        ax.set_title(f'CMD colored by |residuals| in {self.target_param}', fontsize=14)

        cbar = plt.colorbar(sc, ax=ax, label=f'|Residuals| [{self.target_param}]')
        cbar = plotutils.plotparams_cbar(cbar)
        ax = plotutils.plotparams(ax)

        plt.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')

    def evaluate_test_set(self, plot=False, cmd_plot=False, threshold=None, savefig=None):
        """
        Evaluate model performance on the test set

        Parameters
        ----------
        plot : bool
            Whether to create standard evaluation plots (pred vs actual, residuals, histogram)
        cmd_plot : bool
            Whether to create CMD residual plot
        threshold : float, optional
            Threshold for highlighting large residuals in CMD plot
        savefig : str, optional
            Path to save figure

        Returns
        -------
        scores : dict
            R² and MAE scores on test set
        """
        results = self._compute_test_scores()
        scores = results['scores']

        if plot:
            self.plot_test_evaluation(savefig=savefig)

        if cmd_plot:
            self.plot_cmd_residuals(threshold=threshold, savefig=savefig)

        return scores

    def save_model(self, filepath):
        """
        Save trained model

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        """
        Load trained model

        Parameters
        ----------
        filepath : str
            Path to saved model

        Returns
        -------
        model : RFStellarParams
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
