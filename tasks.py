"""
Law tasks for CATHODE reproduction project.

This file contains all workflow tasks for reproducing the CATHODE paper results.
"""

import law
import luigi


# Mixin classes for parameter factorization
class DataMixin(law.Task):
    """Mixin for data-related parameters."""

    signal_url = luigi.Parameter(
        default="https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5",
        description="URL for signal dataset",
    )
    background_url = luigi.Parameter(
        default="https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5",
        description="URL for background dataset",
    )
    data_dir = luigi.Parameter(default="data", description="Directory for data files")


class RegionMixin(law.Task):
    """Mixin for signal/sideband region parameters."""

    sr_low = luigi.FloatParameter(default=3.3, description="SR lower bound in TeV")
    sr_high = luigi.FloatParameter(default=3.7, description="SR upper bound in TeV")
    mjj_low = luigi.FloatParameter(default=1.5, description="mJJ lower bound in TeV")
    mjj_high = luigi.FloatParameter(default=5.5, description="mJJ upper bound in TeV")


# Stage 2: Data Understanding Tasks


class DownloadData(DataMixin, law.Task):
    """
    Task 2.1: Download LHC Olympics R&D datasets from Zenodo.

    Downloads both background and signal datasets to the data directory.
    """

    force = luigi.BoolParameter(default=False, description="Force re-download even if files exist")

    def output(self):
        # Return a marker file that indicates download is complete
        return law.LocalFileTarget(f"{self.data_dir}/.download_complete")

    def run(self):
        from src.data.loader import download_file

        # Download background dataset
        bg_path = f"{self.data_dir}/background.h5"
        download_file(self.background_url, bg_path, force=self.force)

        # Download signal dataset
        sig_path = f"{self.data_dir}/signal.h5"
        download_file(self.signal_url, sig_path, force=self.force)

        # Create marker file
        self.output().parent.touch()
        with self.output().open("w") as f:
            f.write(f"Downloaded:\n{bg_path}\n{sig_path}\n")


class LoadAndValidateData(DataMixin, RegionMixin, law.Task):
    """
    Task 2.1 (continued): Load and validate the downloaded datasets.

    Loads dataset with label column (0=background, 1=signal), prints summary statistics,
    and validates event counts.
    """

    def requires(self):
        return DownloadData.req(self)

    def output(self):
        return law.LocalFileTarget(f"{self.data_dir}/data_validation.txt")

    def run(self):
        import numpy as np
        from src.data.loader import load_data, apply_region_cut

        # Load dataset (contains both background and signal with label column)
        data_path = f"{self.data_dir}/background.h5"
        bg_data, sig_data = load_data(data_path)

        # Validate and print statistics
        stats = []
        stats.append("=" * 80)
        stats.append("DATA VALIDATION SUMMARY")
        stats.append("=" * 80)
        stats.append("")

        # Dataset statistics
        stats.append(f"Dataset: {data_path}")
        stats.append(f"  Total background events (label=0): {len(bg_data['mJJ']):,}")
        stats.append(f"  Total signal events (label=1): {len(sig_data['mJJ']):,}")

        # Apply region cuts
        bg_sr_mask = apply_region_cut(
            bg_data,
            self.mjj_low,
            self.mjj_high,
            self.sr_low,
            self.sr_high,
            signal_region=True,
        )
        bg_sb_mask = apply_region_cut(
            bg_data,
            self.mjj_low,
            self.mjj_high,
            self.sr_low,
            self.sr_high,
            signal_region=False,
        )

        stats.append(
            f"  Background events in SR [{self.sr_low}, {self.sr_high}] TeV: {bg_sr_mask.sum():,}"
        )
        stats.append(f"  Background events in SB: {bg_sb_mask.sum():,}")
        stats.append("")

        # Signal region statistics
        sig_sr_mask = apply_region_cut(
            sig_data,
            self.mjj_low,
            self.mjj_high,
            self.sr_low,
            self.sr_high,
            signal_region=True,
        )
        stats.append(f"  Signal events in SR: {sig_sr_mask.sum():,}")

        # Compute S/B ratio
        s_over_b = sig_sr_mask.sum() / bg_sr_mask.sum() * 100 if bg_sr_mask.sum() > 0 else 0
        stats.append(f"  S/B ratio in SR: {s_over_b:.3f}%")
        stats.append("")

        # Feature ranges
        stats.append("Background Feature Ranges (full dataset):")
        for key in ["mJJ", "mJ1", "mJ2", "delta_mJ", "tau21_J1", "tau21_J2"]:
            vals = bg_data[key]
            stats.append(
                f"  {key:12s}: [{vals.min():.3f}, {vals.max():.3f}] "
                f"(mean={vals.mean():.3f}, std={vals.std():.3f})"
            )
        stats.append("")

        # Expected benchmark comparison (from paper)
        stats.append("Comparison with Paper Benchmark (S/B = 0.6%):")
        stats.append("  Paper: 1M background total, 121,352 in SR, 1k signal, 772 in SR")
        stats.append(
            f"  Our data: {len(bg_data['mJJ']):,} background, {len(sig_data['mJJ']):,} signal total"
        )
        stats.append(
            f"           {bg_sr_mask.sum():,} background in SR, {sig_sr_mask.sum():,} signal in SR"
        )
        stats.append(f"           S/B = {s_over_b:.3f}%")
        stats.append("")

        stats_text = "\n".join(stats)
        print(stats_text)

        # Write to output file
        self.output().parent.touch()
        with self.output().open("w") as f:
            f.write(stats_text)


class VisualizeFeatures(DataMixin, RegionMixin, law.Task):
    """
    Task 2.2: Visualize feature distributions.

    Creates plots comparing background and signal feature distributions,
    similar to Figures 3 and 4 in the CATHODE paper.
    """

    def requires(self):
        return LoadAndValidateData.req(self)

    def output(self):
        return {
            "features": law.LocalFileTarget("results/plots/feature_distributions.png"),
            "mjj": law.LocalFileTarget("results/plots/mjj_distribution.png"),
        }

    def run(self):
        from src.data.loader import load_data, apply_region_cut
        from src.utils.plotting import plot_feature_distributions, plot_mjj_distribution

        # Load dataset (contains both background and signal with label column)
        data_path = f"{self.data_dir}/background.h5"
        bg_data, sig_data = load_data(data_path)

        # Apply region cuts to get SR data
        bg_sr_mask = apply_region_cut(
            bg_data, self.mjj_low, self.mjj_high, self.sr_low, self.sr_high, True
        )
        sig_sr_mask = apply_region_cut(
            sig_data, self.mjj_low, self.mjj_high, self.sr_low, self.sr_high, True
        )

        # Extract SR data
        bg_sr = {key: val[bg_sr_mask] for key, val in bg_data.items()}
        sig_sr = {key: val[sig_sr_mask] for key, val in sig_data.items()}

        print(f"\nCreating feature distribution plots...")
        print(f"Background SR: {len(bg_sr['mJJ']):,} events")
        print(f"Signal SR: {len(sig_sr['mJJ']):,} events")

        # Plot feature distributions (SR only)
        features = ["mJ1", "delta_mJ", "tau21_J1", "tau21_J2", "mJJ"]
        xlabels = {
            "mJ1": r"$m_{J1}$ [GeV]",
            "delta_mJ": r"$\Delta m_J$ [GeV]",
            "tau21_J1": r"$\tau_{21}^{J1}$",
            "tau21_J2": r"$\tau_{21}^{J2}$",
            "mJJ": r"$m_{JJ}$ [TeV]",
        }

        self.output()["features"].parent.touch()
        plot_feature_distributions(
            bg_sr, sig_sr, features, self.output()["features"].path, xlabels=xlabels
        )

        # Plot mJJ distribution (full range)
        self.output()["mjj"].parent.touch()
        plot_mjj_distribution(
            bg_data, sig_data, self.output()["mjj"].path, self.sr_low, self.sr_high
        )


class PreprocessFeatures(DataMixin, RegionMixin, law.Task):
    """
    Task 2.4: Implement feature preprocessing.

    Fits a feature scaler on sideband background data (CATHODE requirement:
    learn normalization from background only, not signal-contaminated SR).
    """

    def requires(self):
        return LoadAndValidateData.req(self)

    def output(self):
        return law.LocalFileTarget("results/models/feature_scaler.pkl")

    def run(self):
        from src.data.loader import load_data, apply_region_cut, get_features
        from src.data.preprocessing import FeatureScaler, preprocess_data

        # Load dataset
        data_path = f"{self.data_dir}/background.h5"
        bg_data, sig_data = load_data(data_path)

        # Get sideband background data for fitting scaler
        # IMPORTANT: Fit only on background in sideband to avoid signal contamination
        bg_sb_mask = apply_region_cut(
            bg_data, self.mjj_low, self.mjj_high, self.sr_low, self.sr_high, False
        )

        # Extract SB background data
        bg_sb = {key: val[bg_sb_mask] for key, val in bg_data.items()}

        # Define features (without mJJ for now - mass is handled separately)
        feature_names = ["mJ1", "delta_mJ", "tau21_J1", "tau21_J2"]

        print(f"\nFitting feature scaler on sideband background...")
        print(f"  Sideband background events: {len(bg_sb['mJJ']):,}")
        print(f"  Features: {feature_names}")

        # Fit scaler on SB background
        X_sb = get_features(bg_sb, include_mass=False)
        scaler = FeatureScaler()
        scaler.fit(X_sb, feature_names)

        # Print scaling parameters
        print(f"\nScaling parameters:")
        for i, name in enumerate(feature_names):
            print(f"  {name:12s}: mean={scaler.mean_[i]:8.3f}, std={scaler.std_[i]:8.3f}")

        # Save scaler
        self.output().parent.touch()
        scaler.save(self.output().path)
        print(f"\nSaved scaler to {self.output().path}")


class VisualizePreprocessing(DataMixin, RegionMixin, law.Task):
    """
    Verify feature preprocessing by visualizing before/after standardization.

    Creates plots showing original and standardized feature distributions
    to verify that features are properly normalized (mean≈0, std≈1).
    """

    def requires(self):
        return PreprocessFeatures.req(self)

    def output(self):
        return law.LocalFileTarget("results/plots/preprocessing_verification.png")

    def run(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from src.data.loader import load_data, apply_region_cut, get_features
        from src.data.preprocessing import FeatureScaler

        # Load scaler
        scaler = FeatureScaler.load(self.requires().output().path)
        feature_names = scaler.feature_names_

        # Load dataset
        data_path = f"{self.data_dir}/background.h5"
        bg_data, sig_data = load_data(data_path)

        # Get sideband background data
        bg_sb_mask = apply_region_cut(
            bg_data, self.mjj_low, self.mjj_high, self.sr_low, self.sr_high, False
        )
        bg_sb = {key: val[bg_sb_mask] for key, val in bg_data.items()}

        # Extract features
        X_original = get_features(bg_sb, include_mass=False)
        X_scaled = scaler.transform(X_original)

        # Create comparison plots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, name in enumerate(feature_names):
            # Original distribution
            ax = axes[i]
            ax.hist(X_original[:, i], bins=50, alpha=0.7, color="blue", density=True)
            ax.set_xlabel(f"{name} (original)")
            ax.set_ylabel("Density")
            ax.set_title(f"Mean={X_original[:, i].mean():.2f}, Std={X_original[:, i].std():.2f}")
            ax.grid(True, alpha=0.3)

            # Standardized distribution
            ax = axes[i + 4]
            ax.hist(X_scaled[:, i], bins=50, alpha=0.7, color="green", density=True)
            ax.set_xlabel(f"{name} (standardized)")
            ax.set_ylabel("Density")
            ax.set_title(f"Mean={X_scaled[:, i].mean():.2f}, Std={X_scaled[:, i].std():.2f}")
            ax.grid(True, alpha=0.3)

            # Add reference normal distribution
            x_range = np.linspace(-4, 4, 100)
            ax.plot(
                x_range,
                np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi),
                "r--",
                alpha=0.5,
                label="N(0,1)",
            )
            ax.legend()

        plt.tight_layout()
        self.output().parent.touch()
        plt.savefig(self.output().path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved preprocessing verification plot to {self.output().path}")

        # Print statistics
        print("\nPreprocessing verification:")
        print("Original features (SB background):")
        for i, name in enumerate(feature_names):
            print(
                f"  {name:12s}: mean={X_original[:, i].mean():8.3f}, std={X_original[:, i].std():8.3f}"
            )
        print("\nStandardized features (should be mean≈0, std≈1):")
        for i, name in enumerate(feature_names):
            print(
                f"  {name:12s}: mean={X_scaled[:, i].mean():8.3f}, std={X_scaled[:, i].std():8.3f}"
            )


# Stage 3: CATHODE Implementation Tasks


class TrainOuterDensityEstimator(DataMixin, RegionMixin, law.Task):
    """
    Task 3.3: Train the CATHODE outer density estimator on sideband data.

    Trains a conditional MAF to learn p(x|mJJ) on sideband background events.
    Uses 500k training samples and ~379k validation samples.
    Implements model selection by keeping 10 best checkpoints.
    """

    n_train = luigi.IntParameter(default=500000, description="Number of training samples")
    n_val = luigi.IntParameter(default=379000, description="Number of validation samples")
    epochs = luigi.IntParameter(default=100, description="Number of training epochs")
    n_checkpoints = luigi.IntParameter(default=10, description="Number of best models to keep")
    batch_size = luigi.IntParameter(default=256, description="Training batch size")
    learning_rate = luigi.FloatParameter(default=1e-4, description="Learning rate")

    def requires(self):
        return PreprocessFeatures.req(self)

    def output(self):
        return {
            "checkpoints": law.LocalDirectoryTarget("results/models/maf_checkpoints"),
            "training_plot": law.LocalFileTarget("results/plots/maf_training_history.png"),
            "best_model": law.LocalFileTarget("results/models/maf_best.pt"),
        }

    def run(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from src.data.loader import load_data, apply_region_cut, get_features
        from src.data.preprocessing import FeatureScaler
        from src.cathode.maf import ConditionalMAF

        print("=" * 80)
        print("TASK 3.3: Training CATHODE Outer Density Estimator")
        print("=" * 80)
        print()

        # Load scaler
        scaler = FeatureScaler.load(self.requires().output().path)
        feature_names = scaler.feature_names_

        # Load dataset
        data_path = f"{self.data_dir}/background.h5"
        bg_data, sig_data = load_data(data_path)

        # Get sideband background data
        bg_sb_mask = apply_region_cut(
            bg_data, self.mjj_low, self.mjj_high, self.sr_low, self.sr_high, False
        )
        bg_sb = {key: val[bg_sb_mask] for key, val in bg_data.items()}

        print(f"Sideband background events: {len(bg_sb['mJJ']):,}")
        print(f"Using {self.n_train:,} for training, {self.n_val:,} for validation")
        print()

        # Extract and preprocess features
        X_sb = get_features(bg_sb, include_mass=False)
        X_sb_scaled = scaler.transform(X_sb)
        m_sb = bg_sb["mJJ"]  # Conditioning variable (in TeV)

        # Split into train/val
        n_total = len(X_sb_scaled)
        indices = np.random.permutation(n_total)
        train_indices = indices[: self.n_train]
        val_indices = indices[self.n_train : self.n_train + self.n_val]

        X_train = X_sb_scaled[train_indices]
        m_train = m_sb[train_indices]
        X_val = X_sb_scaled[val_indices]
        m_val = m_sb[val_indices]

        print(f"Training set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        print()

        # Create MAF with paper specifications
        print("Creating MAF model...")
        print("  Architecture: 15 MADE blocks, 128 hidden units")
        print("  Batch normalization: momentum 1.0")
        print("  Optimizer: Adam, lr=1e-4")
        print()

        maf = ConditionalMAF(
            features=X_train.shape[1],
            context_features=1,
            hidden_features=128,
            num_layers=15,
            use_batch_norm=True,
            batch_norm_momentum=1.0,
        )

        # Train the model
        print(f"Training for {self.epochs} epochs...")
        print()

        history = maf.fit(
            X_train,
            m_train,
            X_val,
            m_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=True,
        )

        # Save all checkpoints and find best ones
        print()
        print("Saving checkpoints...")
        self.output()["checkpoints"].touch()

        checkpoint_dir = self.output()["checkpoints"].path
        val_losses = np.array(history["val_loss"])

        # Save all epochs
        for epoch in range(self.epochs):
            checkpoint_path = f"{checkpoint_dir}/maf_epoch_{epoch:03d}.pt"
            maf.save(checkpoint_path)

        # Find best N checkpoints
        best_epochs = np.argsort(val_losses)[: self.n_checkpoints]
        print(f"\nBest {self.n_checkpoints} epochs (by validation loss):")
        for rank, epoch in enumerate(best_epochs):
            print(f"  Rank {rank+1}: Epoch {epoch+1:3d}, val_loss={val_losses[epoch]:.4f}")

        # Save the single best model
        best_epoch = best_epochs[0]
        best_model_path = f"{checkpoint_dir}/maf_epoch_{best_epoch:03d}.pt"

        # Copy best model to output location
        import shutil
        self.output()["best_model"].parent.touch()
        shutil.copy(best_model_path, self.output()["best_model"].path)

        print(f"\nBest model (epoch {best_epoch+1}) saved to {self.output()['best_model'].path}")
        print(f"  Final train loss: {history['train_loss'][best_epoch]:.4f}")
        print(f"  Final val loss: {history['val_loss'][best_epoch]:.4f}")

        # Plot training history
        print("\nGenerating training history plot...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        epochs_array = np.arange(1, self.epochs + 1)
        ax.plot(epochs_array, history["train_loss"], label="Train loss", alpha=0.7, linewidth=2)
        ax.plot(epochs_array, history["val_loss"], label="Val loss", alpha=0.7, linewidth=2)

        # Mark best epoch
        ax.axvline(best_epoch + 1, color='red', linestyle='--', alpha=0.5, label=f'Best epoch ({best_epoch+1})')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative log-likelihood")
        ax.set_title("CATHODE Outer Density Estimator Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.output()["training_plot"].parent.touch()
        plt.savefig(self.output()["training_plot"].path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Training plot saved to {self.output()['training_plot'].path}")

        # Verification against paper
        print()
        print("=" * 80)
        print("Verification")
        print("=" * 80)
        print(f"Paper reports loss around ~5.3")
        print(f"Our best val loss: {val_losses[best_epoch]:.4f}")
        if 4.5 < val_losses[best_epoch] < 6.0:
            print("✓ Loss is in expected range!")
        else:
            print("⚠ Loss differs from paper - may need hyperparameter tuning")

        print()
        print("Task 3.3 complete!")
