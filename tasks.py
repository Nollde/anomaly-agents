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


# Tasks will be added below as we implement them
