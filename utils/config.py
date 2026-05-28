import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.Probe import ProbeSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    env_value = os.getenv("LINEAR_SVD_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser()
    return PROJECT_ROOT / "Data"


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_root: Path = field(default_factory=_default_data_root)
    results_root: Path = field(default_factory=lambda: PROJECT_ROOT / "Results")

    def __post_init__(self):
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.results_root = Path(self.results_root).expanduser().resolve()

    def probe_settings_file(self, probe_name: str) -> Path:
        return self.project_root / "utils" / "ProbeSettings" / f"{probe_name}.json"

    def presettings_file(self, file_name: str) -> Path:
        return self.project_root / "utils" / "PreSettings" / file_name

    def session_data_dir(self, date: str) -> Path:
        return self.data_root / date

    def unpacked_dir(self, date: str, data_name: str) -> Path:
        return self.session_data_dir(date) / data_name / "unpacked"

    def recon_frame_file(self, date: str, data_name: str, module: str, idx: int) -> Path:
        return self.session_data_dir(date) / f"Recon_{data_name}_{module}_{idx}.npy"

    def result_date_dir(self, date: str) -> Path:
        return self.results_root / date

    def figure_dir(self, date: str) -> Path:
        return self.result_date_dir(date) / "png"

    def figure_file(self, date: str, data_name: str, module: str, dynamic_range: int, idx: int) -> Path:
        return self.figure_dir(date) / f"{data_name}_{module}_{dynamic_range}dB_{idx}.png"

    def video_dir(self, date: str) -> Path:
        return self.result_date_dir(date)

    def reconstruction_video_file(self, date: str, data_name: str, module: str) -> Path:
        return self.video_dir(date) / f"{data_name}_{module}_reconstruction_video.mp4"

    def video_file(self, date: str, data_name: str, module: str) -> Path:
        return self.reconstruction_video_file(date, data_name, module)

    def svd_video_file(self, date: str, data_name: str, module: str, svd_num: int) -> Path:
        return self.video_dir(date) / f"{data_name}_{module}_SVD_{svd_num}_video.mp4"

    def svd_frame_dir(self, date: str) -> Path:
        return self.result_date_dir(date) / "svd_frames"

    def svd_frame_file(self, date: str, data_name: str, module: str, svd_num: int, idx: int) -> Path:
        return self.svd_frame_dir(date) / f"SVD_{data_name}_{module}_{svd_num}_{idx}.npy"

    def svd_figure_file(self, date: str, data_name: str, module: str, svd_num: int, dynamic_range: int) -> Path:
        return self.result_date_dir(date) / f"SVD_{data_name}_{module}_{svd_num}_{dynamic_range}dB.png"

    def ensure_runtime_dirs(self, recon_para: "ReconParams"):
        self.session_data_dir(recon_para.Date).mkdir(parents=True, exist_ok=True)
        self.result_date_dir(recon_para.Date).mkdir(parents=True, exist_ok=True)
        if recon_para.flag_Figure:
            self.figure_dir(recon_para.Date).mkdir(parents=True, exist_ok=True)
        if recon_para.flag_SVD_Frames:
            self.svd_frame_dir(recon_para.Date).mkdir(parents=True, exist_ok=True)


@dataclass
class ReconParams:
    Data_Name: str = "Kidney_US"
    Data_Type: str = "PW"
    angles: np.ndarray = field(default_factory=lambda: np.arange(-15, 16, 5, dtype=int))
    Module: str = "DAS"
    Date: str = "20260114"
    Probe: str = "L15_80M_128"

    frame_num: int = 7
    SVD_num: int = 150
    idx: int = 0
    segment_Nx: int = 1
    dynamic_range_figure: int = 60
    dynamic_range_svd: int = 20
    dynamic_range_video: int = 60
    vc: float = 1510.0

    image_nx: int = 300
    image_nz: int = 300
    image_dx: float = None
    image_dz: float = None
    zcenter: float = 0.0075
    gpu_device: int = 0
    toffset_correction_samples: int = -90

    flag_Data_Process: bool = True
    flag_Recon: bool = True
    flag_SVD: bool = False
    flag_Figure: bool = True
    flag_Video: bool = False
    flag_SVD_Video: bool = False
    flag_SVD_Frames: bool = False
    flag_sensitivity: bool = False
    flag_position: bool = False
    preprocess_workers: int = None
    async_save: bool = True
    emit_perf_metrics: bool = True

    def __post_init__(self):
        self.angles = np.asarray(self.angles, dtype=int)
        if self.dynamic_range_video is None:
            self.dynamic_range_video = self.dynamic_range_figure

    def apply_to_probe(self, para: ProbeSettings):
        para.vc = self.vc
        para.Nx = self.image_nx
        para.Nz = self.image_nz
        para.dx = self.image_dx if self.image_dx is not None else para.res
        para.dz = self.image_dz if self.image_dz is not None else para.dx
        para.zcenter = self.zcenter
        para.sample_offset = int(para.Toffset * para.fs) + self.toffset_correction_samples

    def validate(self, para: ProbeSettings, paths: PathConfig):
        if not self.Data_Name:
            raise ValueError("Data_Name cannot be empty.")
        if not self.Module:
            raise ValueError("Module cannot be empty.")
        if len(self.Date) != 8 or not self.Date.isdigit():
            raise ValueError("Date must use YYYYMMDD format.")
        if self.frame_num <= 0:
            raise ValueError("frame_num must be a positive integer.")
        if self.segment_Nx <= 0:
            raise ValueError("segment_Nx must be a positive integer.")
        if self.preprocess_workers is not None and self.preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be positive when provided.")
        if self.dynamic_range_figure <= 0 or self.dynamic_range_svd <= 0 or self.dynamic_range_video <= 0:
            raise ValueError("Dynamic range parameters must be positive.")
        if self.image_nx <= 0 or self.image_nz <= 0:
            raise ValueError("image_nx and image_nz must be positive integers.")
        if self.angles.ndim != 1 or self.angles.size == 0:
            raise ValueError("angles must be a non-empty 1D array.")
        if self.angles.size > 1:
            diffs = np.diff(self.angles)
            if not np.all(diffs == diffs[0]):
                raise ValueError("angles must be evenly spaced for the current reconstruction implementation.")
        if self.flag_SVD and not (0 <= self.SVD_num < self.frame_num):
            raise ValueError("SVD_num must satisfy 0 <= SVD_num < frame_num when SVD is enabled.")
        if (self.flag_SVD_Video or self.flag_SVD_Frames) and not self.flag_SVD:
            raise ValueError("SVD video or SVD frame outputs require SVD to be enabled.")
        if self.flag_Recon and not self.flag_Data_Process:
            raise ValueError(
                "Reconstruction now requires streaming raw preprocessing; processed frame caches are no longer supported."
            )
        if self.flag_Data_Process and not paths.unpacked_dir(self.Date, self.Data_Name).exists():
            raise FileNotFoundError(f"Raw data directory not found: {paths.unpacked_dir(self.Date, self.Data_Name)}")

        para.validate()
