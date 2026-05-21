import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class ProbeSettings:
    """Probe Settings for Plane Wave Beamforming."""
    vc: float = None               # sound velocity (m/s)
    pitch: float = None            # pitch of the array (mm)
    fs: float = None               # sampling frequency (Hz)
    f0: float = None               # center frequency (Hz)
    Toffset: float = None          # delay of first sample after emission
    num_samples: int = None        # number of samples
    ring_angle: float = None       # angle of the ring (degrees)
    num_ele: int = None            # number of channels
    Nx: int = None                 # number of pixels in x direction
    Nz: int = None                 # number of pixels in z direction
    dx: float = None               # pixel spacing in x direction
    dz: float = None               # pixel spacing in z direction
    zcenter: float = None          # probe z reference
    sample_offset: int = None      # first valid RF sample in samples

    @property
    def res(self) -> float:
        return self.vc / (2 * self.f0)

    @property
    def max_depth(self) -> float:
        return (self.vc * self.num_samples) / (2 * self.fs)
    
    '''@property
    def dx(self) -> float:
        return self.res
    
    @property
    def dz(self) -> float:
        return self.dx'''

    def __str__(self):
        return (
            f"🔧 Probe Settings:\n"
            f"  Sound velocity        : {self.vc} m/s\n"
            f"  Array pitch           : {self.pitch*1e3} mm\n"
            f"  Sampling frequency    : {self.fs/1e6} MHz\n"
            f"  Number of samples     : {self.num_samples}\n"
            f"  Number of channels    : {self.num_ele}\n"
            f"  Center frequency      : {self.f0/1e6} MHz\n"
            f"  Resolution (Res)      : {self.res*1e3} mm\n"
            f"  Max depth (MaxD)      : {self.max_depth*1e3} mm\n"
        )

    def save_json(self, path: str):
        """Save probe configuration to JSON file."""
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"💾 Probe settings saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> "ProbeSettings":
        """Load probe configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def validate(self):
        required_fields = {
            "vc": self.vc,
            "pitch": self.pitch,
            "fs": self.fs,
            "f0": self.f0,
            "Toffset": self.Toffset,
            "num_samples": self.num_samples,
            "num_ele": self.num_ele,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(f"Probe settings are missing required fields: {', '.join(missing)}")

        numeric_checks = {
            "vc": self.vc,
            "pitch": self.pitch,
            "fs": self.fs,
            "f0": self.f0,
            "num_samples": self.num_samples,
            "num_ele": self.num_ele,
        }
        invalid = [name for name, value in numeric_checks.items() if value <= 0]
        if invalid:
            raise ValueError(f"Probe settings must be positive for: {', '.join(invalid)}")

        if self.Nx is not None and self.Nx <= 0:
            raise ValueError("Nx must be positive when provided.")
        if self.Nz is not None and self.Nz <= 0:
            raise ValueError("Nz must be positive when provided.")
        if self.dx is not None and self.dx <= 0:
            raise ValueError("dx must be positive when provided.")
        if self.dz is not None and self.dz <= 0:
            raise ValueError("dz must be positive when provided.")
