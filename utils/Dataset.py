import numpy as np
from pathlib import Path
from typing import List, Union
import re

from utils.Probe import ProbeSettings


class TsingPaiDataset:
    """
    Data loader for TsingPai ultrasound frames.

    Each frame is a subfolder inside the 'unpacked' directory.
    Each frame contains multiple .dat files (one per angle).
    """

    def __init__(self, unpacked_path: Union[str, Path], probe: ProbeSettings, verbose: bool = False):
        self.unpacked_path = Path(unpacked_path)
        self.probe = probe
        self.verbose = verbose

        if not self.unpacked_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.unpacked_path}")

        self.frame_folders = self._find_frame_folders()
        self._dat_files_cache: dict[Path, list[Path]] = {}
        if verbose:
            print(f"Found {len(self.frame_folders)} frame folders.")

    @staticmethod
    def natural_sort_key(value: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", value)]

    def _find_frame_folders(self) -> List[Path]:
        folders = [f for f in self.unpacked_path.iterdir() if f.is_dir() and not f.name.startswith(".")]
        return sorted(folders, key=lambda folder: self.natural_sort_key(folder.name))

    def _frame_dat_files(self, folder: Path) -> list[Path]:
        cached = self._dat_files_cache.get(folder)
        if cached is not None:
            return cached

        dat_files = sorted(
            [f for f in folder.glob("*.dat") if f.is_file()],
            key=lambda path: self.natural_sort_key(path.name),
        )
        self._dat_files_cache[folder] = dat_files
        return dat_files

    def frame_name(self, idx: int) -> str:
        return self.frame_folders[idx].name

    def frame_emission_count(self, idx: int) -> int:
        return len(self._frame_dat_files(self.frame_folders[idx]))

    def __len__(self):
        return len(self.frame_folders)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self):
            raise IndexError("Frame index out of range.")

        folder = self.frame_folders[idx]
        if self.verbose:
            print(f"Loading frame {idx}: {folder.name}")

        return self._load_frame(folder)

    def _load_frame(self, folder: Path) -> np.ndarray:
        dat_files = self._frame_dat_files(folder)

        num_emissions = len(dat_files)
        if self.verbose:
            print(f"  {num_emissions} emissions detected")

        D = np.zeros((self.probe.num_samples, self.probe.num_ele, num_emissions), dtype=np.int16)

        for i, dat_file in enumerate(dat_files):
            with open(dat_file, "rb") as f:
                data = np.fromfile(f, dtype=np.int16)

            total_samples = data.size
            if total_samples % self.probe.num_ele != 0:
                print(f"Skipping invalid file: {dat_file.name}")
                continue

            reshaped = data.reshape((self.probe.num_ele, -1), order="F").T
            if reshaped.shape[0] > self.probe.num_samples:
                reshaped = reshaped[-self.probe.num_samples :, :]
            elif reshaped.shape[0] < self.probe.num_samples:
                padded = np.zeros((self.probe.num_samples, self.probe.num_ele), dtype=np.int16)
                padded[-reshaped.shape[0] :, :] = reshaped
                reshaped = padded

            D[:, :, i] = reshaped

        return D
