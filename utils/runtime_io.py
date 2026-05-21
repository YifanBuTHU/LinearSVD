import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from utils.config import PathConfig, ReconParams
from utils.Probe import ProbeSettings


def recon_params_to_dict(recon_para: ReconParams) -> dict:
    data = asdict(recon_para)
    data["angles"] = np.asarray(recon_para.angles, dtype=int).tolist()
    return data


def recon_params_from_dict(data: dict) -> ReconParams:
    loaded = dict(data)
    loaded["angles"] = np.asarray(loaded.get("angles", []), dtype=int)
    return ReconParams(**loaded)


def probe_settings_to_dict(probe: ProbeSettings) -> dict:
    return asdict(probe)


def probe_settings_from_dict(data: dict) -> ProbeSettings:
    return ProbeSettings(**dict(data))


def path_config_to_dict(paths: PathConfig) -> dict:
    return {
        "project_root": str(paths.project_root),
        "data_root": str(paths.data_root),
        "results_root": str(paths.results_root),
    }


def path_config_from_dict(data: dict) -> PathConfig:
    return PathConfig(
        project_root=Path(data["project_root"]),
        data_root=Path(data["data_root"]),
        results_root=Path(data["results_root"]),
    )


def save_runtime_bundle(path, paths: PathConfig, recon_para: ReconParams, probe: ProbeSettings):
    payload = {
        "paths": path_config_to_dict(paths),
        "recon": recon_params_to_dict(recon_para),
        "probe": probe_settings_to_dict(probe),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_runtime_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    paths = path_config_from_dict(payload["paths"])
    recon_para = recon_params_from_dict(payload["recon"])
    probe = probe_settings_from_dict(payload["probe"])
    return paths, recon_para, probe
