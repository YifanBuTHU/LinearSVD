from pathlib import Path
import shutil
import sys
import tempfile

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Modules.DAS.Reconstruction import Reconstruction
from pipeline.runner import run_pipeline
from utils.Data_Process import build_preprocess_context, process_frame_data
from utils.Linear_TOF_Cal import Linear_TOF_Cal
from utils.Probe import ProbeSettings
from utils.config import PathConfig, ReconParams


def _probe(num_samples=256, num_ele=4):
    return ProbeSettings(
        vc=1510.0,
        pitch=0.3e-3,
        fs=40e6,
        f0=5e6,
        Toffset=4e-6,
        num_samples=num_samples,
        num_ele=num_ele,
    )


def _write_raw_frame(frame_dir: Path, probe: ProbeSettings, emission_count: int, seed: int):
    rng = np.random.default_rng(seed)
    frame_dir.mkdir(parents=True)
    for emission_idx in range(emission_count):
        data = rng.integers(
            -256,
            256,
            size=(probe.num_samples, probe.num_ele),
            dtype=np.int16,
        )
        packed = data.T.reshape(-1, order="F")
        (frame_dir / f"line{emission_idx:010d}.dat").write_bytes(packed.tobytes())


def _make_dataset(root: Path, date="20250101", name="tiny", frames=2, emissions=2):
    probe = _probe()
    unpacked = root / "data" / date / name / "unpacked"
    for frame_idx in range(frames):
        _write_raw_frame(unpacked / f"frame_{frame_idx:04d}", probe, emissions, seed=frame_idx)
    return probe


def _legacy_processed_frame(paths: PathConfig, date: str, name: str, idx: int):
    return paths.session_data_dir(date) / f"{name}_{idx}.npy"


def check_preprocess_is_gpu_only(tmp_root: Path):
    source = (Path.cwd() / "utils" / "Data_Process.py").read_text(encoding="utf-8")
    forbidden = ("scipy.signal", "filtfilt", "hilbert")
    for token in forbidden:
        assert token not in source, f"preprocessing must not use CPU SciPy signal path: {token}"

    probe = _probe(num_samples=64, num_ele=4)
    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_root / "data",
        results_root=tmp_root / "results",
    )
    recon_para = ReconParams(
        Data_Name="tiny",
        Date="20250101",
        angles=np.array([-5, 0]),
        frame_num=1,
        flag_sensitivity=False,
    )
    recon_para.apply_to_probe(probe)
    context = build_preprocess_context(recon_para, probe, paths)
    raw = np.random.default_rng(3).normal(size=(probe.num_samples, probe.num_ele, 2)).astype(np.float32)

    processed = process_frame_data(raw, recon_para, probe, context)

    assert isinstance(processed, cp.ndarray)
    assert processed.dtype == cp.complex64
    assert processed.shape == raw.shape


def check_das_angle_batching_matches_reference():
    rng = np.random.default_rng(11)
    para = _probe(num_samples=48, num_ele=5)
    recon_para = ReconParams(
        angles=np.array([-5, 0, 5]),
        image_nx=6,
        image_nz=5,
        image_dx=0.2e-3,
        image_dz=0.2e-3,
        zcenter=0.0,
        toffset_correction_samples=0,
        das_angle_batch_size=3,
    )
    recon_para.apply_to_probe(para)
    recon_context = Linear_TOF_Cal(para, recon_para)

    real = rng.normal(size=(para.num_samples, para.num_ele, recon_para.angles.size)).astype(np.float32)
    imag = rng.normal(size=real.shape).astype(np.float32)
    data = cp.asarray(real + 1j * imag)

    reference_para = ReconParams(
        angles=recon_para.angles,
        image_nx=recon_para.image_nx,
        image_nz=recon_para.image_nz,
        image_dx=recon_para.image_dx,
        image_dz=recon_para.image_dz,
        zcenter=recon_para.zcenter,
        toffset_correction_samples=0,
        das_angle_batch_size=1,
    )
    reference_para.apply_to_probe(para)

    expected = Reconstruction(data, recon_context, para, reference_para)
    actual = Reconstruction(data, recon_context, para, recon_para)
    cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def check_streaming_pipeline_does_not_write_processed_cache(tmp_root: Path):
    date = "20250101"
    name = "tiny"
    probe = _make_dataset(tmp_root, date=date, name=name, frames=2, emissions=2)
    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_root / "data",
        results_root=tmp_root / "results",
    )
    recon_para = ReconParams(
        Data_Name=name,
        Date=date,
        angles=np.array([-5, 0]),
        frame_num=2,
        image_nx=6,
        image_nz=5,
        image_dx=0.2e-3,
        image_dz=0.2e-3,
        zcenter=0.0,
        toffset_correction_samples=0,
        flag_Data_Process=True,
        flag_Recon=True,
        flag_Figure=False,
        flag_Video=False,
        flag_SVD=False,
        preprocess_workers=1,
        async_save=False,
    )

    run_pipeline(recon_para, probe, paths, use_tqdm=False, logger=lambda *_: None)
    assert paths.recon_frame_file(date, name, "DAS", 0).exists()
    assert paths.recon_frame_file(date, name, "DAS", 1).exists()
    assert not _legacy_processed_frame(paths, date, name, 0).exists()
    assert not _legacy_processed_frame(paths, date, name, 1).exists()


def check_reconstruction_requires_streaming_preprocess(tmp_root: Path):
    probe = _probe()
    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_root / "data",
        results_root=tmp_root / "results",
    )
    recon_para = ReconParams(
        Data_Name="tiny",
        Date="20250101",
        angles=np.array([-5, 0]),
        frame_num=1,
        flag_Data_Process=False,
        flag_Recon=True,
        flag_Figure=False,
        flag_Video=False,
        flag_SVD=False,
    )

    try:
        run_pipeline(recon_para, probe, paths, use_tqdm=False, logger=lambda *_: None)
    except ValueError as exc:
        assert "streaming raw preprocessing" in str(exc)
    else:
        raise AssertionError("run_pipeline should reject reconstruction without raw preprocessing")


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="linear_svd_stream_checks_"))
    try:
        check_das_angle_batching_matches_reference()
        check_reconstruction_requires_streaming_preprocess(tmp_root)
        check_streaming_pipeline_does_not_write_processed_cache(tmp_root)
        check_preprocess_is_gpu_only(tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
