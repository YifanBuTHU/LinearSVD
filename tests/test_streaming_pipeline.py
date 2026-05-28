from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from Modules.DAS.Reconstruction import Reconstruction
from pipeline.runner import run_pipeline
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
        # TsingPaiDataset reads with Fortran reshape then transposes.
        packed = data.T.reshape(-1, order="F")
        (frame_dir / f"line{emission_idx:010d}.dat").write_bytes(packed.tobytes())


def _make_dataset(tmp_path: Path, date="20250101", name="tiny", frames=2, emissions=2):
    probe = _probe()
    unpacked = tmp_path / "data" / date / name / "unpacked"
    for frame_idx in range(frames):
        _write_raw_frame(unpacked / f"frame_{frame_idx:04d}", probe, emissions, seed=frame_idx)
    return probe


def _legacy_processed_frame(paths: PathConfig, date: str, name: str, idx: int):
    return paths.session_data_dir(date) / f"{name}_{idx}.npy"


def _reference_das(data, recon_context, para):
    data_np = cp.asnumpy(data).astype(np.complex64, copy=False)
    data_np[0, :, :] = 0
    output = np.empty((para.Nx, para.Nz), dtype=np.complex64)
    angle_count = int(recon_context["angle_count"])

    for segment in recon_context["segments"]:
        sample_rf = cp.asnumpy(segment["sample_rf"])
        weights = cp.asnumpy(segment["weights"])
        segment_out = np.zeros((sample_rf.shape[0], para.Nz), dtype=np.complex64)

        for x_idx in range(sample_rf.shape[0]):
            for z_idx in range(para.Nz):
                total = np.complex64(0)
                for ele_idx in range(para.num_ele):
                    weight = weights[x_idx, z_idx, ele_idx]
                    for angle_idx in range(angle_count):
                        sample_idx = sample_rf[x_idx, z_idx, ele_idx, angle_idx]
                        if 0 < sample_idx < para.num_samples:
                            total += data_np[sample_idx, ele_idx, angle_idx] * weight
                segment_out[x_idx, z_idx] = total / (para.num_ele * angle_count)

        output[segment["x_slice"], :] = segment_out

    return cp.asarray(output)


def test_das_cuda_kernel_matches_reference_formula():
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
    )
    recon_para.apply_to_probe(para)
    recon_context = Linear_TOF_Cal(para, recon_para)

    real = rng.normal(size=(para.num_samples, para.num_ele, recon_para.angles.size)).astype(np.float32)
    imag = rng.normal(size=real.shape).astype(np.float32)
    data = cp.asarray(real + 1j * imag)

    expected = _reference_das(data, recon_context, para)
    actual = Reconstruction(data, recon_context, para, recon_para)

    cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_streaming_pipeline_does_not_write_processed_cache(tmp_path: Path):
    date = "20250101"
    name = "tiny"
    probe = _make_dataset(tmp_path, date=date, name=name, frames=2, emissions=2)
    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
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


def test_reconstruction_requires_streaming_preprocess(tmp_path: Path):
    probe = _probe()
    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
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

    with pytest.raises(ValueError, match="streaming raw preprocessing"):
        run_pipeline(recon_para, probe, paths, use_tqdm=False, logger=lambda *_: None)
