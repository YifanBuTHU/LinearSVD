from pathlib import Path

import numpy as np
import pytest

from utils.config import PathConfig, ReconParams

from visualization.rendering import frame_to_global_uint8


def test_video_frame_mapping_uses_global_maximum():
    weak_frame = np.array([[1.0 + 0.0j, 0.1 + 0.0j]], dtype=np.complex64)
    strong_frame = np.array([[10.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex64)

    weak_uint8 = frame_to_global_uint8(weak_frame, global_max=10.0, dynamic_range=40)
    strong_uint8 = frame_to_global_uint8(strong_frame, global_max=10.0, dynamic_range=40)

    assert strong_uint8.max() == 255
    assert weak_uint8.max() < 255


def test_svd_filter_saves_complex_frames_with_original_indices(tmp_path: Path):
    cp = pytest.importorskip("cupy")
    from utils.svd import SVD_filter

    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )
    recon_para = ReconParams(
        Data_Name="tiny",
        Date="20250101",
        SVD_num=1,
        frame_num=5,
        flag_SVD=True,
        flag_SVD_Frames=True,
    )
    svd_filter = SVD_filter(recon_para, paths)

    svd_stack = cp.asarray(
        np.array(
            [
                [[1 + 2j, 3 + 4j]],
                [[5 + 6j, 7 + 8j]],
            ],
            dtype=np.complex64,
        )
    )
    svd_filter.save_svd_frames(svd_stack, used_indices=[2, 4])

    first = np.load(paths.svd_frame_file("20250101", "tiny", "DAS", 1, 2))
    second = np.load(paths.svd_frame_file("20250101", "tiny", "DAS", 1, 4))

    assert first.dtype == np.complex64
    np.testing.assert_array_equal(first, np.array([[1 + 2j], [5 + 6j]], dtype=np.complex64))
    np.testing.assert_array_equal(second, np.array([[3 + 4j], [7 + 8j]], dtype=np.complex64))


def test_direct_svd_pdi_matches_stack_path(tmp_path: Path):
    cp = pytest.importorskip("cupy")
    from utils.svd import SVD_filter

    paths = PathConfig(
        project_root=Path.cwd(),
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )
    recon_para = ReconParams(
        Data_Name="tiny",
        Date="20250101",
        SVD_num=1,
        frame_num=4,
        flag_SVD=True,
    )
    svd_filter = SVD_filter(recon_para, paths)
    rng = np.random.default_rng(12)
    data_np = (
        rng.normal(size=(3, 2, 4)).astype(np.float32)
        + 1j * rng.normal(size=(3, 2, 4)).astype(np.float32)
    )
    data = cp.asarray(data_np, dtype=cp.complex64)

    data_svd, data_noise = svd_filter.SVD_Cal(data, recon_para.SVD_num)
    expected_pdi = svd_filter.pdi_from_svd_stack(data_svd)
    expected_noise = svd_filter.pdi_from_svd_stack(data_noise)
    actual_pdi, actual_noise = svd_filter.SVD_PDI_Cal(data, recon_para.SVD_num)

    cp.testing.assert_allclose(actual_pdi, expected_pdi, rtol=1e-5, atol=1e-5)
    cp.testing.assert_allclose(actual_noise, expected_noise, rtol=1e-5, atol=1e-5)
