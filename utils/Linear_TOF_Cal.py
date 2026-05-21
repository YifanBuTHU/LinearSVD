import numpy as np
import cupy as cp

def Linear_TOF_Cal(Para, Recon_Para):
    Nx = Para.Nx
    Nz = Para.Nz
    dx = cp.float32(Para.dx)
    dz = cp.float32(Para.dz)
    num_ele = Para.num_ele
    vc = cp.float32(Para.vc)
    fs = cp.float32(Para.fs)
    pitch = cp.float32(Para.pitch)
    ez = cp.float32(Para.zcenter)
    f_number = cp.float32(1.0)
    sample_offset = cp.float32(
        Para.sample_offset if Para.sample_offset is not None else int(Para.Toffset * Para.fs)
    )
    num_samples = int(Para.num_samples)

    segment_count = max(1, min(int(Recon_Para.segment_Nx), Nx))
    x_bounds = np.linspace(0, Nx, segment_count + 1, dtype=int)

    x_indices = cp.arange(Nx, dtype=cp.float32) - Nx / 2 + 0.5
    z_indices = cp.arange(Nz, dtype=cp.float32) - Nz / 2 + 0.5
    x_coords = x_indices * dx
    z_coords = z_indices * dz
    element_x = (cp.arange(num_ele, dtype=cp.float32) - num_ele / 2 + 0.5) * pitch
    angle_radians = cp.deg2rad(cp.asarray(Recon_Para.angles, dtype=cp.float32))

    z_grid = z_coords[None, :, None]
    z_grid_angles = z_coords[None, :, None]
    ex = element_x[None, None, :]
    angle_sin = cp.sin(angle_radians)[None, None, :]
    angle_cos = cp.cos(angle_radians)[None, None, :]
    angle_bias = (num_ele / 2) * pitch * cp.abs(angle_sin)

    segments = []
    for start, end in zip(x_bounds[:-1], x_bounds[1:]):
        if end <= start:
            continue

        x_segment = x_coords[start:end][:, None, None]
        receive = cp.sqrt((x_segment - ex) ** 2 + (z_grid - ez) ** 2) / vc * fs

        aperture_ratio = cp.abs((ex - x_segment) / cp.maximum(cp.abs(ez - z_grid), cp.float32(1e-6)))
        w_flag = (aperture_ratio < f_number / 2).astype(cp.float32)
        w_num = cp.maximum(cp.sum(w_flag, axis=2, keepdims=True), cp.float32(1.0))
        w = cp.cumsum(w_flag, axis=2) / w_num
        w = 0.5 - 0.5 * cp.cos(2 * cp.pi * w)
        w = cp.where(w_flag > 0, w, 0).astype(cp.float32, copy=False)

        transmit = ((ez - z_grid_angles) * angle_cos) + (x_segment * angle_sin) + angle_bias
        transmit = transmit / vc * fs

        total_tof = receive[:, :, :, None] + transmit[:, :, None, :] - sample_offset
        sample_rf = cp.floor(total_tof).astype(cp.int32)
        sample_rf = cp.where((sample_rf >= 1) & (sample_rf < num_samples), sample_rf, 0)

        segments.append(
            {
                "x_slice": slice(int(start), int(end)),
                "sample_rf": cp.ascontiguousarray(sample_rf),
                "weights": cp.ascontiguousarray(w),
            }
        )

    return {
        "segments": segments,
        "channel_indices": cp.arange(num_ele, dtype=cp.int32)[None, None, :],
        "angle_count": int(angle_radians.size),
    }
