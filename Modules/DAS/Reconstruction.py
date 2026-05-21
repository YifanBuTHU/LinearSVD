import cupy as cp


def Reconstruction(data, recon_context, Para, Recon_Para):
    Nx = Para.Nx
    Nz = Para.Nz
    num_ele = Para.num_ele
    angle_count = recon_context["angle_count"]
    channel_indices = recon_context["channel_indices"][:, :, :, None]
    segments = recon_context["segments"]
    angle_batch_size = Recon_Para.das_angle_batch_size or angle_count
    angle_batch_size = max(1, min(int(angle_batch_size), int(angle_count)))

    data = cp.asarray(data, dtype=cp.complex64)
    data[0, :, :] = 0

    if data.ndim != 3:
        raise ValueError(f"Expected beamforming input with 3 dimensions, but got shape {data.shape}.")
    if data.shape[2] != angle_count:
        raise ValueError(
            f"Beamforming angle mismatch: frame data has {data.shape[2]} emissions, "
            f"but the reconstruction tables expect {angle_count}."
        )

    data_recon = cp.zeros((Nx, Nz), dtype=cp.complex64)

    for segment in segments:
        sample_rf = segment["sample_rf"]
        weights = segment["weights"][:, :, :, None]
        data_segment = cp.zeros((sample_rf.shape[0], Nz), dtype=cp.complex64)

        for start in range(0, angle_count, angle_batch_size):
            stop = min(start + angle_batch_size, angle_count)
            angle_indices = cp.arange(start, stop, dtype=cp.int32)[None, None, None, :]
            gathered = data[sample_rf[:, :, :, start:stop], channel_indices, angle_indices]
            data_segment += cp.sum(gathered * weights, axis=(2, 3)) / num_ele

        data_recon[segment["x_slice"], :] = data_segment / angle_count

    return data_recon
