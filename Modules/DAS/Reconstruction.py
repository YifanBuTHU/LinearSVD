from Modules.DAS.cuda_kernel import das_fused_reconstruction


def Reconstruction(data, recon_context, Para, Recon_Para):
    Nx = Para.Nx
    Nz = Para.Nz
    num_ele = Para.num_ele
    angle_count = recon_context["angle_count"]
    segments = recon_context["segments"]

    if data.ndim != 3:
        raise ValueError(f"Expected beamforming input with 3 dimensions, but got shape {data.shape}.")
    if data.shape[2] != angle_count:
        raise ValueError(
            f"Beamforming angle mismatch: frame data has {data.shape[2]} emissions, "
            f"but the reconstruction tables expect {angle_count}."
        )

    return das_fused_reconstruction(data, segments, Nx, Nz, num_ele, angle_count)
