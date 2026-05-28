import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(data):
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return data.get()
    except ImportError:
        pass
    return np.asarray(data)


def format_db_image(data, dynamic_range):
    data_np = _to_numpy(data)
    magnitude = np.abs(data_np)
    max_val = np.max(magnitude)
    if max_val <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    magnitude = magnitude / max_val
    threshold = 10 ** (-dynamic_range / 20)
    magnitude[magnitude < threshold] = threshold
    return 20 * np.log10(magnitude)


def frame_to_global_uint8(frame, global_max, dynamic_range):
    magnitude = np.abs(_to_numpy(frame)).astype(np.float32, copy=False)
    if global_max <= 0:
        return np.zeros_like(magnitude, dtype=np.uint8)

    threshold = 10 ** (-dynamic_range / 20)
    normalized = np.clip(magnitude / float(global_max), threshold, 1.0)
    db_image = 20 * np.log10(normalized)
    scaled = (db_image + dynamic_range) / dynamic_range * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _cv2():
    import cv2

    return cv2


def _write_video_frames(frames, video_name, dynamic_range, progress_callback=None, stage="video", message="Generating video"):
    frames = list(frames)
    if not frames:
        raise ValueError("No frames are available for video generation.")

    global_max = max(float(np.max(np.abs(frame))) for _, frame in frames)
    first_frame = frame_to_global_uint8(frames[0][1], global_max, dynamic_range)
    height, width = first_frame.shape

    cv2 = _cv2()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_name), fourcc, 30, (width, height))

    if progress_callback:
        progress_callback({"stage": stage, "current": 0, "total": len(frames), "message": message})

    try:
        for current, (_, frame) in enumerate(frames, start=1):
            frame_uint8 = frame_to_global_uint8(frame, global_max, dynamic_range)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame_bgr)
            if progress_callback:
                progress_callback({"stage": stage, "current": current, "total": len(frames), "message": message})
    finally:
        video_writer.release()


def save_reconstruction_figure(recon_para, para, paths, data_recon=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_figure
    idx = recon_para.idx

    if data_recon is None:
        data_recon = np.load(paths.recon_frame_file(date, data_name, module, idx))

    nz, nx = data_recon.shape
    x_coords = np.arange(nx) * para.dx * 1000
    z_coords = np.arange(nz) * para.dz * 1000
    image_db = format_db_image(data_recon, dynamic_range)

    dpi = 200
    img_width_in = nx * 4 / dpi
    img_height_in = nz * 4 / dpi
    fig_width = img_width_in + 1.2
    fig_height = img_height_in + 0.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    im = ax.imshow(
        image_db,
        cmap="gray",
        extent=[x_coords[0], x_coords[-1], z_coords[-1], z_coords[0]],
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(f"{data_name}_{module}_{dynamic_range}dB_{idx}")

    if x_coords[-1] != x_coords[0]:
        ax.set_box_aspect((z_coords[-1] - z_coords[0]) / (x_coords[-1] - x_coords[0]))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Amplitude (dB)")

    plt.savefig(paths.figure_file(date, data_name, module, dynamic_range, idx), dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close()


def save_svd_figure(data_PDI, recon_para, paths, para=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    image_db = format_db_image(data_PDI, recon_para.dynamic_range_svd)

    nz, nx = data_PDI.shape
    if para is not None:
        x_coords = np.arange(nx) * para.dx * 1000
        z_coords = np.arange(nz) * para.dz * 1000
        extent = [x_coords[0], x_coords[-1], z_coords[-1], z_coords[0]]
        xlabel = "X (mm)"
        ylabel = "Z (mm)"
        aspect = "equal"
        box_aspect = (z_coords[-1] - z_coords[0]) / (x_coords[-1] - x_coords[0]) if (x_coords[-1] != x_coords[0]) else 1
    else:
        extent = [0, nx, nz, 0]
        xlabel = "X (pixel)"
        ylabel = "Z (pixel)"
        aspect = "equal"
        box_aspect = nz / nx if nx != 0 else 1

    dpi = 200
    img_width_in = nx * 4 / dpi
    img_height_in = nz * 4 / dpi
    fig_width = img_width_in + 1.2
    fig_height = img_height_in + 0.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    im = ax.imshow(image_db, cmap="hot", extent=extent, aspect=aspect, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{recon_para.Data_Name}_{recon_para.Module}_SVD{recon_para.SVD_num}_{recon_para.dynamic_range_svd}dB"
    )
    if nx > 0 and nz > 0:
        ax.set_box_aspect(box_aspect)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Amplitude (dB)")

    plt.savefig(
        paths.svd_figure_file(
            recon_para.Date,
            recon_para.Data_Name,
            recon_para.Module,
            recon_para.SVD_num,
            recon_para.dynamic_range_svd,
        ),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close()


def create_video_from_stack(
    recon_para,
    data,
    paths,
    progress_callback=None,
    logger=None,
    video_path=None,
    stage="video",
    message="Generating video",
):
    logger = logger or print
    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_video

    output_dir = paths.video_dir(date)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path or paths.reconstruction_video_file(date, data_name, module)

    data_np = _to_numpy(data)
    _, _, frame_num = data_np.shape
    frames = [(idx, data_np[:, :, idx]) for idx in range(frame_num)]

    logger(f"Generating video from data with {frame_num} frames...")
    _write_video_frames(frames, video_name, dynamic_range, progress_callback, stage=stage, message=message)
    logger(f"Video saved as: {video_name}")


def create_video_from_files(recon_para, paths, progress_callback=None, logger=None):
    logger = logger or print
    data_name = recon_para.Data_Name
    module = recon_para.Module
    date = recon_para.Date
    dynamic_range = recon_para.dynamic_range_video
    frame_num = recon_para.frame_num

    output_dir = paths.video_dir(date)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = paths.reconstruction_video_file(date, data_name, module)

    frames = []
    for idx in range(frame_num):
        try:
            frames.append((idx, np.load(paths.recon_frame_file(date, data_name, module, idx))))
        except FileNotFoundError:
            logger(f"Warning: file for frame {idx} not found, skipping...")

    logger(f"Processing {len(frames)} reconstructed frames for video generation...")
    _write_video_frames(frames, video_name, dynamic_range, progress_callback)
    logger(f"Video saved as: {video_name}")
