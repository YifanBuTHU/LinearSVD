import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app.gui.runtime import GuiRuntimeMixin
from app.gui.shared import (
    PROJECT_ROOT,
    configure_tk_scaling,
    default_conda_executable,
    enable_high_dpi_awareness,
)
from utils.Probe import ProbeSettings
from utils.config import PathConfig, ReconParams


class LinearSVDGui(GuiRuntimeMixin):
    def __init__(self, root):
        self.root = root
        self.ui_scale = configure_tk_scaling(self.root)
        self.root.title("Linear SVD Ultrasound GUI")
        self.root.minsize(int(1280 * self.ui_scale), int(800 * self.ui_scale))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_window_geometry()

        self.process = None
        self.log_queue = queue.Queue()
        self.preview_image = None
        self.output_files = []
        self.temp_config_path = None
        self.run_started_at = None

        self._build_variables()
        self._build_layout()
        self._load_defaults()
        self._bind_traces()
        self._update_path_preview()
        self._update_toggle_states()
        self._poll_queue()

    def _configure_window_geometry(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        width = max(int(1280 * self.ui_scale), int(screen_width * 0.94))
        height = max(int(800 * self.ui_scale), int(screen_height * 0.92))
        width = min(width, screen_width)
        height = min(height, screen_height)
        pos_x = max((screen_width - width) // 2, 0)
        pos_y = max((screen_height - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    def _build_variables(self):
        default_paths = PathConfig()
        default_recon = ReconParams()

        self.data_root_var = tk.StringVar(value=r"E:\Study\PhD2\Data\Marsonics")
        self.results_root_var = tk.StringVar(value=str(default_paths.results_root))
        self.conda_executable_var = tk.StringVar(value=str(default_conda_executable()))
        self.conda_env_var = tk.StringVar(value="3DUS")
        self.data_name_var = tk.StringVar(value="")
        self.date_var = tk.StringVar(value="")
        self.data_type_var = tk.StringVar(value=default_recon.Data_Type)

        self.raw_input_preview_var = tk.StringVar()
        self.session_data_preview_var = tk.StringVar()
        self.results_preview_var = tk.StringVar()

        self.builtin_probe_var = tk.StringVar()
        self.probe_name_var = tk.StringVar(value=default_recon.Probe)
        self.probe_pitch_var = tk.StringVar()
        self.probe_fs_var = tk.StringVar()
        self.probe_f0_var = tk.StringVar()
        self.probe_toffset_var = tk.StringVar()
        self.probe_num_samples_var = tk.StringVar()
        self.probe_num_ele_var = tk.StringVar()
        self.probe_ring_angle_var = tk.StringVar()

        self.module_var = tk.StringVar(value=default_recon.Module)
        self.frame_num_var = tk.StringVar(value="")
        self.angle_start_var = tk.StringVar(value=str(int(default_recon.angles[0])))
        self.angle_end_var = tk.StringVar(value=str(int(default_recon.angles[-1])))
        self.angle_step_var = tk.StringVar(value=str(int(default_recon.angles[1] - default_recon.angles[0])))

        self.flag_data_process_var = tk.BooleanVar(value=default_recon.flag_Data_Process)
        self.flag_recon_var = tk.BooleanVar(value=default_recon.flag_Recon)
        self.flag_figure_var = tk.BooleanVar(value=default_recon.flag_Figure)
        self.flag_svd_var = tk.BooleanVar(value=default_recon.flag_SVD)
        self.flag_video_var = tk.BooleanVar(value=default_recon.flag_Video)
        self.flag_sensitivity_var = tk.BooleanVar(value=default_recon.flag_sensitivity)

        self.dynamic_range_figure_var = tk.StringVar(value=str(default_recon.dynamic_range_figure))
        self.svd_num_var = tk.StringVar(value=str(default_recon.SVD_num))
        self.dynamic_range_svd_var = tk.StringVar(value=str(default_recon.dynamic_range_svd))
        self.dynamic_range_video_var = tk.StringVar(value=str(default_recon.dynamic_range_video))

        self.runtime_vc_var = tk.StringVar(value=str(default_recon.vc))
        self.gpu_device_var = tk.StringVar(value=str(default_recon.gpu_device))
        self.toffset_correction_var = tk.StringVar(value=str(default_recon.toffset_correction_samples))
        self.segment_nx_var = tk.StringVar(value=str(default_recon.segment_Nx))

        self.image_nx_var = tk.StringVar(value=str(default_recon.image_nx))
        self.image_nz_var = tk.StringVar(value=str(default_recon.image_nz))
        self.image_dx_var = tk.StringVar(value="")
        self.image_dz_var = tk.StringVar(value="")
        self.zcenter_var = tk.StringVar(value=self._format_length_mm(default_recon.zcenter))

        self.status_var = tk.StringVar(value="Ready.")
        self.output_path_var = tk.StringVar(value="")
        self.progress_stage_var = tk.StringVar(value="idle")
        self.progress_message_var = tk.StringVar(value="Waiting to start")
        self.stage_progress_value_var = tk.DoubleVar(value=0.0)
        self.stage_progress_text_var = tk.StringVar(value="0 / 0")
        self.overall_progress_value_var = tk.DoubleVar(value=0.0)
        self.overall_progress_text_var = tk.StringVar(value="0%")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RIDGE,
            showhandle=True,
            bg="#cccccc",
        )
        main_pane.grid(row=0, column=0, sticky="nsew")

        controls_container = ttk.Frame(main_pane, padding=8)
        output_container = ttk.Frame(main_pane, padding=8)
        main_pane.add(controls_container, width=800, minsize=500)
        main_pane.add(output_container, minsize=300)

        self._build_scrollable_controls(controls_container)
        self._build_output_panel(output_container)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN, padding=(8, 4))
        status_bar.grid(row=1, column=0, sticky="ew")

    def _build_scrollable_controls(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        canvas = tk.Canvas(parent, highlightthickness=0)
        self.controls_canvas = canvas
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.controls_frame = ttk.Frame(canvas, padding=(0, 0, 12, 0))

        self.controls_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Enter>", self._bind_canvas_mousewheel)
        canvas.bind("<Leave>", self._unbind_canvas_mousewheel)
        self.controls_frame.bind("<Enter>", self._bind_canvas_mousewheel)
        self.controls_frame.bind("<Leave>", self._unbind_canvas_mousewheel)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.controls_frame.columnconfigure(0, weight=1)
        self._build_sections()

    def _build_sections(self):
        self._build_system_section()
        self._build_probe_section()
        self._build_reconstruction_section()
        self._build_action_section()

    def _build_system_section(self):
        section = ttk.LabelFrame(self.controls_frame, text="System Paths And Data", padding=10)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        section.columnconfigure(1, weight=1)

        ttk.Label(section, text="Conda Executable").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.conda_executable_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(section, text="Browse", command=lambda: self._browse_file(self.conda_executable_var)).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(section, text="Conda Env").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.conda_env_var).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(section, text="Data Root").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.data_root_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(section, text="Browse", command=lambda: self._browse_directory(self.data_root_var)).grid(row=2, column=2, padx=(8, 0), pady=4)

        ttk.Label(section, text="Results Root").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.results_root_var).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Button(section, text="Browse", command=lambda: self._browse_directory(self.results_root_var)).grid(row=3, column=2, padx=(8, 0), pady=4)

        ttk.Label(section, text="Data Name").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.data_name_var).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(section, text="Date (YYYYMMDD)").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.date_var).grid(row=5, column=1, sticky="ew", pady=4)

        ttk.Label(section, text="Data Type").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(section, textvariable=self.data_type_var, values=("PW",), state="readonly").grid(row=6, column=1, sticky="ew", pady=4)

        ttk.Separator(section, orient=tk.HORIZONTAL).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 8))

        ttk.Label(section, text="Raw Input Folder").grid(row=8, column=0, sticky="nw", padx=(0, 8), pady=2)
        ttk.Label(section, textvariable=self.raw_input_preview_var, wraplength=560, justify="left").grid(row=8, column=1, columnspan=2, sticky="w", pady=2)

        ttk.Label(section, text="Session Data Folder").grid(row=9, column=0, sticky="nw", padx=(0, 8), pady=2)
        ttk.Label(section, textvariable=self.session_data_preview_var, wraplength=560, justify="left").grid(row=9, column=1, columnspan=2, sticky="w", pady=2)

        ttk.Label(section, text="Results Folder").grid(row=10, column=0, sticky="nw", padx=(0, 8), pady=2)
        ttk.Label(section, textvariable=self.results_preview_var, wraplength=560, justify="left").grid(row=10, column=1, columnspan=2, sticky="w", pady=2)

    def _build_probe_section(self):
        section = ttk.LabelFrame(self.controls_frame, text="Probe Settings", padding=10)
        section.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        section.columnconfigure(1, weight=1)
        section.columnconfigure(3, weight=1)

        ttk.Label(section, text="Builtin Template").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.builtin_probe_combo = ttk.Combobox(section, textvariable=self.builtin_probe_var, state="readonly")
        self.builtin_probe_combo.grid(row=0, column=1, sticky="ew", pady=4)
        self.builtin_probe_combo.bind("<<ComboboxSelected>>", lambda event: self._load_builtin_probe())
        ttk.Button(section, text="Load Template", command=self._load_builtin_probe).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(section, text="Probe Label").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.probe_name_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(section, text="Import JSON", command=self._import_probe).grid(row=1, column=2, padx=(8, 0), pady=4)
        ttk.Button(section, text="Export JSON", command=self._export_probe).grid(row=1, column=3, padx=(8, 0), pady=4, sticky="w")

        note = "Probe file sound speed is ignored. Adjust sound speed in the Reconstruction section."
        ttk.Label(section, text=note, foreground="#8a5200", wraplength=700, justify="left").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 8)
        )

        self._labeled_entry(section, 3, 0, "pitch (mm)", self.probe_pitch_var)
        self._labeled_entry(section, 3, 2, "fs (MHz)", self.probe_fs_var)
        self._labeled_entry(section, 4, 0, "f0 (MHz)", self.probe_f0_var)
        self._labeled_entry(section, 4, 2, "Toffset (us)", self.probe_toffset_var)
        self._labeled_entry(section, 5, 0, "num_samples", self.probe_num_samples_var)
        self._labeled_entry(section, 5, 2, "num_ele", self.probe_num_ele_var)
        self._labeled_entry(section, 6, 0, "ring_angle (deg)", self.probe_ring_angle_var)

    def _build_reconstruction_section(self):
        section = ttk.LabelFrame(self.controls_frame, text="Reconstruction And Runtime", padding=10)
        section.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            section.columnconfigure(column, weight=1)

        ttk.Label(section, text="Module").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(section, textvariable=self.module_var, values=("DAS",), state="readonly").grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(section, text="Frame Count").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.frame_num_var).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Checkbutton(section, text="Preprocess Raw Data", variable=self.flag_data_process_var).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(section, text="Run Reconstruction", variable=self.flag_recon_var).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Checkbutton(section, text="Save Per-Frame PNG", variable=self.flag_figure_var, command=self._update_toggle_states).grid(row=1, column=2, sticky="w", pady=4)
        ttk.Checkbutton(section, text="Run SVD", variable=self.flag_svd_var, command=self._update_toggle_states).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Checkbutton(section, text="Generate Video", variable=self.flag_video_var, command=self._update_toggle_states).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Checkbutton(section, text="Sensitivity Correction", variable=self.flag_sensitivity_var).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(section, text="Angle Start").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.angle_start_var).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Angle End").grid(row=3, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.angle_end_var).grid(row=3, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="Angle Step").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.angle_step_var).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Sound Speed (m/s)").grid(row=4, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.runtime_vc_var).grid(row=4, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="GPU Device").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.gpu_device_var).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Toffset Correction (samples)").grid(row=5, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.toffset_correction_var).grid(row=5, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="X Segments (segment_Nx)").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.segment_nx_var).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Figure Dynamic Range").grid(row=6, column=2, sticky="w", padx=(12, 8), pady=4)
        self.figure_dynamic_entry = ttk.Entry(section, textvariable=self.dynamic_range_figure_var)
        self.figure_dynamic_entry.grid(row=6, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="SVD Num").grid(row=7, column=0, sticky="w", padx=(0, 8), pady=4)
        self.svd_num_entry = ttk.Entry(section, textvariable=self.svd_num_var)
        self.svd_num_entry.grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="SVD Dynamic Range").grid(row=7, column=2, sticky="w", padx=(12, 8), pady=4)
        self.svd_dynamic_entry = ttk.Entry(section, textvariable=self.dynamic_range_svd_var)
        self.svd_dynamic_entry.grid(row=7, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="Video Dynamic Range").grid(row=8, column=0, sticky="w", padx=(0, 8), pady=4)
        self.video_dynamic_entry = ttk.Entry(section, textvariable=self.dynamic_range_video_var)
        self.video_dynamic_entry.grid(row=8, column=1, sticky="ew", pady=4)

        ttk.Separator(section, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 8))

        ttk.Label(section, text="Image Nx").grid(row=10, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.image_nx_var).grid(row=10, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Image Nz").grid(row=10, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.image_nz_var).grid(row=10, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="Image dx (mm, blank=probe res)").grid(row=11, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.image_dx_var).grid(row=11, column=1, sticky="ew", pady=4)
        ttk.Label(section, text="Image dz (mm, blank=dx)").grid(row=11, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(section, textvariable=self.image_dz_var).grid(row=11, column=3, sticky="ew", pady=4)

        ttk.Label(section, text="zcenter (mm)").grid(row=12, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(section, textvariable=self.zcenter_var).grid(row=12, column=1, sticky="ew", pady=4)

    def _build_action_section(self):
        section = ttk.LabelFrame(self.controls_frame, text="Actions", padding=10)
        section.grid(row=3, column=0, sticky="ew")

        self.run_button = ttk.Button(section, text="Run", command=self._start_run)
        self.run_button.grid(row=0, column=0, padx=(0, 8), pady=4)

        self.stop_button = ttk.Button(section, text="Stop", command=self._stop_run, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 8), pady=4)

        ttk.Button(section, text="Validate", command=self._validate_only).grid(row=0, column=2, padx=(0, 8), pady=4)
        ttk.Button(section, text="Refresh Outputs", command=self._refresh_outputs).grid(row=0, column=3, padx=(0, 8), pady=4)
        ttk.Button(section, text="Open Results Folder", command=self._open_results_folder).grid(row=0, column=4, padx=(0, 8), pady=4)

    def _bind_traces(self):
        watched = (
            self.data_root_var,
            self.results_root_var,
            self.data_name_var,
            self.date_var,
        )
        for variable in watched:
            variable.trace_add("write", lambda *_: self._update_path_preview())

        self.flag_figure_var.trace_add("write", lambda *_: self._update_toggle_states())
        self.flag_svd_var.trace_add("write", lambda *_: self._update_toggle_states())
        self.flag_video_var.trace_add("write", lambda *_: self._update_toggle_states())
        self.probe_name_var.trace_add("write", lambda *_: self._sync_builtin_template_from_label())

    def _load_defaults(self):
        probe_files = self._get_builtin_probe_files()
        self.builtin_probe_combo["values"] = [path.stem for path in probe_files]
        if probe_files:
            preferred = next((path for path in probe_files if path.stem == "L15_80M_128"), probe_files[0])
            self.builtin_probe_var.set(preferred.stem)
            self._load_probe_from_path(preferred)

    def _get_builtin_probe_files(self):
        probe_dir = PROJECT_ROOT / "utils" / "ProbeSettings"
        return sorted(probe_dir.glob("*.json"))

    def _sync_builtin_template_from_label(self):
        builtin_names = set(self.builtin_probe_combo.cget("values"))
        label = self.probe_name_var.get().strip()
        self.builtin_probe_var.set(label if label in builtin_names else "")

    def _load_builtin_probe(self):
        probe_name = self.builtin_probe_var.get().strip()
        if not probe_name:
            messagebox.showinfo("Probe", "Please select a built-in probe template first.")
            return

        probe_path = PROJECT_ROOT / "utils" / "ProbeSettings" / f"{probe_name}.json"
        self._load_probe_from_path(probe_path)

    def _load_probe_from_path(self, probe_path):
        try:
            probe = ProbeSettings.from_json(probe_path)
        except Exception as exc:
            messagebox.showerror("Probe", f"Failed to load probe file:\n{exc}")
            return

        builtin_names = set(self.builtin_probe_combo.cget("values"))
        self.builtin_probe_var.set(probe_path.stem if probe_path.stem in builtin_names else "")
        self.probe_name_var.set(probe_path.stem)
        self.probe_pitch_var.set(self._format_length_mm(probe.pitch))
        self.probe_fs_var.set(self._format_frequency_mhz(probe.fs))
        self.probe_f0_var.set(self._format_frequency_mhz(probe.f0))
        self.probe_toffset_var.set(self._format_time_us(probe.Toffset))
        self.probe_num_samples_var.set(self._format_optional(probe.num_samples))
        self.probe_num_ele_var.set(self._format_optional(probe.num_ele))
        self.probe_ring_angle_var.set(self._format_optional(probe.ring_angle))
        if not self.image_dx_var.get().strip():
            self.image_dx_var.set(self._format_length_mm(probe.res))
        if not self.image_dz_var.get().strip():
            self.image_dz_var.set(self.image_dx_var.get())
        self.status_var.set(f"Loaded probe template: {probe_path} (sound speed ignored)")

    def _import_probe(self):
        file_path = filedialog.askopenfilename(
            title="Import Probe Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self._load_probe_from_path(Path(file_path))

    def _export_probe(self):
        try:
            probe = self._collect_probe_settings()
        except Exception as exc:
            messagebox.showerror("Probe", f"Current probe settings are invalid:\n{exc}")
            return

        initial_name = f"{self.probe_name_var.get().strip() or 'probe'}.json"
        file_path = filedialog.asksaveasfilename(
            title="Export Probe Settings",
            defaultextension=".json",
            initialfile=initial_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        probe.save_json(file_path)
        self.status_var.set(f"Probe settings exported to: {file_path}")

    def _update_path_preview(self):
        data_root = Path(self.data_root_var.get().strip() or ".").expanduser()
        results_root = Path(self.results_root_var.get().strip() or ".").expanduser()
        data_name = self.data_name_var.get().strip() or "<Data_Name>"
        date = self.date_var.get().strip() or "<YYYYMMDD>"

        raw_input = data_root / date / data_name / "unpacked"
        session_data = data_root / date
        results_dir = results_root / date

        self.raw_input_preview_var.set(str(raw_input))
        self.session_data_preview_var.set(str(session_data))
        self.results_preview_var.set(str(results_dir))
        self.output_path_var.set(str(results_dir))

    def _update_toggle_states(self):
        self.figure_dynamic_entry.configure(state=tk.NORMAL if self.flag_figure_var.get() else tk.DISABLED)
        self.svd_num_entry.configure(state=tk.NORMAL if self.flag_svd_var.get() else tk.DISABLED)
        self.svd_dynamic_entry.configure(state=tk.NORMAL if self.flag_svd_var.get() else tk.DISABLED)
        self.video_dynamic_entry.configure(state=tk.NORMAL if self.flag_video_var.get() else tk.DISABLED)

    def _bind_canvas_mousewheel(self, event):
        self.root.bind_all("<MouseWheel>", self._on_canvas_mousewheel)
        self.root.bind_all("<Button-4>", self._on_canvas_mousewheel)
        self.root.bind_all("<Button-5>", self._on_canvas_mousewheel)

    def _unbind_canvas_mousewheel(self, event):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_canvas_mousewheel(self, event):
        if getattr(event, "delta", 0):
            step = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            step = -1
        else:
            step = 1
        self.controls_canvas.yview_scroll(step, "units")

    def _labeled_entry(self, parent, row, column, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=column + 1, sticky="ew", pady=4)


def main():
    enable_high_dpi_awareness()
    root = tk.Tk()
    LinearSVDGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

