import locale
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from app.gui.shared import PROJECT_ROOT
from app.progress_protocol import parse_progress_line
from utils.Probe import ProbeSettings
from utils.config import PathConfig, ReconParams
from utils.runtime_io import save_runtime_bundle


class GuiRuntimeMixin:
    def _build_output_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")

        log_tab = ttk.Frame(notebook, padding=8)
        output_tab = ttk.Frame(notebook, padding=8)
        notebook.add(log_tab, text="System Output")
        notebook.add(output_tab, text="Generated Files")

        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(5, weight=1)

        progress_frame = ttk.LabelFrame(log_tab, text="Run Progress", padding=8)
        progress_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(1, weight=1)

        ttk.Label(progress_frame, text="Stage").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(progress_frame, textvariable=self.progress_stage_var).grid(row=0, column=1, sticky="w")
        ttk.Label(progress_frame, text="Elapsed").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Label(progress_frame, textvariable=self.elapsed_time_var).grid(row=0, column=3, sticky="w")

        ttk.Label(progress_frame, text="Message").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=(6, 0))
        ttk.Label(
            progress_frame,
            textvariable=self.progress_message_var,
            justify="left",
            wraplength=420,
        ).grid(row=1, column=1, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(log_tab, text="Overall Progress").grid(row=1, column=0, sticky="w")
        self.overall_progress_bar = ttk.Progressbar(
            log_tab,
            variable=self.overall_progress_value_var,
            maximum=100,
            mode="determinate",
        )
        self.overall_progress_bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(log_tab, textvariable=self.overall_progress_text_var).grid(row=3, column=0, sticky="w", pady=(4, 4))

        stage_frame = ttk.Frame(log_tab)
        stage_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        stage_frame.columnconfigure(0, weight=1)

        ttk.Label(stage_frame, text="Current Stage Progress").grid(row=0, column=0, sticky="w")
        self.stage_progress_bar = ttk.Progressbar(
            stage_frame,
            variable=self.stage_progress_value_var,
            maximum=100,
            mode="determinate",
        )
        self.stage_progress_bar.grid(row=1, column=0, sticky="ew")
        ttk.Label(stage_frame, textvariable=self.stage_progress_text_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.log_text = ScrolledText(log_tab, wrap=tk.WORD, height=20)
        self.log_text.grid(row=5, column=0, sticky="nsew")

        output_tab.columnconfigure(0, weight=1)
        output_tab.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(output_tab)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="Current Output").grid(row=0, column=0, sticky="w")
        ttk.Label(toolbar, textvariable=self.output_path_var).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(toolbar, text="Open Selected", command=self._open_selected_output).grid(row=0, column=2, padx=(12, 0))

        file_preview_pane = tk.PanedWindow(
            output_tab,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RIDGE,
            showhandle=True,
            bg="#cccccc",
        )
        file_preview_pane.grid(row=1, column=0, sticky="nsew")

        left_frame = ttk.Frame(file_preview_pane)
        file_preview_pane.add(left_frame, minsize=150)
        right_frame = ttk.Frame(file_preview_pane)
        file_preview_pane.add(right_frame, minsize=300)

        self.output_listbox = tk.Listbox(left_frame, exportselection=False)
        self.output_listbox.pack(fill=tk.BOTH, expand=True)
        self.output_listbox.bind("<<ListboxSelect>>", self._on_output_selected)
        self.output_listbox.bind("<Double-Button-1>", lambda event: self._open_selected_output())

        preview_frame = ttk.LabelFrame(right_frame, text="Preview", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        self.preview_path_label = ttk.Label(preview_frame, text="", wraplength=500, justify="left")
        self.preview_path_label.grid(row=0, column=0, sticky="w")

        self.preview_label = ttk.Label(preview_frame, text="Select a generated file to preview.")
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", self._on_preview_resize)

    def _browse_directory(self, variable):
        directory = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_ROOT))
        if directory:
            variable.set(directory)

    def _browse_file(self, variable):
        file_path = filedialog.askopenfilename(
            initialdir=str(PROJECT_ROOT),
            filetypes=[("Executable files", "*.bat;*.cmd;*.exe"), ("All files", "*.*")],
        )
        if file_path:
            variable.set(file_path)

    def _start_run(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Run", "A run is already in progress.")
            return

        try:
            paths, recon_para, probe = self._collect_runtime_bundle()
        except Exception as exc:
            messagebox.showerror("Run", str(exc))
            return

        fd, config_path = tempfile.mkstemp(prefix="linear_svd_gui_", suffix=".json")
        os.close(fd)
        self.temp_config_path = config_path
        save_runtime_bundle(config_path, paths, recon_para, probe)

        command, display_command = self._build_run_command(config_path)

        self.log_text.delete("1.0", tk.END)
        self._append_log(f"Launching: {display_command}\n")
        self.status_var.set("Run started...")
        self.progress_stage_var.set("starting")
        self.progress_message_var.set("Preparing runtime bundle")
        self.stage_progress_value_var.set(0.0)
        self.stage_progress_text_var.set("0 / 0")
        self.overall_progress_value_var.set(0.0)
        self.overall_progress_text_var.set("0%")
        self.elapsed_time_var.set("00:00:00")
        self.run_started_at = time.perf_counter()
        self._update_elapsed_time()

        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=self._get_subprocess_encoding(),
            errors="replace",
            env=self._build_subprocess_env(),
        )

        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self):
        assert self.process is not None
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self.log_queue.put(("log", line))
        return_code = self.process.wait()
        self.log_queue.put(("done", return_code))

    def _stop_run(self):
        if self.process is None or self.process.poll() is not None:
            return

        self._append_log("Stopping current run...\n")
        self.status_var.set("Stopping run...")
        self.progress_message_var.set("Stopping current run")
        self._terminate_process_tree()

    def _validate_only(self):
        try:
            paths, recon_para, probe = self._collect_runtime_bundle()
            recon_para.apply_to_probe(probe)
            recon_para.validate(probe, paths)
        except Exception as exc:
            messagebox.showerror("Validate", str(exc))
            return

        summary = (
            "Validation passed.\n\n"
            f"Raw input: {paths.unpacked_dir(recon_para.Date, recon_para.Data_Name)}\n"
            f"Session data: {paths.session_data_dir(recon_para.Date)}\n"
            f"Results: {paths.result_date_dir(recon_para.Date)}"
        )
        messagebox.showinfo("Validate", summary)
        self.status_var.set("Validation passed.")

    def _collect_runtime_bundle(self):
        paths = PathConfig(
            data_root=self.data_root_var.get().strip(),
            results_root=self.results_root_var.get().strip(),
        )
        probe = self._collect_probe_settings()
        recon_para = self._collect_recon_params()

        recon_para.apply_to_probe(probe)
        recon_para.validate(probe, paths)
        return paths, recon_para, probe

    def _collect_probe_settings(self):
        probe = ProbeSettings(
            vc=self._parse_float(self.runtime_vc_var.get(), "Sound Speed"),
            pitch=self._parse_length_mm(self.probe_pitch_var.get(), "Probe pitch"),
            fs=self._parse_frequency_mhz(self.probe_fs_var.get(), "Probe fs"),
            f0=self._parse_frequency_mhz(self.probe_f0_var.get(), "Probe f0"),
            Toffset=self._parse_time_us(self.probe_toffset_var.get(), "Probe Toffset"),
            num_samples=self._parse_int(self.probe_num_samples_var.get(), "Probe num_samples"),
            ring_angle=self._parse_optional_float(self.probe_ring_angle_var.get(), "Probe ring_angle"),
            num_ele=self._parse_int(self.probe_num_ele_var.get(), "Probe num_ele"),
        )
        probe.validate()
        return probe

    def _collect_recon_params(self):
        angle_start = self._parse_int(self.angle_start_var.get(), "Angle start")
        angle_end = self._parse_int(self.angle_end_var.get(), "Angle end")
        angle_step = self._parse_int(self.angle_step_var.get(), "Angle step")
        angles = self._build_angle_array(angle_start, angle_end, angle_step)

        return ReconParams(
            Data_Name=self._required_text(self.data_name_var.get(), "Data Name"),
            Data_Type=self._required_text(self.data_type_var.get(), "Data Type"),
            angles=angles,
            Module=self.module_var.get().strip().upper(),
            Date=self._required_text(self.date_var.get(), "Date"),
            Probe=self._required_text(self.probe_name_var.get(), "Probe Label"),
            frame_num=self._parse_int(self.frame_num_var.get(), "Frame Count"),
            SVD_num=self._parse_int(self.svd_num_var.get(), "SVD Num"),
            idx=0,
            segment_Nx=self._parse_int(self.segment_nx_var.get(), "segment_Nx"),
            dynamic_range_figure=self._parse_int(self.dynamic_range_figure_var.get(), "Figure Dynamic Range"),
            dynamic_range_svd=self._parse_int(self.dynamic_range_svd_var.get(), "SVD Dynamic Range"),
            dynamic_range_video=self._parse_int(self.dynamic_range_video_var.get(), "Video Dynamic Range"),
            vc=self._parse_float(self.runtime_vc_var.get(), "Runtime vc"),
            image_nx=self._parse_int(self.image_nx_var.get(), "Image Nx"),
            image_nz=self._parse_int(self.image_nz_var.get(), "Image Nz"),
            image_dx=self._parse_optional_length_mm(self.image_dx_var.get(), "Image dx"),
            image_dz=self._parse_optional_length_mm(self.image_dz_var.get(), "Image dz"),
            zcenter=self._parse_length_mm(self.zcenter_var.get(), "zcenter"),
            gpu_device=self._parse_optional_int(self.gpu_device_var.get(), "GPU Device", default=0),
            toffset_correction_samples=self._parse_int(self.toffset_correction_var.get(), "Toffset Correction"),
            flag_Data_Process=self.flag_data_process_var.get(),
            flag_Recon=self.flag_recon_var.get(),
            flag_SVD=self.flag_svd_var.get(),
            flag_Figure=self.flag_figure_var.get(),
            flag_Video=self.flag_video_var.get(),
            flag_SVD_Video=self.flag_svd_video_var.get(),
            flag_SVD_Frames=self.flag_svd_frames_var.get(),
            flag_sensitivity=self.flag_sensitivity_var.get(),
        )

    def _build_angle_array(self, start, end, step):
        if step == 0:
            raise ValueError("Angle step cannot be zero.")
        if step > 0 and end < start:
            raise ValueError("Angle end must be >= start when step is positive.")
        if step < 0 and end > start:
            raise ValueError("Angle end must be <= start when step is negative.")
        return np.arange(start, end + (1 if step > 0 else -1), step, dtype=int)

    def _build_run_command(self, config_path):
        conda_executable = self._resolve_conda_executable(self.conda_executable_var.get())
        conda_env = self._required_text(self.conda_env_var.get(), "Conda Env")
        job_script = PROJECT_ROOT / "run_pipeline_job.py"

        if Path(conda_executable).suffix.lower() in {".bat", ".cmd"}:
            shell_command = (
                f'call "{conda_executable}" run --no-capture-output '
                f'-n "{conda_env}" python -u "{job_script}" "{config_path}"'
            )
            return ["cmd.exe", "/d", "/c", shell_command], f'cmd.exe /d /c {shell_command}'

        command = [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-u",
            str(job_script),
            config_path,
        ]
        return command, " ".join(command)

    def _resolve_conda_executable(self, raw_value):
        text = self._required_text(raw_value, "Conda Executable")
        resolved = shutil.which(text)
        if resolved:
            text = resolved
        path = Path(text).expanduser()
        if path.exists():
            if path.suffix.lower() in {".bat", ".cmd"}:
                exe_candidate = path.parents[1] / "Scripts" / "conda.exe"
                if exe_candidate.exists():
                    return str(exe_candidate.resolve())
            return str(path.resolve())
        raise ValueError(f"Conda executable not found: {text}")

    def _build_subprocess_env(self):
        return os.environ.copy()

    def _get_subprocess_encoding(self):
        if os.name == "nt":
            return locale.getpreferredencoding(False) or "mbcs"
        return locale.getpreferredencoding(False) or "utf-8"

    def _refresh_outputs(self):
        results_dir = Path(self.results_root_var.get().strip() or ".") / (self.date_var.get().strip() or "")
        self.output_files = self._gather_output_files(results_dir)

        self.output_listbox.delete(0, tk.END)
        for path in self.output_files:
            label = str(path.relative_to(results_dir))
            self.output_listbox.insert(tk.END, label)

        self.output_path_var.set(str(results_dir))
        if self.output_files:
            latest_png_index = self._latest_png_index()
            target_index = latest_png_index if latest_png_index is not None else len(self.output_files) - 1
            self.output_listbox.selection_clear(0, tk.END)
            self.output_listbox.selection_set(target_index)
            self.output_listbox.see(target_index)
            self._show_output_preview(self.output_files[target_index])
        else:
            self.preview_path_label.configure(text="")
            self.preview_label.configure(text="No generated files found yet.", image="")
            self.preview_image = None
            self._preview_pil_image = None

    def _gather_output_files(self, results_dir):
        if not results_dir.exists():
            return []

        files = [path for path in sorted(results_dir.glob("*")) if path.is_file()]
        png_dir = results_dir / "png"
        if png_dir.exists():
            files.extend(path for path in sorted(png_dir.glob("*")) if path.is_file())
        svd_frame_dir = results_dir / "svd_frames"
        if svd_frame_dir.exists():
            files.extend(path for path in sorted(svd_frame_dir.glob("*.npy")) if path.is_file())
        return files

    def _latest_png_index(self):
        png_candidates = [(idx, path) for idx, path in enumerate(self.output_files) if path.suffix.lower() == ".png"]
        if not png_candidates:
            return None
        return max(png_candidates, key=lambda item: item[1].stat().st_mtime)[0]

    def _on_output_selected(self, event):
        selection = self.output_listbox.curselection()
        if not selection:
            return
        self._show_output_preview(self.output_files[selection[0]])

    def _show_output_preview(self, path):
        self.preview_path_label.configure(text=str(path))
        self._preview_source_path = path

        if path.suffix.lower() != ".png":
            self.preview_image = None
            self._preview_pil_image = None
            self.preview_label.configure(
                text=f"Preview is available for PNG files.\nSelected: {path.name}",
                image="",
            )
            return

        if _HAS_PIL:
            try:
                self._preview_pil_image = Image.open(str(path))
                self._update_preview_size()
            except Exception as exc:
                self.preview_image = None
                self._preview_pil_image = None
                self.preview_label.configure(text=f"Failed to load preview:\n{exc}", image="")
        else:
            try:
                image = tk.PhotoImage(file=str(path))
                scale = max((image.width() + 719) // 720, (image.height() + 719) // 720, 1)
                if scale > 1:
                    image = image.subsample(scale, scale)
                self.preview_image = image
                self.preview_label.configure(image=self.preview_image, text="")
            except Exception as exc:
                self.preview_image = None
                self.preview_label.configure(text=f"Failed to load preview:\n{exc}", image="")

    def _on_preview_resize(self, event=None):
        if _HAS_PIL and getattr(self, "_preview_pil_image", None) is not None:
            self._update_preview_size()

    def _update_preview_size(self):
        pil_image = getattr(self, "_preview_pil_image", None)
        if pil_image is None:
            return

        frame_width = max(self.preview_label.winfo_width(), 1)
        frame_height = max(self.preview_label.winfo_height(), 1)
        if frame_width <= 1 or frame_height <= 1:
            self.preview_label.after(100, self._update_preview_size)
            return

        img_w, img_h = pil_image.size
        scale = min(frame_width / img_w, frame_height / img_h)
        new_w = max(int(img_w * scale), 1)
        new_h = max(int(img_h * scale), 1)

        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_label.configure(image=self.preview_image, text="")

    def _open_results_folder(self):
        results_dir = Path(self.results_root_var.get().strip() or ".") / (self.date_var.get().strip() or "")
        results_dir.mkdir(parents=True, exist_ok=True)
        self._open_path(results_dir)

    def _open_selected_output(self):
        selection = self.output_listbox.curselection()
        if not selection:
            messagebox.showinfo("Output", "Please select a generated file first.")
            return
        self._open_path(self.output_files[selection[0]])

    def _open_path(self, path):
        path = Path(path)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open Path", f"Failed to open path:\n{exc}")

    def _read_queue_item(self, item):
        kind, payload = item
        if kind == "log":
            progress_payload = parse_progress_line(payload.strip())
            if progress_payload is not None:
                self._apply_progress_update(progress_payload)
            else:
                self._append_log(payload)
        elif kind == "done":
            self._handle_process_done(payload)

    def _poll_queue(self):
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._read_queue_item(item)
        self.root.after(120, self._poll_queue)

    def _apply_progress_update(self, payload):
        stage = payload.get("message") or payload.get("stage") or "Running"
        current = int(payload.get("current", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        indeterminate = bool(payload.get("indeterminate", False))
        overall_current = float(payload.get("overall_current", current or 0))
        overall_total = float(payload.get("overall_total", total or 1))

        self.progress_stage_var.set(payload.get("stage") or "running")
        self.progress_message_var.set(stage)

        if indeterminate or total <= 0:
            self.stage_progress_bar.configure(mode="indeterminate")
            self.stage_progress_bar.start(12)
            self.stage_progress_text_var.set(stage)
        else:
            self.stage_progress_bar.stop()
            self.stage_progress_bar.configure(mode="determinate")
            self.stage_progress_value_var.set(min(max(current / total * 100, 0), 100))
            self.stage_progress_text_var.set(f"{current} / {total}")

        overall_total = max(overall_total, 1.0)
        overall_percent = min(max(overall_current / overall_total * 100, 0), 100)
        self.overall_progress_value_var.set(overall_percent)
        self.overall_progress_text_var.set(f"{overall_percent:.1f}% ({overall_current:.1f} / {overall_total:.1f})")

    def _update_elapsed_time(self):
        if self.run_started_at is None:
            return
        elapsed_seconds = int(time.perf_counter() - self.run_started_at)
        self.elapsed_time_var.set(self._format_elapsed_time(elapsed_seconds))
        self.root.after(250, self._update_elapsed_time)

    def _format_elapsed_time(self, total_seconds):
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _handle_process_done(self, return_code):
        if return_code == 0:
            self.status_var.set("Run completed successfully.")
            self.progress_stage_var.set("completed")
            self.progress_message_var.set("Run completed")
            self.stage_progress_bar.stop()
            self.stage_progress_bar.configure(mode="determinate")
            self.stage_progress_value_var.set(100.0)
            self.stage_progress_text_var.set("Completed")
            self.overall_progress_value_var.set(100.0)
            self.overall_progress_text_var.set("100.0%")
        else:
            self.status_var.set(f"Run finished with exit code {return_code}.")
            self.progress_stage_var.set("failed")
            self.progress_message_var.set(f"Run finished with exit code {return_code}")
            self.stage_progress_bar.stop()
            self.stage_progress_bar.configure(mode="determinate")

        self.process = None
        self.run_started_at = None
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self._cleanup_temp_config()
        self._refresh_outputs()

    def _append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _required_text(self, value, field_name):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty.")
        return text

    def _parse_int(self, value, field_name):
        try:
            return int(str(value).strip())
        except Exception as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc

    def _parse_optional_int(self, value, field_name, default=None):
        text = str(value).strip()
        if not text:
            return default
        return self._parse_int(text, field_name)

    def _parse_float(self, value, field_name):
        try:
            return float(str(value).strip())
        except Exception as exc:
            raise ValueError(f"{field_name} must be a number.") from exc

    def _parse_optional_float(self, value, field_name):
        text = str(value).strip()
        if not text:
            return None
        return self._parse_float(text, field_name)

    def _format_optional(self, value):
        if value is None:
            return ""
        return str(value)

    def _parse_frequency_mhz(self, value, field_name):
        return self._parse_float(value, field_name) * 1e6

    def _parse_time_us(self, value, field_name):
        return self._parse_float(value, field_name) * 1e-6

    def _parse_length_mm(self, value, field_name):
        return self._parse_float(value, field_name) * 1e-3

    def _parse_optional_length_mm(self, value, field_name):
        text = str(value).strip()
        if not text:
            return None
        return self._parse_length_mm(text, field_name)

    def _format_frequency_mhz(self, value):
        if value is None:
            return ""
        return self._format_number(value / 1e6)

    def _format_time_us(self, value):
        if value is None:
            return ""
        return self._format_number(value * 1e6)

    def _format_length_mm(self, value):
        if value is None:
            return ""
        return self._format_number(value * 1e3)

    def _format_number(self, value):
        return f"{value:.6g}"

    def _cleanup_temp_config(self):
        if not self.temp_config_path:
            return
        try:
            Path(self.temp_config_path).unlink(missing_ok=True)
        except Exception:
            pass
        self.temp_config_path = None

    def _on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Exit", "A run is still active. Stop it and close the GUI?"):
                return
            self._terminate_process_tree()
        self._cleanup_temp_config()
        self.root.destroy()

    def _terminate_process_tree(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self.process.terminate()
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass
