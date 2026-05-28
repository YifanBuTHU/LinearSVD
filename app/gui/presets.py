import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_preset_dir() -> Path:
    return PROJECT_ROOT / "config" / "gui_presets"


class GuiPresetStore:
    def __init__(self, directory: Optional[Union[Path, str]] = None):
        self.directory = Path(directory).expanduser() if directory is not None else default_preset_dir()

    def list_names(self) -> list[str]:
        if not self.directory.exists():
            return []

        names = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = self._read_payload(path)
            except Exception:
                continue
            name = str(payload.get("name", "")).strip()
            if name:
                names.append(name)
        return sorted(names, key=str.casefold)

    def save(self, name: str, values: dict) -> Path:
        display_name = self._normalize_name(name)
        path = self.path_for_name(display_name)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "name": display_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "gui": dict(values),
        }

        self.directory.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return path

    def load(self, name: str) -> dict:
        path = self.path_for_name(name)
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")
        return self._read_payload(path)

    def delete(self, name: str):
        path = self.path_for_name(name)
        if path.exists():
            path.unlink()

    def exists(self, name: str) -> bool:
        return self.path_for_name(name).exists()

    def path_for_name(self, name: str) -> Path:
        return self.directory / f"{self._safe_file_stem(name)}.json"

    def _read_payload(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported preset schema version in {path}")
        if not isinstance(payload.get("gui"), dict):
            raise ValueError(f"Preset has no GUI settings: {path}")
        return payload

    def _normalize_name(self, name: str) -> str:
        display_name = str(name).strip()
        if not display_name:
            raise ValueError("Preset name cannot be empty.")
        self._safe_file_stem(display_name)
        return display_name

    def _safe_file_stem(self, name: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
        stem = re.sub(r"_+", "_", stem).strip("._-")
        if not stem:
            raise ValueError("Preset name must contain letters or numbers.")
        return stem[:80]
