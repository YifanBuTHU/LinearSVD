import json

import pytest

from app.gui.main_window import LinearSVDGui
from app.gui.presets import GuiPresetStore, default_preset_dir
from tests.test_gui_runtime import FakeVar


def test_default_preset_dir_uses_project_config_dir():
    assert default_preset_dir().name == "gui_presets"
    assert default_preset_dir().parent.name == "config"


def test_preset_store_roundtrips_and_deletes_values(tmp_path):
    store = GuiPresetStore(tmp_path)
    values = {
        "data_root": r"E:\Study\Data",
        "results_root": r"E:\Study\Results",
        "probe_pitch": "0.1",
        "flag_svd": False,
    }

    saved_path = store.save("Kidney Default", values)

    assert saved_path.name == "Kidney_Default.json"
    assert store.list_names() == ["Kidney Default"]
    payload = store.load("Kidney Default")
    assert payload["schema_version"] == 1
    assert payload["name"] == "Kidney Default"
    assert payload["gui"] == values

    store.delete("Kidney Default")

    assert store.list_names() == []
    assert not saved_path.exists()


def test_preset_store_rejects_empty_or_unsafe_names(tmp_path):
    store = GuiPresetStore(tmp_path)

    with pytest.raises(ValueError, match="Preset name cannot be empty"):
        store.save("", {})
    with pytest.raises(ValueError, match="Preset name must contain"):
        store.save("///", {})


def test_gui_preset_values_roundtrip_all_declared_fields():
    gui = LinearSVDGui.__new__(LinearSVDGui)
    for _, variable_name in LinearSVDGui.GUI_PRESET_FIELDS:
        setattr(gui, variable_name, FakeVar(f"initial-{variable_name}"))

    gui.flag_svd_var.set(True)
    gui.flag_video_var.set(False)

    values = gui._collect_gui_preset_values()

    assert set(values) == {key for key, _ in LinearSVDGui.GUI_PRESET_FIELDS}
    assert values["flag_svd"] is True
    assert values["flag_video"] is False

    applied = {key: f"loaded-{key}" for key, _ in LinearSVDGui.GUI_PRESET_FIELDS}
    applied["flag_svd"] = False
    applied["flag_video"] = True
    gui._apply_gui_preset_values(applied, refresh=False)

    assert gui.flag_svd_var.get() is False
    assert gui.flag_video_var.get() is True
    assert gui.data_root_var.get() == "loaded-data_root"


def test_preset_store_overwrites_same_name(tmp_path):
    store = GuiPresetStore(tmp_path)
    store.save("Same Name", {"data_root": "first"})
    store.save("Same Name", {"data_root": "second"})

    assert store.list_names() == ["Same Name"]
    assert store.load("Same Name")["gui"]["data_root"] == "second"
    assert len(list(tmp_path.glob("*.json"))) == 1
