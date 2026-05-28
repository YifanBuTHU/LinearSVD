from app.gui.runtime import GuiRuntimeMixin


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeProgressBar:
    def __init__(self):
        self.mode = None
        self.started = False

    def configure(self, **kwargs):
        self.mode = kwargs.get("mode", self.mode)

    def start(self, interval):
        self.started = True
        self.interval = interval

    def stop(self):
        self.started = False


def test_progress_update_does_not_override_gui_elapsed_timer():
    gui = GuiRuntimeMixin()
    gui.progress_stage_var = FakeVar()
    gui.progress_message_var = FakeVar()
    gui.elapsed_time_var = FakeVar("00:00:10")
    gui.stage_progress_bar = FakeProgressBar()
    gui.stage_progress_value_var = FakeVar()
    gui.stage_progress_text_var = FakeVar()
    gui.overall_progress_value_var = FakeVar()
    gui.overall_progress_text_var = FakeVar()

    gui._apply_progress_update(
        {
            "stage": "stream_pipeline",
            "message": "Processing and reconstructing frames",
            "current": 1,
            "total": 10,
            "elapsed_seconds": 1.0,
            "overall_current": 1,
            "overall_total": 10,
        }
    )

    assert gui.elapsed_time_var.get() == "00:00:10"
