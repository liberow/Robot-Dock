from robot_dock.cli.interaction import next_instruction


def test_window_keeps_processing_events_until_instruction_arrives(monkeypatch):
    calls = []

    def tick():
        calls.append("sync")
        return True

    def read(prompt):
        return "Place the red cube on the table."

    monkeypatch.setattr("builtins.input", read)
    assert next_instruction(tick) == "Place the red cube on the table."
    assert len(calls) >= 2


def test_closed_viewer_exits_without_waiting_for_console_input(monkeypatch):
    def forbidden(prompt):
        raise AssertionError("Must not prompt after window closed")

    monkeypatch.setattr("builtins.input", forbidden)
    assert next_instruction(lambda: False) is None


def test_quit_ends_session(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")
    assert next_instruction(lambda: True) is None


def test_uncertain_execution_keeps_viewer_open_but_cannot_start_next_task(monkeypatch):
    answers = iter(["", "move", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert next_instruction(lambda: True, allow_next=False) is None
    assert list(answers) == []


def test_eof_ends_session(monkeypatch):
    def eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert next_instruction(lambda: True) is None
