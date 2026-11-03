import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from robot_dock.visualization import PreviewState, WebViewer


@pytest.fixture
def viewer():
    state = PreviewState()
    server = WebViewer(state, port=0)
    server.start()
    yield state, f"http://127.0.0.1:{server.port}"
    server.close()


def post(url, value, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    return urlopen(Request(url, json.dumps(value).encode(), headers), timeout=3)


def test_latest_frames_and_status_are_served(viewer):
    state, url = viewer
    pixels = np.full((10, 20, 3), 100, dtype=np.uint8)
    state.publish(pixels, pixels, phase="executing")
    with urlopen(url, timeout=3) as response:
        assert b"Robot Dock" in response.read()
    with urlopen(url + "/status", timeout=3) as response:
        assert json.load(response)["phase"] == "executing"
    with urlopen(url + "/frame/wrist.jpg", timeout=3) as response:
        assert Image.open(io.BytesIO(response.read())).size == (20, 10)


def test_camera_commands_are_queued_not_executed_on_server_thread(viewer):
    state, url = viewer
    with post(url + "/camera", {"azimuth": 90}, origin=url) as response:
        assert response.status == 202
    with post(url + "/camera", {"distance": 2}, origin=url):
        pass
    assert state.take_camera() == {"azimuth": 90.0, "distance": 2.0}
    assert state.take_camera() is None


@pytest.mark.parametrize(
    "value", [{"distance": -1}, {"azimuth": float("nan")}, {"action": "move"}, []]
)
def test_invalid_controls_are_rejected(viewer, value):
    state, url = viewer
    with pytest.raises(HTTPError) as error:
        post(url + "/camera", value)
    assert error.value.code == 400
    assert state.take_camera() is None


def test_cross_origin_camera_request_is_rejected(viewer):
    state, url = viewer
    with pytest.raises(HTTPError) as error:
        post(url + "/camera", {"distance": 1}, origin="http://another-site.invalid")
    assert error.value.code == 403 and state.take_camera() is None


def test_finished_viewer_keeps_frames_but_does_not_accept_camera_motion(viewer):
    state, url = viewer
    state.update(phase="finished")
    with pytest.raises(HTTPError) as error:
        post(url + "/camera", {"distance": 1})
    assert error.value.code == 409
    with urlopen(url + "/status", timeout=3) as response:
        assert json.load(response)["phase"] == "finished"


def test_no_frame_and_unknown_endpoint_are_explicit(viewer):
    state, url = viewer
    for path, expected in [("/frame/wrist.jpg", 503), ("/stop", 404)]:
        with pytest.raises(HTTPError) as error:
            urlopen(url + path, timeout=3)
        assert error.value.code == expected
