"""``POST /api/model/set`` accepts the ``voice_brief`` auxiliary slot over REST.

The allowlist itself is pinned in ``test_aux_slot_allowlist.py``; this module keeps the
end-to-end case (route -> config write) that used to live in ``test_web_server.py``, which is
under a line-count ratchet and only takes *new* topical modules.
"""

import pytest

# Imported at module scope, like ``test_web_server.py`` does: building the app pulls in
# fastapi/pydantic, whose plugin loader reads installed-distribution metadata. Doing that
# inside a test would trip the home-I/O guard (it reads the real hermes home's egg-info).
from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app


class TestAuxVoiceBriefSlotOverRest:
    """The dashboard REST write path for one auxiliary slot.

    The client fixture mirrors ``TestWebServerEndpoints._setup_test_client`` in
    ``test_web_server.py``: a Starlette ``TestClient`` over the real ``hermes_cli.web_server``
    app with the session token header, so the POST is authorized exactly like the dashboard's.
    Duplicated rather than imported because importing that module would re-collect its whole
    suite here.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        """Create a TestClient and isolate the state DB under the test HERMES_HOME."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_model_set_aux_accepts_the_voice_brief_slot(self):
        """``POST /api/model/set`` accepts every slot ``hermes model`` offers.

        ``auxiliary.voice_brief`` shipped in DEFAULT_CONFIG and the CLI picker, but
        ``_AUX_TASK_SLOTS`` (the REST allowlist) was never extended, so the API answered 400
        ``unknown auxiliary task: voice_brief`` and the Web/Desktop pickers had no row.
        """
        from hermes_cli.config import load_config

        resp = self.client.post(
            "/api/model/set",
            json={
                "scope": "auxiliary",
                "provider": "openrouter",
                "model": "probe/model-x",
                "task": "voice_brief",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["tasks"] == ["voice_brief"]

        # A non-default pin: DEFAULT_CONFIG ships provider="auto"/model="", and load_config()
        # deep-merges the defaults, so asserting those would pass even with the write dropped.
        stored = load_config()["auxiliary"]["voice_brief"]
        assert stored["provider"] == "openrouter"
        assert stored["model"] == "probe/model-x"
