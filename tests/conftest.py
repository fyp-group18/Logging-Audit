import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def _require_gcp_project():
    """Skip rather than error when the app configuration cannot be imported.

    core.config raises at import time without GOOGLE_CLOUD_PROJECT, which turns
    every client-backed test into a collection error for anyone who has not set
    it. Skipping keeps the remaining suite runnable from a clean checkout.
    """
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("GOOGLE_CLOUD_PROJECT is not set — app configuration unavailable")


@pytest.fixture
def client():
    """FastAPI test client with mocked agent_manager and DB."""
    _require_gcp_project()
    with patch("api.agent_manager.agent_manager"):
        from app import app
        from core.database import get_db

        mock_session = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_session

        with TestClient(app, raise_server_exceptions=True) as c:
            c.mock_db = mock_session
            yield c

        app.dependency_overrides.clear()


@pytest.fixture
def senior_client():
    """FastAPI test client for senior-level test scenarios."""
    _require_gcp_project()
    with patch("api.agent_manager.agent_manager"):
        from app import app
        from core.database import get_db

        mock_session = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_session

        with TestClient(app, raise_server_exceptions=True) as c:
            c.mock_db = mock_session
            yield c

        app.dependency_overrides.clear()


@pytest.fixture
def junior_client():
    """FastAPI test client for junior-level test scenarios."""
    _require_gcp_project()
    with patch("api.agent_manager.agent_manager"):
        from app import app
        from core.database import get_db

        mock_session = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_session

        with TestClient(app, raise_server_exceptions=True) as c:
            c.mock_db = mock_session
            yield c

        app.dependency_overrides.clear()
