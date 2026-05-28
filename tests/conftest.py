import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    """FastAPI test client with mocked agent_manager and DB."""
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
    with patch("api.agent_manager.agent_manager"):
        from app import app
        from core.database import get_db

        mock_session = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_session

        with TestClient(app, raise_server_exceptions=True) as c:
            c.mock_db = mock_session
            yield c

        app.dependency_overrides.clear()
