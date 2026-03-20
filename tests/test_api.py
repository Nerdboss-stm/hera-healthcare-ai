import pytest
from unittest.mock import patch, MagicMock


class TestFastAPIEndpoints:
    """Tests for the serving API layer."""

    def test_root_endpoint(self):
        """Test the health check endpoint returns 200."""
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_metrics_endpoint(self):
        """Test the Prometheus metrics endpoint returns 200."""
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200

    @patch("serving.api.generate_summary")
    @patch("serving.api.log_to_db")
    def test_summarize_endpoint(self, mock_log, mock_summary):
        """Test the /summarize endpoint with a mocked model."""
        mock_summary.return_value = "Patient has chest pain. Started treatment."
        mock_log.return_value = None

        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/summarize",
            json={"note": "Patient is a 65-year-old male presenting with chest pain."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "timestamp" in data

    def test_summarize_endpoint_requires_body(self):
        """Test that /summarize rejects requests without a body."""
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/summarize")
        assert response.status_code == 422
