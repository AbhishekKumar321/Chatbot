"""Tests for the FastAPI application entrypoint."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_ingest_rejects_non_pdf_upload() -> None:
    response = client.post(
        "/ingest",
        files={"file": ("A mechanic.docx", b"not a pdf", "application/vnd.openxmlformats")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"
