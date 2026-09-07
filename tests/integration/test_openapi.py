"""OpenAPI security and schema tests."""

from fastapi.testclient import TestClient

from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import make_settings


def test_openapi_documents_authentication_boundary(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["components"]["securitySchemes"]["ApiKeyHeader"] == {
        "type": "apiKey",
        "description": "Deployment-issued API key.",
        "in": "header",
        "name": "X-API-Key",
    }
    assert schema["paths"]["/v1/capabilities"]["get"]["security"] == [{"ApiKeyHeader": []}]
    assert "security" not in schema["paths"]["/health/live"]["get"]


def test_openapi_exposes_problem_and_health_schemas(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    schemas = document["components"]["schemas"]

    assert "ProblemDetail" in schemas
    assert "HealthResponse" in schemas
    assert "CapabilitiesResponse" in schemas
    assert document["paths"]["/v1/capabilities"]["get"]["responses"]["429"] == {
        "description": "Caller rate limit exceeded.",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ProblemDetail"}}},
    }


def test_docs_and_openapi_can_be_disabled() -> None:
    app = create_app(make_settings(docs_enabled=False))

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_operator_metrics_endpoint_is_hidden_from_openapi() -> None:
    settings = make_settings().model_copy(update={"metrics_enabled": True})
    app = create_app(settings)

    with TestClient(app) as client:
        document = client.get("/openapi.json").json()

    assert "/metrics" not in document["paths"]
