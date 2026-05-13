import anyio
import importlib
import sys

import pytest

MODULE_NAME = "minimax_mcp.server"
SERVER_ENV_VARS = (
    "MINIMAX_API_KEY",
    "MINIMAX_API_HOST",
    "MINIMAX_MCP_TRANSPORT",
    "MINIMAX_MCP_HOST",
    "MINIMAX_MCP_PORT",
    "MINIMAX_MCP_BEARER_TOKEN",
)


@pytest.fixture
def load_server_module(monkeypatch):
    def _load(**env_overrides):
        for env_name in SERVER_ENV_VARS:
            monkeypatch.delenv(env_name, raising=False)

        env = {
            "MINIMAX_API_KEY": "test-api-key",
            "MINIMAX_API_HOST": "https://api.minimax.io",
            **env_overrides,
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        sys.modules.pop(MODULE_NAME, None)
        module = importlib.import_module(MODULE_NAME)
        return importlib.reload(module)

    yield _load
    sys.modules.pop(MODULE_NAME, None)


def test_main_supports_streamable_transport_alias(load_server_module, monkeypatch):
    server = load_server_module(MINIMAX_MCP_TRANSPORT="streamable")
    transports = []
    monkeypatch.setattr(server.mcp, "run", lambda transport="stdio": transports.append(transport))

    server.main()

    assert transports == ["streamable-http"]


def test_invalid_transport_raises_value_error(load_server_module):
    with pytest.raises(ValueError, match="Unsupported MINIMAX_MCP_TRANSPORT value"):
        load_server_module(MINIMAX_MCP_TRANSPORT="invalid")


def test_network_settings_and_bearer_token_are_applied(load_server_module):
    server = load_server_module(
        MINIMAX_MCP_TRANSPORT="sse",
        MINIMAX_MCP_HOST="127.0.0.1",
        MINIMAX_MCP_PORT="9001",
        MINIMAX_MCP_BEARER_TOKEN="secret-token",
    )

    assert server.transport == "sse"
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.settings.port == 9001
    assert server.mcp.settings.auth is not None
    assert str(server.mcp.settings.auth.resource_server_url) == "http://127.0.0.1:9001/"

    verifier = server.StaticBearerTokenVerifier(
        expected_token="secret-token",
        resource_url="http://127.0.0.1:9001",
    )
    auth_info = anyio.run(verifier.verify_token, "secret-token")
    invalid_auth_info = anyio.run(verifier.verify_token, "wrong-token")

    assert auth_info is not None
    assert auth_info.client_id == server.STATIC_BEARER_TOKEN_CLIENT_ID
    assert invalid_auth_info is None
