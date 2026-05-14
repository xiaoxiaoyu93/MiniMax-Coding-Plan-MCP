"""
MiniMax Coding Plan MCP Server

⚠️ IMPORTANT: This server connects to Minimax API endpoints which may involve costs.
Any tool that makes an API call is clearly marked with a cost warning. Please follow these guidelines:

1. Only use these tools when users specifically ask for them
2. For audio generation tools, be mindful that text length affects the cost
3. Voice cloning features are charged upon first use after cloning

Note: Tools without cost warnings are free to use as they only read existing data.
"""

import os
import json
import secrets
from dotenv import load_dotenv
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from minimax_mcp.utils import (
    process_image_url,
)

from minimax_mcp.const import (
    ENV_FASTMCP_LOG_LEVEL,
    ENV_MINIMAX_API_HOST,
    ENV_MINIMAX_API_KEY,
    ENV_MINIMAX_MCP_BEARER_TOKEN,
    ENV_MINIMAX_MCP_HOST,
    ENV_MINIMAX_MCP_PORT,
    ENV_MINIMAX_MCP_TRANSPORT,
)
from minimax_mcp.exceptions import MinimaxAPIError, MinimaxRequestError
from minimax_mcp.client import MinimaxAPIClient

load_dotenv()
DEFAULT_NETWORK_HOST = "0.0.0.0"
DEFAULT_NETWORK_PORT = 8000
STREAMABLE_HTTP_TRANSPORT = "streamable-http"
TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable": STREAMABLE_HTTP_TRANSPORT,
    STREAMABLE_HTTP_TRANSPORT: STREAMABLE_HTTP_TRANSPORT,
}
STATIC_BEARER_TOKEN_CLIENT_ID = "minimax-static-bearer-token"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def get_transport() -> str:
    raw_transport = os.getenv(ENV_MINIMAX_MCP_TRANSPORT, "stdio").strip().lower()
    transport = TRANSPORT_ALIASES.get(raw_transport)
    if transport is None:
        supported_transports = ", ".join(sorted(TRANSPORT_ALIASES))
        raise ValueError(
            f"Unsupported {ENV_MINIMAX_MCP_TRANSPORT} value: {raw_transport}. "
            f"Supported values: {supported_transports}"
        )
    return transport


def get_listen_host() -> str:
    return os.getenv(ENV_MINIMAX_MCP_HOST, DEFAULT_NETWORK_HOST).strip() or DEFAULT_NETWORK_HOST


def get_listen_port() -> int:
    raw_port = os.getenv(ENV_MINIMAX_MCP_PORT, str(DEFAULT_NETWORK_PORT)).strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"{ENV_MINIMAX_MCP_PORT} must be a valid integer") from exc

    if port < 1 or port > 65535:
        raise ValueError(f"{ENV_MINIMAX_MCP_PORT} must be between 1 and 65535")
    return port


def build_public_server_url(host: str, port: int) -> str:
    public_host = host
    if public_host in {"0.0.0.0", "::", "[::]"}:
        public_host = "127.0.0.1"
    elif ":" in public_host and not public_host.startswith("["):
        public_host = f"[{public_host}]"
    return f"http://{public_host}:{port}"


class StaticBearerTokenVerifier:
    def __init__(self, expected_token: str, resource_url: str):
        self.expected_token = expected_token
        self.resource_url = resource_url

    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token, self.expected_token):
            return None

        return AccessToken(
            token=token,
            client_id=STATIC_BEARER_TOKEN_CLIENT_ID,
            scopes=[],
            resource=self.resource_url,
        )


api_key = get_required_env(ENV_MINIMAX_API_KEY)
api_host = get_required_env(ENV_MINIMAX_API_HOST)
fastmcp_log_level = os.getenv(ENV_FASTMCP_LOG_LEVEL) or "WARNING"
transport = get_transport()
listen_host = get_listen_host()
listen_port = get_listen_port()
bearer_token = os.getenv(ENV_MINIMAX_MCP_BEARER_TOKEN)

fastmcp_settings = {
    "log_level": fastmcp_log_level,
    "host": listen_host,
    "port": listen_port,
}

if bearer_token:
    public_server_url = build_public_server_url(listen_host, listen_port)
    fastmcp_settings["auth"] = AuthSettings(
        issuer_url=public_server_url,
        resource_server_url=public_server_url,
    )
    fastmcp_settings["token_verifier"] = StaticBearerTokenVerifier(
        expected_token=bearer_token,
        resource_url=public_server_url,
    )

mcp = FastMCP("Minimax", **fastmcp_settings)
api_client = MinimaxAPIClient(api_key, api_host)

@mcp.tool(
    description="""
    
    You MUST use this tool whenever you need to search for real-time or external information on the web.
    
    A web search API that works just like Google Search.
    
    Args:
        query (str): The search query. Aim for 3-5 keywords for best results. For time-sensitive topics, include the current date (e.g. `latest iPhone 2025`).
        
    Search Strategy:
        - If no useful results are returned, try rephrasing your query with different keywords.
        
    Returns:
        A JSON object containing the search results, structured as follows:
        {
            "organic": [
                {
                    "title": "string - The title of the search result",
                    "link": "string - The URL link to the result",
                    "snippet": "string - A brief description or excerpt",
                    "date": "string - The date of the result"
                }
            ],
            "related_searches": [
                {
                    "query": "string - A related search query suggestion"
                }
            ],
            "base_resp": {
                "status_code": "int - Response status code",
                "status_msg": "string - Response status message"
            }
        }
    """
)
def minimax_web_search(
    query: str,
) -> TextContent:
    try:
        if not query:
            raise MinimaxRequestError("Query is required")
        
        # Build request payload
        payload = {
            "q": query
        }
        
        # Call search API
        response_data = api_client.post("/v1/coding_plan/search", json=payload)
        
        # Return JSON dump of response data
        return TextContent(
            type="text",
            text=json.dumps(response_data, ensure_ascii=False, indent=2)
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to perform search: {str(e)}"
        )


@mcp.tool(
    description="""
    
    You MUST use this tool whenever you need to analyze, describe, or extract information from an image,
    including when you get an image from user input or any task-related image.

    An LLM-powered vision tool that can analyze and interpret image content from local files or URLs based on your instructions.
    Only JPEG, PNG, and WebP formats are supported. Other formats (e.g. PDF, GIF, PSD, SVG) are not supported.

    Args:
        prompt (str): A text prompt describing what you want to analyze or extract from the image.
        image_source (str): The location of the image to analyze.
            Accepts:
            - HTTP/HTTPS URL: "https://example.com/image.jpg"
            - Local file path:
                - Relative path: "images/photo.png"
                - Absolute path: "/Users/username/Documents/image.jpg"
            IMPORTANT: If the file path starts with an @ symbol, you MUST strip the @ prefix before passing it to this function.
            For example:
                - "@Documents/photo.jpg" → "Documents/photo.jpg"
                - "@/Users/username/image.png" → "/Users/username/image.png"
            Supported formats: JPEG, PNG, WebP
        
    Returns:
        A text description of the image analysis result.
    """
)
def minimax_understand_image(
    prompt: str,
    image_source: str,
) -> TextContent:
    try:
        if not prompt:
            raise MinimaxRequestError("Prompt is required")
        if not image_source:
            raise MinimaxRequestError("Image source is required")
        
        # Process image_source: convert HTTP URL or local file to base64, or pass through existing base64
        processed_image_url = process_image_url(image_source)
        
        # Build request payload
        payload = {
            "prompt": prompt,
            "image_url": processed_image_url
        }
        
        # Call VLM API
        response_data = api_client.post("/v1/coding_plan/vlm", json=payload)
        
        # Extract content from response
        content = response_data.get("content", "")
        
        if not content:
            raise MinimaxRequestError("No content returned from VLM API")
        
        # Return the content
        return TextContent(
            type="text",
            text=content
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to perform VLM analysis: {str(e)}"
        )


def main():
    """Run the Minimax MCP server"""
    print("Starting Minimax MCP server")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
