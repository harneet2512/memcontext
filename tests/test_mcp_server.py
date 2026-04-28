from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")


def test_server_can_be_created():
    from mcp.server import Server
    server = Server("memcontext-test")
    assert server is not None


def test_server_module_importable():
    from memcontext import mcp_server
    assert hasattr(mcp_server, "run_server")
