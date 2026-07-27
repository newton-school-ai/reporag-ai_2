"""Unit tests for GraphRetriever (Issue 18)."""

from __future__ import annotations

import pytest

from reporag.graph.call_graph import CallEdge
from reporag.graph.dependency_graph import DependencyEdge
from reporag.graph.neo4j_store import NetworkXGraphStore
from reporag.graph.symbol_table import SymbolRecord, SymbolTable
from reporag.retrieval.graph_traversal import GraphRetriever
from reporag.retrieval.vector_search import RetrievalResult


@pytest.fixture
def test_graph_store() -> NetworkXGraphStore:
    """Fixture that builds a realistic small graph store with NetworkX fallback."""
    # 1. Create SymbolTable
    table = SymbolTable()
    records = [
        SymbolRecord(
            symbol_id="func_auth",
            name="authenticate",
            qualified_name="auth.authenticate",
            type="function",
            file_path="src/auth.py",
            module="auth",
            start_line=10,
            end_line=20,
            signature="def authenticate(token: str):",
            docstring="Authenticates user.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="func_verify",
            name="verify_token",
            qualified_name="auth.verify_token",
            type="function",
            file_path="src/auth.py",
            module="auth",
            start_line=22,
            end_line=30,
            signature="def verify_token(token: str):",
            docstring="Verifies the JWT token.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="func_login",
            name="login",
            qualified_name="api.login",
            type="function",
            file_path="src/api.py",
            module="api",
            start_line=5,
            end_line=15,
            signature="def login(req):",
            docstring="Login endpoint.",
            parent="",
            is_async=False,
            language="python",
        ),
        SymbolRecord(
            symbol_id="class_user",
            name="User",
            qualified_name="models.User",
            type="class",
            file_path="src/models.py",
            module="models",
            start_line=1,
            end_line=5,
            signature="class User:",
            docstring="User model.",
            parent="",
            is_async=False,
            language="python",
        ),
    ]
    for r in records:
        table.add(r)

    # 2. Call Edges (login -> authenticate -> verify_token)
    call_edges = [
        CallEdge(
            caller="api.login",
            callee="auth.authenticate",
            caller_file="src/api.py",
            callee_file="src/auth.py",
            call_type="direct",
            resolution="exact",
            call_site_line=10,
        ),
        CallEdge(
            caller="auth.authenticate",
            callee="auth.verify_token",
            caller_file="src/auth.py",
            callee_file="src/auth.py",
            call_type="direct",
            resolution="exact",
            call_site_line=15,
        ),
    ]

    # 3. Dependency Edges (api -> auth)
    dep_edges = [
        DependencyEdge(
            source_module="api",
            target_module="auth",
            import_type="direct",
            line=1,
            resolved=True,
            source="src/api.py",
            target="src/auth.py",
        )
    ]

    # 4. Initialize store and persist
    store = NetworkXGraphStore()
    store.persist_graph(call_edges, dep_edges, table)
    return store


@pytest.fixture
def graph_retriever(test_graph_store: NetworkXGraphStore) -> GraphRetriever:
    return GraphRetriever(store=test_graph_store)


def test_schema_parity(graph_retriever: GraphRetriever) -> None:
    """Ensure graph retrieval returns standard RetrievalResult objects."""
    results = graph_retriever.get_neighbors("func_auth")
    assert results
    assert isinstance(results[0], RetrievalResult)
    r = results[0]
    assert hasattr(r, "score")
    assert hasattr(r, "file_path")
    assert hasattr(r, "start_line")
    assert hasattr(r, "end_line")
    assert hasattr(r, "symbol_name")
    assert hasattr(r, "chunk_text")
    assert hasattr(r, "metadata")
    assert "type" in r.metadata


def test_get_neighbors(graph_retriever: GraphRetriever) -> None:
    """Test retrieving generic neighbors."""
    # Neighbors of func_auth should include func_verify (callee) and func_login (caller)
    results = graph_retriever.get_neighbors("func_auth", depth=1, direction="both")
    symbol_names = {r.symbol_name for r in results}
    assert "verify_token" in symbol_names
    assert "login" in symbol_names


def test_get_callers(graph_retriever: GraphRetriever) -> None:
    """Test retrieving only callers via directed 'in' edge traversal."""
    results = graph_retriever.get_callers("func_auth", depth=1)
    symbol_names = {r.symbol_name for r in results}
    assert "login" in symbol_names
    assert "verify_token" not in symbol_names  # Callee, not a caller

    results_deep = graph_retriever.get_callers("func_verify", depth=2)
    symbol_names_deep = {r.symbol_name for r in results_deep}
    assert "authenticate" in symbol_names_deep
    assert "login" in symbol_names_deep


def test_find_paths(graph_retriever: GraphRetriever) -> None:
    """Test shortest path finding."""
    # Path from login -> verify_token
    results = graph_retriever.find_paths("func_login", "func_verify")
    # Path: func_login -> func_auth -> func_verify
    assert len(results) == 3
    names = [r.symbol_name for r in results]
    assert names == ["login", "authenticate", "verify_token"]

    # Check distances/scores
    assert results[0].score == 1.0  # distance 0
    assert results[1].score == 0.5  # distance 1
    assert results[2].score == pytest.approx(0.333, abs=0.01)  # distance 2


def test_find_paths_no_path(graph_retriever: GraphRetriever) -> None:
    """Test when no path exists."""
    results = graph_retriever.find_paths("func_login", "class_user")
    assert results == []


def test_extract_subgraph(graph_retriever: GraphRetriever) -> None:
    """Test subgraph extraction."""
    results = graph_retriever.extract_subgraph(["func_login", "func_auth"])
    assert len(results) == 2
    names = {r.symbol_name for r in results}
    assert "login" in names
    assert "authenticate" in names


def test_node_to_result_chunk_text(graph_retriever: GraphRetriever) -> None:
    """Test correct synthesis of chunk_text from signature and docstring."""
    results = graph_retriever.get_neighbors("func_login", depth=1, direction="out")
    # One of them is authenticate
    auth_result = next(r for r in results if r.symbol_name == "authenticate")
    expected_chunk = 'def authenticate(token: str):\n"""Authenticates user."""'
    assert auth_result.chunk_text == expected_chunk
