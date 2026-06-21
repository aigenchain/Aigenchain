"""
MCP runtime implementation.

Extracted from src.tool_implementations.py
"""

from typing import Optional, Dict, Any

from src.tool_implementations import (
    _parse_tool_args,
    logger,
)


async def do_manage_mcp(content: str, owner: Optional[str] = None) -> Dict:
    """Manage MCP servers: list, add, delete, enable, disable, reconnect."""
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}

    action = args.get("action", "list")

    if action == "list":
        mcp = get_mcp_manager()
        if not mcp:
            return {"response": "No MCP manager available", "servers": [], "exit_code": 0}
        from core.database import SessionLocal, McpServer
        db = SessionLocal()
        try:
            servers = db.query(McpServer).all()
            items = []
            for s in servers:
                st = mcp.get_server_status(s.id)
                status = st.get("status", "disconnected")
                tool_count = st.get("tool_count", 0)
                items.append({"id": s.id, "name": s.name, "transport": s.transport,
                              "is_enabled": s.is_enabled, "status": status,
                              "tool_count": tool_count})
            return {"response": f"{len(items)} MCP servers", "servers": items, "exit_code": 0}
        finally:
            db.close()

    elif action == "add":
        from core.database import SessionLocal, McpServer
        import uuid as _uuid
        from datetime import datetime
        name = args.get("name", "")
        command = args.get("command", "")
        cmd_args = args.get("args", [])
        env = args.get("env", {})
        if not name or not command:
            return {"error": "name and command are required", "exit_code": 1}
        sid = str(_uuid.uuid4())[:8]
        db = SessionLocal()
        try:
            srv = McpServer(id=sid, name=name, transport="stdio", command=command,
                            args=json.dumps(cmd_args) if isinstance(cmd_args, list) else cmd_args,
                            env=json.dumps(env) if isinstance(env, dict) else env,
                            is_enabled=True, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
            db.add(srv)
            db.commit()
        finally:
            db.close()
        # Try to connect
        mcp = get_mcp_manager()
        tool_count = 0
        if mcp:
            try:
                await mcp.connect_server(
                    sid, name, "stdio", command=command,
                    args=cmd_args if isinstance(cmd_args, list) else json.loads(cmd_args),
                    env=env if isinstance(env, dict) else json.loads(env),
                )
                st = mcp.get_server_status(sid)
                tool_count = st.get("tool_count", 0)
            except Exception as e:
                logger.warning(f"MCP connect failed for {name}: {e}")
        return {"response": f"Added MCP server '{name}' ({tool_count} tools)", "exit_code": 0}

    elif action == "delete":
        sid = args.get("server_id", "")
        from core.database import SessionLocal, McpServer
        db = SessionLocal()
        try:
            srv = db.query(McpServer).filter(McpServer.id == sid).first()
            if not srv:
                return {"error": f"Server {sid} not found", "exit_code": 1}
            name = srv.name
            mcp = get_mcp_manager()
            if mcp:
                try:
                    await mcp.disconnect_server(sid)
                except Exception:
                    pass
            db.delete(srv)
            db.commit()
            return {"response": f"Deleted MCP server '{name}'", "exit_code": 0}
        finally:
            db.close()

    elif action == "reconnect":
        sid = args.get("server_id", "")
        mcp = get_mcp_manager()
        if not mcp:
            return {"error": "MCP manager not available", "exit_code": 1}
        try:
            await mcp.disconnect_server(sid)
            from core.database import SessionLocal, McpServer
            db2 = SessionLocal()
            try:
                srv = db2.query(McpServer).filter(McpServer.id == sid).first()
                if srv:
                    _args = json.loads(srv.args) if srv.args else []
                    _env = json.loads(srv.env) if srv.env else {}
                    await mcp.connect_server(
                        server_id=sid,
                        name=srv.name,
                        transport=srv.transport,
                        command=srv.command,
                        args=_args,
                        env=_env,
                        url=srv.url,
                    )
                    st = mcp.get_server_status(sid)
                    return {"response": f"Reconnected '{srv.name}' ({st.get('tool_count', 0)} tools)", "exit_code": 0}
                return {"error": f"Server {sid} not found", "exit_code": 1}
            finally:
                db2.close()
        except Exception as e:
            return {"error": str(e), "exit_code": 1}

    elif action in ("enable", "disable"):
        sid = args.get("server_id", "")
        from core.database import SessionLocal, McpServer
        db = SessionLocal()
        try:
            srv = db.query(McpServer).filter(McpServer.id == sid).first()
            if not srv:
                return {"error": f"Server {sid} not found", "exit_code": 1}
            srv.is_enabled = (action == "enable")
            db.commit()
            return {"response": f"MCP server '{srv.name}' {action}d", "exit_code": 0}
        finally:
            db.close()

    elif action == "list_tools":
        mcp = get_mcp_manager()
        if not mcp:
            return {"response": "No MCP manager", "tools": [], "exit_code": 0}
        tools = mcp.get_all_tools()
        items = [{"name": t["name"], "server": t["server_name"],
                  "description": t.get("description", "")[:100]} for t in tools]
        return {"response": f"{len(items)} MCP tools available", "tools": items, "exit_code": 0}

    else:
        return {"error": f"Unknown action: {action}", "exit_code": 1}


# ---------------------------------------------------------------------------
# Webhook management tool
# ---------------------------------------------------------------------------

