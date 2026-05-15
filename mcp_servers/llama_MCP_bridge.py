import asyncio
import json
import os
from pathlib import Path
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

sessions = {}
tool_ids = {}
exit_stack = None


def _expand_env_vars(value):
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda match: os.getenv(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


async def start_servers(config_path: str):
    """Initialize and connect to all MCP servers"""
    global exit_stack, sessions, tool_ids
    sessions = {}
    tool_ids = {}
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()
    
    resolved_path = Path(config_path).resolve()
    config_dir = resolved_path.parent
    with resolved_path.open('r', encoding='utf-8') as f:
        config = json.load(f)
    
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcp.json must contain an 'mcpServers' object.")

    for server_name, server_config in servers.items():
        print(f"Connecting to {server_name}...")
        merged_env = os.environ.copy()
        merged_env.update(_expand_env_vars(server_config.get("env") or {}))
        resolved_args = []
        for arg in _expand_env_vars(server_config.get("args", [])):
            if isinstance(arg, str):
                candidate = (config_dir / arg).resolve()
                if not Path(arg).is_absolute() and candidate.exists():
                    resolved_args.append(str(candidate))
                    continue
            resolved_args.append(arg)

        # Prepare the connection parameters
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=resolved_args,
            env=merged_env
        )

        try:
            # Connect to the server
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            sessions[server_name] = session
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")


async def get_all_mcp_tools():
    """Get all tools from connected MCP servers"""
    all_openai_tools = []

    for server_name, session in sessions.items():
        try:
            # Get tools from this specific server
            mcp_tools = await session.list_tools()
            
            for tool in mcp_tools.tools:
                tool_ids[tool.name] = server_name
                # Convert to OpenAI Format
                openai_tool = {
                    "type": "function",
                    "function":{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {k: v for k, v in vals.items() if k != 'title'}
                                for name, vals in tool.inputSchema.get('properties', {}).items()
                            },
                            "required": tool.inputSchema.get('required', [])
                        }
                    }
                }
                all_openai_tools.append(openai_tool)
        except Exception as e:
            print(f"Failed to fetch tools from {server_name}: {e}")

    return all_openai_tools


async def get_all_tool_names():
    tools = await get_all_mcp_tools()
    return [tool["function"]["name"] for tool in tools]

async def execute_tool(tool_name: str, tool_args: dict):
    server_name = tool_ids.get(tool_name)
    if not server_name:
        raise ValueError(f"Tool {tool_name} not found in any server.")
    session = sessions[server_name]    
    response = await session.call_tool(tool_name, tool_args)
    return response

async def cleanup():
    """Clean up all MCP connections"""
    global exit_stack
    if exit_stack:
        try:
            await exit_stack.__aexit__(None, None, None)
        except (RuntimeError, asyncio.CancelledError):
            pass
    sessions.clear()
    tool_ids.clear()
    exit_stack = None
    
async def main():
    config_file = "mcp.json" 
    try:
        await start_servers(config_file)
        final_tools = await get_all_mcp_tools()
        print(json.dumps(final_tools, indent=2))
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
    

