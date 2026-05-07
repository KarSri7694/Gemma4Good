from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import signal
import sys
import threading
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_servers.llama_MCP_bridge import cleanup, execute_tool, get_all_mcp_tools, start_servers


WINDOWS_CTRL_C_EVENT = 0
WINDOWS_CTRL_BREAK_EVENT = 1
_windows_ctrl_handler_ref = None


async def _run(config_path: str, tool_name: str | None, tool_args: str | None) -> None:
    stop_requested = threading.Event()

    def request_stop(*_args) -> None:
        if not stop_requested.is_set():
            print("\nShutting down MCP bridge...")
            stop_requested.set()

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)
    previous_sigbreak = None
    if hasattr(signal, "SIGBREAK"):
        previous_sigbreak = signal.getsignal(signal.SIGBREAK)
        signal.signal(signal.SIGBREAK, request_stop)

    windows_handler_installed = False
    if sys.platform == "win32":
        global _windows_ctrl_handler_ref

        def _windows_ctrl_handler(ctrl_type):
            if ctrl_type in (WINDOWS_CTRL_C_EVENT, WINDOWS_CTRL_BREAK_EVENT):
                request_stop()
                return 1
            return 0

        handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        _windows_ctrl_handler_ref = handler_type(_windows_ctrl_handler)
        windows_handler_installed = bool(
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_windows_ctrl_handler_ref, True)
        )

    try:
        await start_servers(config_path)
        tools = await get_all_mcp_tools()
        print(f"Connected MCP servers. Loaded {len(tools)} tools.")

        if tool_name:
            parsed_args = json.loads(tool_args) if tool_args else {}
            response = await execute_tool(tool_name, parsed_args)
            print(f"\nTool response for {tool_name}:")
            print(response)
            return

        print("\nBridge is running. Press Ctrl+C to stop.")
        while not stop_requested.is_set():
            await asyncio.sleep(0.25)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigbreak is not None and hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, previous_sigbreak)
        if sys.platform == "win32" and windows_handler_installed and _windows_ctrl_handler_ref is not None:
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_windows_ctrl_handler_ref, False)

        await cleanup()
        print("MCP bridge stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start MCP servers from mcp.json using llama_MCP_bridge.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "mcp.json"),
        help="Path to mcp.json",
    )
    parser.add_argument("--tool", help="Optional tool name to execute after connecting.")
    parser.add_argument(
        "--tool-args",
        help="Optional JSON object string to pass as tool arguments.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.config, args.tool, args.tool_args))


if __name__ == "__main__":
    main()
