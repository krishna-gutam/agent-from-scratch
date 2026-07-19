import subprocess
import json
import sys

class MCPClient:
    def __init__(self, command, args):
        # On Windows, npx is a batch file, so we need shell=True
        is_windows = sys.platform == "win32"
        self.process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8',
            bufsize=0,
            shell=is_windows
        )
        self.request_id = 1
        self._initialize()

    def _send(self, method, params=None):
        message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            message["params"] = params
        
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        self.request_id += 1
        
        # Read response
        line = self.process.stdout.readline()
        return json.loads(line)

    def _initialize(self):
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "my-agent", "version": "1.0"}
        })
        self._send("initialized")

    def list_tools(self):
        response = self._send("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name, arguments):
        response = self._send("tools/call", {"name": name, "arguments": arguments})
        return response.get("result", {}).get("content", [])

    def close(self):
        self.process.terminate()
