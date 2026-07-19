import os
import requests
import json
from dotenv import load_dotenv
from tools import TOOLS, TOOL_MAP, read_file, write_file, run_command, apply_patch
from mcp_client import MCPClient

load_dotenv()

# Initialize MCP Client
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
mcp = MCPClient("npx", ["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"])

# Fetch tools from MCP and add to our tool list
mcp_tools = mcp.list_tools()

def clean_schema(schema):
    """Recursively remove fields not supported by Gemini's function calling."""
    if not isinstance(schema, dict):
        return schema
    
    new_schema = {}
    for k, v in schema.items():
        if k in ["additionalProperties", "const"]:
            continue
        if isinstance(v, dict):
            new_schema[k] = clean_schema(v)
        elif isinstance(v, list):
            new_schema[k] = [clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            new_schema[k] = v
    return new_schema

user_input = input("\nAdd Tavily MCP Tools: ")

# Convert MCP tool format to Gemini tool format
if user_input.lower() in ["y", "yes"]:
    # Access the list of function_declarations inside the first element of TOOLS
    target_list = TOOLS[0]["function_declarations"]

    for tool in mcp_tools:
        target_list.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": clean_schema(tool["inputSchema"])
        })
        # Create a wrapper to call the MCP tool
        def create_wrapper(name):
            return lambda **args: mcp.call_tool(name, args)
        TOOL_MAP[tool["name"]] = create_wrapper(tool["name"])

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def call_gemini(prompt, history=None):
    if history is None:
        history = []
    
    # System instruction
    system_instruction = "You are an expert coding agent. You have access to tools to read files, write files, and execute shell commands. Always be concise, explain your reasoning, and prioritize safety when executing commands."
    
    history.append({"role": "user", "parts": [{"text": prompt}]})
    
    while True:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": history,
            "tools": TOOLS,
            "system_instruction": {"parts": [{"text": system_instruction}]}
        }
        print("\n--- PAYLOAD ---")
        print(json.dumps(payload, indent=2))
        print("---------------\n")
        
        response = requests.post(URL, headers=headers, json=payload)
        
        print("\n--- RESPONSE ---")
        print(json.dumps(response.json(), indent=2))
        print("----------------\n")
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}", history

        data = response.json()
        candidate = data['candidates'][0]
        metadata = data['usageMetadata']
        print(f"Prompt_Tokens: {metadata["promptTokenCount"]}")
        print(f"Output_Tokens: {metadata["candidatesTokenCount"]}")
        print(f"Total_Tokens:  {metadata["totalTokenCount"]}")
        
        # Find a function call in any of the parts
        function_call = None
        for part in candidate['content']['parts']:
            if 'functionCall' in part:
                function_call = part['functionCall']
                break
        
        if function_call:
            name = function_call['name']
            args = function_call['args']
            
            print(f"Agent calling: {name} with {args}")
            
            # Execute the tool
            result = TOOL_MAP[name](**args)
            
            # Add the tool call and result to history
            history.append(candidate['content'])
            history.append({
                "role": "function",
                "parts": [{"functionResponse": {"name": name, "response": {"result": result}}}]
            })
            # Continue the loop to let the model process the tool result
        else:
            # Handle text response
            text_response = ""
            for part in candidate['content']['parts']:
                if 'text' in part:
                    text_response += " " + part['text']
            
            history.append({"role": "model", "parts": [{"text": text_response}]})
            return text_response, history

if __name__ == "__main__":
    print("Coding Agent initialized. Type 'exit' to quit.")
    history = []
    while True:
        human_input = input("\nYou: ")
        if human_input.lower() == 'exit':
            break
        
        response, history = call_gemini(human_input, history)
        print(f"\nAgent: {response}")
