import os
import requests
import json
from dotenv import load_dotenv
from tools import TOOLS, TOOL_MAP

load_dotenv()

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
        
        response = requests.post(URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}", history

        data = response.json()
        candidate = data['candidates'][0]
        
        # Find a function call in any of the parts
        function_call = None
        for part in candidate['content']['parts']:
            if 'functionCall' in part:
                function_call = part['functionCall']
                break
        
        if function_call:
            name = function_call['name']
            args = function_call['args']
            
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
