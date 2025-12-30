"""
Agentic AI with Decision Making
===============================
An intelligent agent that analyzes user input and makes decisions
by selecting and executing appropriate tools/actions.
"""

import google.generativeai as genai
import json
import re
from typing import Callable, Any
from datetime import datetime

# ============================================================================
# CONFIGURATION - SECURE API KEY HANDLING
# ============================================================================
import os
from dotenv import load_dotenv

# Load API key from .env file (SECURE METHOD)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("="*60)
    print("⚠️  API KEY NOT FOUND!")
    print("="*60)
    print("Please create a .env file with your API key:")
    print("   1. Create file: .env")
    print("   2. Add line: GOOGLE_API_KEY=your_actual_key_here")
    print("="*60)
    exit(1)
    
genai.configure(api_key=GOOGLE_API_KEY)
print("✅ API Key loaded securely from .env file")


# ============================================================================
# TOOL DEFINITIONS - Functions the agent can use
# ============================================================================

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Allow only safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the given text."""
    positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'amazing', 'wonderful', 'fantastic', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'sad', 'angry', 'disappointed']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return f"Sentiment: POSITIVE (confidence: {positive_count}/{positive_count + negative_count + 1})"
    elif negative_count > positive_count:
        return f"Sentiment: NEGATIVE (confidence: {negative_count}/{positive_count + negative_count + 1})"
    else:
        return "Sentiment: NEUTRAL"


def word_count(text: str) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


def summarize_data(data: str) -> str:
    """Provide a summary analysis of numerical data."""
    try:
        # Extract numbers from text
        numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', data)]
        if not numbers:
            return "No numerical data found to analyze"
        
        avg = sum(numbers) / len(numbers)
        minimum = min(numbers)
        maximum = max(numbers)
        total = sum(numbers)
        
        return f"""Data Analysis:
- Count: {len(numbers)} numbers
- Sum: {total:.2f}
- Average: {avg:.2f}
- Minimum: {minimum:.2f}
- Maximum: {maximum:.2f}
- Range: {maximum - minimum:.2f}"""
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


def create_action_plan(goal: str) -> str:
    """Create a step-by-step action plan for achieving a goal."""
    return f"""Action Plan for: {goal}
    
Step 1: Define clear objectives and success criteria
Step 2: Break down the goal into smaller, manageable tasks
Step 3: Identify required resources and dependencies
Step 4: Set realistic timelines for each task
Step 5: Execute tasks in priority order
Step 6: Monitor progress and adjust as needed
Step 7: Review outcomes and document learnings

Note: This is a general template. Specific steps depend on the goal's nature."""


def search_knowledge(query: str) -> str:
    """Search internal knowledge base (simulated)."""
    knowledge_base = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn from data.",
        "agent": "An AI agent is an autonomous system that perceives its environment and takes actions to achieve goals.",
        "gemini": "Gemini is Google's multimodal AI model family.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Found: {value}"
    
    return f"No specific information found for '{query}'. Please try a different search term."


# ============================================================================
# TOOL REGISTRY
# ============================================================================

AVAILABLE_TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Perform mathematical calculations. Use for any math operations.",
        "parameters": "expression (string): The mathematical expression to evaluate"
    },
    "get_datetime": {
        "function": get_current_datetime,
        "description": "Get the current date and time.",
        "parameters": "None"
    },
    "sentiment_analysis": {
        "function": analyze_sentiment,
        "description": "Analyze the sentiment (positive/negative/neutral) of text.",
        "parameters": "text (string): The text to analyze"
    },
    "word_count": {
        "function": word_count,
        "description": "Count words, characters, and sentences in text.",
        "parameters": "text (string): The text to analyze"
    },
    "data_analysis": {
        "function": summarize_data,
        "description": "Analyze numerical data and provide statistics.",
        "parameters": "data (string): Text containing numbers to analyze"
    },
    "action_planner": {
        "function": create_action_plan,
        "description": "Create an action plan for achieving a goal.",
        "parameters": "goal (string): The goal to plan for"
    },
    "knowledge_search": {
        "function": search_knowledge,
        "description": "Search the knowledge base for information.",
        "parameters": "query (string): The search query"
    }
}


# ============================================================================
# AGENTIC AI CLASS
# ============================================================================

class AgenticAI:
    """
    An AI Agent that makes decisions based on user input
    and executes appropriate tools to accomplish tasks.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.max_iterations = 5  # Prevent infinite loops
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt with available tools."""
        tools_description = "\n".join([
            f"- {name}: {info['description']} | Parameters: {info['parameters']}"
            for name, info in AVAILABLE_TOOLS.items()
        ])
        
        return f"""You are an intelligent AI agent with decision-making capabilities.
Your task is to analyze user input and decide the best course of action.

AVAILABLE TOOLS:
{tools_description}

DECISION-MAKING PROCESS:
1. Analyze the user's input carefully
2. Determine what the user wants to achieve
3. Decide which tool(s) to use (if any)
4. Provide your reasoning and execute the appropriate action

RESPONSE FORMAT:
You MUST respond in this exact JSON format:
{{
    "thought": "Your reasoning about what the user wants and what action to take",
    "decision": "TOOL" or "DIRECT_RESPONSE",
    "tool_name": "name of the tool to use (only if decision is TOOL)",
    "tool_input": "input for the tool (only if decision is TOOL)",
    "response": "Your direct response to the user (only if decision is DIRECT_RESPONSE)"
}}

IMPORTANT:
- Use tools when the task requires computation, data analysis, or specific actions
- Use DIRECT_RESPONSE for greetings, general questions, or when no tool is needed
- Always explain your reasoning in the "thought" field
- Be helpful and provide clear, actionable responses
"""

    def _parse_response(self, response_text: str) -> dict:
        """Parse the AI's JSON response."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "Could not parse response", "raw": response_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": response_text}

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result."""
        if tool_name in AVAILABLE_TOOLS:
            try:
                tool_func = AVAILABLE_TOOLS[tool_name]["function"]
                if tool_name == "get_datetime":
                    return tool_func()
                else:
                    return tool_func(tool_input)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    def process_input(self, user_input: str) -> dict:
        """
        Process user input through the agentic decision-making loop.
        Returns the agent's decision, actions taken, and final response.
        """
        result = {
            "user_input": user_input,
            "thought_process": [],
            "tools_used": [],
            "final_response": ""
        }
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser Input: {user_input}"
        
        # Get AI's decision
        try:
            response = self.model.generate_content(full_prompt)
            parsed = self._parse_response(response.text)
            
            if "error" in parsed:
                result["final_response"] = f"I encountered an issue processing your request. Let me try to help directly.\n\nYour input was: {user_input}"
                return result
            
            # Record the thought process
            result["thought_process"].append(parsed.get("thought", "No thought recorded"))
            
            # Make decision
            decision = parsed.get("decision", "DIRECT_RESPONSE")
            
            if decision == "TOOL":
                tool_name = parsed.get("tool_name", "")
                tool_input = parsed.get("tool_input", "")
                
                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_input)
                result["tools_used"].append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": tool_result
                })
                
                # Generate final response based on tool output
                final_prompt = f"""Based on the tool execution, provide a helpful response to the user.

User's original input: {user_input}
Tool used: {tool_name}
Tool result: {tool_result}

Provide a clear, helpful response that explains the result to the user."""
                
                final_response = self.model.generate_content(final_prompt)
                result["final_response"] = final_response.text
                
            else:
                # Direct response
                result["final_response"] = parsed.get("response", "I'm here to help! Please tell me more about what you need.")
                
        except Exception as e:
            result["final_response"] = f"An error occurred: {str(e)}"
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": result["final_response"]
        })
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def display_result(result: dict):
    """Display the agent's result in a formatted way."""
    print("\n" + "="*60)
    print("🤖 AGENTIC AI DECISION MAKING")
    print("="*60)
    
    print(f"\n📝 USER INPUT: {result['user_input']}")
    
    if result['thought_process']:
        print(f"\n💭 THOUGHT PROCESS:")
        for i, thought in enumerate(result['thought_process'], 1):
            print(f"   {i}. {thought}")
    
    if result['tools_used']:
        print(f"\n🔧 TOOLS USED:")
        for tool in result['tools_used']:
            print(f"   • Tool: {tool['tool']}")
            print(f"     Input: {tool['input']}")
            print(f"     Output: {tool['output']}")
    
    print(f"\n💬 RESPONSE:")
    print(f"   {result['final_response']}")
    print("\n" + "="*60)


def main():
    """Main function to run the Agentic AI."""
    print("\n" + "🚀"*20)
    print("       AGENTIC AI - DECISION MAKING SYSTEM")
    print("🚀"*20)
    print("\nAvailable capabilities:")
    for name, info in AVAILABLE_TOOLS.items():
        print(f"  • {name}: {info['description']}")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    agent = AgenticAI()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("Please enter some input.")
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Process the input through the agent
            result = agent.process_input(user_input)
            display_result(result)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
