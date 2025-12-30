"""
SmolAgents Research Agent
=========================
A lightweight agent using Hugging Face's smolagents framework.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ Please add GOOGLE_API_KEY to your .env file")
    exit(1)

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, VisitWebpageTool

# ============================================================================
# CONFIGURE LLM (Gemini via LiteLLM)
# ============================================================================

model = LiteLLMModel(
    model_id="gemini/gemini-2.5-flash",
    api_key=api_key
)

# ============================================================================
# DEFINE TOOLS
# ============================================================================

# Built-in tools from smolagents
tools = [
    DuckDuckGoSearchTool(),  # Web search
    VisitWebpageTool(),      # Visit and read webpages
]

# ============================================================================
# CREATE AGENT
# ============================================================================

agent = CodeAgent(
    tools=tools,
    model=model,
    max_steps=10,
    verbosity_level=2
)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🤖"*20)
    print("       SMOLAGENTS - RESEARCH ASSISTANT")
    print("🤖"*20)
    print("\nCapabilities:")
    print("  • 🔍 DuckDuckGo Web Search")
    print("  • 🌐 Visit and Read Webpages")
    print("\nType 'quit' to exit\n")
    
    while True:
        query = input("👤 You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not query:
            continue
        
        print("\n🔄 Researching...\n")
        
        try:
            result = agent.run(query)
            print("\n" + "="*60)
            print("📋 RESULT:")
            print("="*60)
            print(result)
            print("="*60 + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
