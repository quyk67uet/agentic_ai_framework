"""
Agno AI Agent
=============
An intelligent agent using Agno framework with Gemini.
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

from agno.agent import Agent
from agno.models.google import Gemini

# ============================================================================
# CREATE AGENT WITH GEMINI (No external tools - uses Gemini's knowledge)
# ============================================================================

agent = Agent(
    name="Research Assistant",
    model=Gemini(
        id="gemini-2.5-flash",
        api_key=api_key
    ),
    instructions=[
        "You are a helpful research assistant.",
        "Provide detailed and well-organized responses.",
        "Use your knowledge to answer questions accurately.",
        "If you don't know something, say so honestly."
    ],
    markdown=True
)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🔬"*20)
    print("      AGNO AI RESEARCH AGENT")
    print("🔬"*20)
    print("\nCapabilities:")
    print("  • 🔍 DuckDuckGo Web Search")
    print("  • 🤖 Powered by Gemini 2.5 Flash")
    print("\nType 'quit' to exit\n")
    
    while True:
        query = input("👤 You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not query:
            continue
        
        print("\n🔄 Processing...\n")
        
        try:
            response = agent.run(query)
            print("\n" + "="*60)
            print("📋 RESPONSE:")
            print("="*60)
            print(response.content)
            print("="*60 + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

