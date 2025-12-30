import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------
# STEP 1: SETUP THE "BRAIN" (The LLM)
# ---------------------------------------------------------
# Load API key from .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Please add GOOGLE_API_KEY to your .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=api_key
)


# ---------------------------------------------------------
# STEP 2: DEFINE THE "HANDS" (The Tools)
# ---------------------------------------------------------
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """Use this to search the web for current/live information. 
    Good for news, current events, facts, and any real-time data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            
            output = "Search Results:\n\n"
            for i, result in enumerate(results, 1):
                output += f"{i}. **{result['title']}**\n"
                output += f"   {result['body']}\n"
                output += f"   Source: {result['href']}\n\n"
            return output
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Use this to get the weather for a specific city."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"current weather in {city}", max_results=1))
            if results:
                return f"Weather info for {city}: {results[0]['body']}"
            return f"Weather data not available for {city}."
    except Exception as e:
        return f"Weather error: {str(e)}"

@tool
def multiply(a: int, b: int) -> int:
    """Use this to multiply two numbers."""
    return a * b

@tool
def calculate(expression: str) -> str:
    """Use this to evaluate mathematical expressions. Example: '25 * 4 + 100'"""
    try:
        allowed = set('0123456789+-*/.() ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Calculation error: {str(e)}"

tools = [web_search, get_weather, multiply, calculate]

# ---------------------------------------------------------
# STEP 3: CREATE THE AGENT (The Body)
# ---------------------------------------------------------
agent_executor = create_react_agent(llm, tools)

# ---------------------------------------------------------
# STEP 4: STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Agent Chat", page_icon="🤖", layout="centered")

st.title("🤖 AI Agent Chat")
st.caption("🔍 Web Search | 🌤️ Weather | 🧮 Calculator")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"messages": [("user", prompt)]})
            final_answer = response["messages"][-1].content
        st.markdown(final_answer)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    