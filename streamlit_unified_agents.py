"""
Unified Multi-Agent Framework UI
================================
A Streamlit web interface to interact with multiple AI agent frameworks:
- Agno
- SmolAgents
- CrewAI
- LangChain
- Agentic AI
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Multi-Agent Framework Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "selected_framework" not in st.session_state:
    st.session_state.selected_framework = "Agno"
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "agents" not in st.session_state:
    st.session_state.agents = {}

# ============================================================================
# CHECK API KEY
# ============================================================================
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Please add GOOGLE_API_KEY to your .env file")
    st.stop()

# ============================================================================
# AGENT INITIALIZATION FUNCTIONS
# ============================================================================

def init_agno_agent():
    """Initialize Agno agent."""
    if "agno" not in st.session_state.agents:
        try:
            from agno.agent import Agent
            from agno.models.google import Gemini
            
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
            st.session_state.agents["agno"] = agent
        except Exception as e:
            st.error(f"Error initializing Agno agent: {e}")
            return None
    return st.session_state.agents.get("agno")

def init_smol_agent():
    """Initialize SmolAgents agent."""
    if "smol" not in st.session_state.agents:
        try:
            from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, VisitWebpageTool
            
            model = LiteLLMModel(
                model_id="gemini/gemini-2.5-flash",
                api_key=api_key
            )
            
            tools = [
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
            ]
            
            agent = CodeAgent(
                tools=tools,
                model=model,
                max_steps=10,
                verbosity_level=2
            )
            st.session_state.agents["smol"] = agent
        except Exception as e:
            st.error(f"Error initializing SmolAgents: {e}")
            return None
    return st.session_state.agents.get("smol")

def init_crewai_agent():
    """Initialize CrewAI components."""
    if "crewai" not in st.session_state.agents:
        try:
            from crewai import Agent, Task, Crew, Process, LLM
            
            gemini_llm = LLM(
                model="gemini/gemini-2.5-flash",
                api_key=api_key
            )
            
            # Store LLM and agent definitions for later use
            st.session_state.agents["crewai"] = {
                "llm": gemini_llm,
                "initialized": True
            }
        except Exception as e:
            st.error(f"Error initializing CrewAI: {e}")
            return None
    return st.session_state.agents.get("crewai")

def init_langchain_agent():
    """Initialize LangChain agent."""
    if "langchain" not in st.session_state.agents:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
            from duckduckgo_search import DDGS
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=api_key
            )
            
            @tool
            def web_search(query: str) -> str:
                """Search the web for current information."""
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
            def calculate(expression: str) -> str:
                """Evaluate mathematical expressions."""
                try:
                    allowed = set('0123456789+-*/.() ')
                    if all(c in allowed for c in expression):
                        result = eval(expression)
                        return f"Result: {result}"
                    return "Invalid expression"
                except Exception as e:
                    return f"Calculation error: {str(e)}"
            
            tools = [web_search, calculate]
            agent = create_react_agent(llm, tools)
            st.session_state.agents["langchain"] = agent
        except Exception as e:
            st.error(f"Error initializing LangChain: {e}")
            return None
    return st.session_state.agents.get("langchain")

def init_agentic_ai():
    """Initialize Agentic AI."""
    if "agentic" not in st.session_state.agents:
        try:
            import google.generativeai as genai
            import json
            import re
            from typing import Callable, Any
            from datetime import datetime
            
            genai.configure(api_key=api_key)
            
            # Define tools inline to avoid importing the full module
            def calculator(expression: str) -> str:
                try:
                    allowed_chars = set('0123456789+-*/.() ')
                    if not all(c in allowed_chars for c in expression):
                        return "Error: Invalid characters in expression"
                    result = eval(expression)
                    return f"Result: {result}"
                except Exception as e:
                    return f"Error: {str(e)}"
            
            def get_current_datetime() -> str:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            AVAILABLE_TOOLS = {
                "calculator": {
                    "function": calculator,
                    "description": "Perform mathematical calculations",
                    "parameters": "expression (string)"
                },
                "get_datetime": {
                    "function": get_current_datetime,
                    "description": "Get current date and time",
                    "parameters": "None"
                }
            }
            
            # Simplified AgenticAI class
            class AgenticAI:
                def __init__(self, model_name: str = "gemini-2.5-flash"):
                    self.model = genai.GenerativeModel(model_name)
                    self.max_iterations = 5
                
                def _parse_response(self, response_text: str) -> dict:
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            return json.loads(json_match.group())
                        return {"error": "Could not parse response", "raw": response_text}
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON", "raw": response_text}
                
                def _execute_tool(self, tool_name: str, tool_input: str) -> str:
                    if tool_name in AVAILABLE_TOOLS:
                        try:
                            tool_func = AVAILABLE_TOOLS[tool_name]["function"]
                            if tool_name == "get_datetime":
                                return tool_func()
                            else:
                                return tool_func(tool_input)
                        except Exception as e:
                            return f"Error executing {tool_name}: {str(e)}"
                    return f"Unknown tool: {tool_name}"
                
                def process_input(self, user_input: str) -> dict:
                    system_prompt = f"""You are an intelligent AI agent. Analyze user input and respond in JSON:
{{
    "thought": "Your reasoning",
    "decision": "TOOL" or "DIRECT_RESPONSE",
    "tool_name": "calculator or get_datetime (if TOOL)",
    "tool_input": "input for tool (if TOOL)",
    "response": "Your response (if DIRECT_RESPONSE)"
}}

Available tools: calculator, get_datetime"""
                    
                    try:
                        response = self.model.generate_content(f"{system_prompt}\n\nUser: {user_input}")
                        parsed = self._parse_response(response.text)
                        
                        result = {
                            "user_input": user_input,
                            "thought_process": [parsed.get("thought", "No thought")],
                            "tools_used": [],
                            "final_response": ""
                        }
                        
                        if parsed.get("decision") == "TOOL":
                            tool_name = parsed.get("tool_name", "")
                            tool_input = parsed.get("tool_input", "")
                            tool_result = self._execute_tool(tool_name, tool_input)
                            result["tools_used"].append(tool_name)
                            result["final_response"] = f"Tool Result: {tool_result}"
                        else:
                            result["final_response"] = parsed.get("response", "No response")
                        
                        return result
                    except Exception as e:
                        return {
                            "user_input": user_input,
                            "thought_process": [],
                            "tools_used": [],
                            "final_response": f"Error: {str(e)}"
                        }
            
            agent = AgenticAI(model_name="gemini-2.5-flash")
            st.session_state.agents["agentic"] = agent
        except Exception as e:
            st.error(f"Error initializing Agentic AI: {e}")
            return None
    return st.session_state.agents.get("agentic")

# ============================================================================
# AGENT EXECUTION FUNCTIONS
# ============================================================================

def run_agno(query: str):
    """Run Agno agent."""
    agent = init_agno_agent()
    if agent:
        response = agent.run(query)
        return response.content
    return "Error: Agent not initialized"

def run_smol(query: str):
    """Run SmolAgents agent."""
    agent = init_smol_agent()
    if agent:
        result = agent.run(query)
        return str(result)
    return "Error: Agent not initialized"

def run_crewai(topic: str):
    """Run CrewAI research crew."""
    crewai_data = init_crewai_agent()
    if not crewai_data:
        return "Error: CrewAI not initialized"
    
    try:
        from crewai import Agent, Task, Crew, Process
        
        gemini_llm = crewai_data["llm"]
        
        # Create agents
        research_analyst = Agent(
            role="Senior Research Analyst",
            goal="Conduct comprehensive research on the given topic",
            backstory="You are an expert research analyst with years of experience.",
            verbose=False,
            allow_delegation=False,
            llm=gemini_llm
        )
        
        content_synthesizer = Agent(
            role="Content Synthesizer",
            goal="Synthesize research findings into a comprehensive report",
            backstory="You excel at organizing information into clear, actionable reports.",
            verbose=False,
            allow_delegation=False,
            llm=gemini_llm
        )
        
        # Create tasks
        research_task = Task(
            description=f"Research and gather detailed information about: {topic}",
            expected_output="A comprehensive research report with key findings",
            agent=research_analyst
        )
        
        synthesis_task = Task(
            description=f"Synthesize the research findings on: {topic} into a clear report",
            expected_output="A well-structured synthesis report",
            agent=content_synthesizer
        )
        
        # Create crew
        crew = Crew(
            agents=[research_analyst, content_synthesizer],
            tasks=[research_task, synthesis_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Error running CrewAI: {e}"

def run_langchain(query: str):
    """Run LangChain agent."""
    agent = init_langchain_agent()
    if agent:
        response = agent.invoke({"messages": [("user", query)]})
        return response["messages"][-1].content
    return "Error: Agent not initialized"

def run_agentic_ai(query: str):
    """Run Agentic AI."""
    agent = init_agentic_ai()
    if agent:
        result = agent.process_input(query)
        # Format the result nicely
        output = f"**Thought Process:**\n{result.get('thought_process', ['N/A'])[-1]}\n\n"
        if result.get('tools_used'):
            output += f"**Tools Used:** {', '.join(result['tools_used'])}\n\n"
        output += f"**Response:**\n{result.get('final_response', 'No response')}"
        return output
    return "Error: Agent not initialized"

# ============================================================================
# SIDEBAR - FRAMEWORK SELECTION
# ============================================================================
with st.sidebar:
    st.title("🤖 Agent Framework Hub")
    st.markdown("---")
    
    # Framework selection
    framework_options = {
        "Agno": {
            "icon": "🔬",
            "description": "Agno v2.0 - Research Assistant",
            "capabilities": "Knowledge-based responses"
        },
        "SmolAgents": {
            "icon": "🤖",
            "description": "SmolAgents - Lightweight Agent",
            "capabilities": "Web Search, Visit Webpages"
        },
        "CrewAI": {
            "icon": "🚀",
            "description": "CrewAI - Multi-Agent Research",
            "capabilities": "Deep Research, Synthesis"
        },
        "LangChain": {
            "icon": "🔗",
            "description": "LangChain - Tool-Enabled Agent",
            "capabilities": "Web Search, Calculator"
        },
        "Agentic AI": {
            "icon": "🧠",
            "description": "Agentic AI - Decision Making",
            "capabilities": "Smart Tool Selection"
        }
    }
    
    selected = st.selectbox(
        "Select Framework",
        options=list(framework_options.keys()),
        index=list(framework_options.keys()).index(st.session_state.selected_framework)
    )
    
    st.session_state.selected_framework = selected
    
    # Display framework info
    st.markdown("---")
    st.markdown(f"### {framework_options[selected]['icon']} {selected}")
    st.caption(framework_options[selected]['description'])
    st.markdown(f"**Capabilities:** {framework_options[selected]['capabilities']}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        if selected in st.session_state.messages:
            st.session_state.messages[selected] = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This unified interface allows you to interact with multiple AI agent frameworks.
    Each framework has unique capabilities and strengths.
    """)

# ============================================================================
# MAIN INTERFACE
# ============================================================================
st.title(f"{framework_options[selected]['icon']} {selected} Agent")
st.caption(framework_options[selected]['description'])

# Initialize messages for selected framework
if selected not in st.session_state.messages:
    st.session_state.messages[selected] = []

# Display chat history
for message in st.session_state.messages[selected]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(f"Ask {selected} agent..."):
    # Add user message
    st.session_state.messages[selected].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 {selected} is thinking..."):
            try:
                if selected == "Agno":
                    response = run_agno(prompt)
                elif selected == "SmolAgents":
                    response = run_smol(prompt)
                elif selected == "CrewAI":
                    response = run_crewai(prompt)
                elif selected == "LangChain":
                    response = run_langchain(prompt)
                elif selected == "Agentic AI":
                    response = run_agentic_ai(prompt)
                else:
                    response = "Unknown framework"
            except Exception as e:
                response = f"❌ Error: {str(e)}"
        
        st.markdown(response)
    
    # Add assistant response
    st.session_state.messages[selected].append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Tip: Switch between frameworks in the sidebar to compare their capabilities!")

