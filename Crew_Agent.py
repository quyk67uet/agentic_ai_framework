"""
CrewAI Research Agent
=====================
A multi-agent system for deep research using CrewAI framework.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ Please add GOOGLE_API_KEY to your .env file")
    exit(1)

# Set the API key for CrewAI to use with Gemini
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from crewai import Agent, Task, Crew, Process, LLM

# ============================================================================
# CONFIGURE LLM (Gemini)
# ============================================================================

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================================
# DEFINE AGENTS
# ============================================================================

# Agent 1: Research Analyst
research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Conduct comprehensive research on the given topic and gather detailed information",
    backstory="""You are an expert research analyst with years of experience 
    in conducting deep research. You excel at finding relevant information, 
    analyzing data, and identifying key insights. You are thorough and 
    leave no stone unturned in your research.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# Agent 2: Content Synthesizer
content_synthesizer = Agent(
    role="Content Synthesizer",
    goal="Synthesize research findings into a clear, comprehensive report",
    backstory="""You are a skilled content synthesizer who excels at taking 
    complex research findings and transforming them into clear, organized, 
    and actionable reports. You identify patterns, connections, and key 
    takeaways from raw research data.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# Agent 3: Critical Reviewer
critical_reviewer = Agent(
    role="Critical Reviewer",
    goal="Review and validate research findings for accuracy and completeness",
    backstory="""You are a meticulous critical reviewer with expertise in 
    fact-checking and quality assurance. You identify gaps, inconsistencies, 
    and areas that need more exploration. Your feedback ensures the highest 
    quality research output.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)


# ============================================================================
# DEFINE TASKS
# ============================================================================

def create_research_tasks(topic: str):
    """Create tasks for the research crew based on the topic."""
    
    # Task 1: Deep Research
    research_task = Task(
        description=f"""Conduct comprehensive research on: {topic}
        
        Your research should cover:
        1. Key concepts and definitions
        2. Current state and trends
        3. Major challenges and problems
        4. Notable examples and case studies
        5. Expert opinions and perspectives
        6. Future predictions and implications
        
        Be thorough and provide detailed findings.""",
        expected_output="A detailed research document with all findings organized by category",
        agent=research_analyst
    )
    
    # Task 2: Synthesize Findings
    synthesis_task = Task(
        description=f"""Synthesize the research findings on: {topic}
        
        Create a comprehensive report that includes:
        1. Executive Summary (key takeaways)
        2. Main Findings (organized by theme)
        3. Analysis (patterns, connections, insights)
        4. Recommendations (actionable next steps)
        5. Conclusion
        
        Make it clear, concise, and actionable.""",
        expected_output="A well-structured synthesis report with clear sections and actionable insights",
        agent=content_synthesizer
    )
    
    # Task 3: Critical Review
    review_task = Task(
        description=f"""Review the synthesized research report on: {topic}
        
        Your review should:
        1. Verify accuracy of key claims
        2. Identify any gaps or missing information
        3. Check for logical consistency
        4. Suggest improvements
        5. Rate the overall quality (1-10)
        
        Provide constructive feedback and a final quality assessment.""",
        expected_output="A critical review with quality rating and suggestions for improvement",
        agent=critical_reviewer
    )
    
    return [research_task, synthesis_task, review_task]


# ============================================================================
# CREATE AND RUN CREW
# ============================================================================

def run_research_crew(topic: str):
    """Run the research crew on a given topic."""
    
    print("\n" + "="*60)
    print("🔬 CREWAI DEEP RESEARCH SYSTEM")
    print("="*60)
    print(f"\n📋 Research Topic: {topic}")
    print("\n👥 Agents:")
    print("   1. Senior Research Analyst - Gathering information")
    print("   2. Content Synthesizer - Creating report")
    print("   3. Critical Reviewer - Quality assurance")
    print("\n" + "="*60)
    print("Starting research... This may take a few minutes.")
    print("="*60 + "\n")
    
    # Create tasks
    tasks = create_research_tasks(topic)
    
    # Create the crew
    research_crew = Crew(
        agents=[research_analyst, content_synthesizer, critical_reviewer],
        tasks=tasks,
        process=Process.sequential,  # Tasks run in order
        verbose=True
    )
    
    # Run the crew
    result = research_crew.kickoff()
    
    print("\n" + "="*60)
    print("✅ RESEARCH COMPLETE")
    print("="*60)
    print(f"\n{result}")
    
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*20)
    print("       CREWAI DEEP RESEARCH AGENT")
    print("🚀"*20)
    
    # Get research topic from user
    print("\nEnter a topic to research (or 'quit' to exit):")
    
    while True:
        topic = input("\n🔍 Research Topic: ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not topic:
            print("Please enter a valid topic.")
            continue
        
        try:
            result = run_research_crew(topic)
            
            # Save results to file
            filename = f"research_{topic.replace(' ', '_')[:30]}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Research Topic: {topic}\n")
                f.write("="*60 + "\n\n")
                f.write(str(result))
            print(f"\n📁 Results saved to: {filename}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "-"*40)
        print("Enter another topic or 'quit' to exit")
