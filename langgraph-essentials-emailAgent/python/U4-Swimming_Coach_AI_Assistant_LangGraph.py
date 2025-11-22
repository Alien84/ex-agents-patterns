"""
SCAS: Swimming Coach AI System

Overview:
SCAS demonstrates how to build an intelligent multi-agent system that transforms
personalized swim training through AI-powered coaching. Using LangGraph's workflow
framework, the system coordinates multiple specialized agents to provide comprehensive
training plans, technique analysis, performance tracking, and personalized coaching advice.

System Architecture:
- Coordinator Agent: Orchestrates interactions between specialized agents
- Training Planner Agent: Handles workout scheduling and training plan generation
- Technique Analyst Agent: Analyzes swimming technique and provides drill recommendations
- Performance Coach Agent: Tracks progress and provides motivational coaching
"""

# =============================================================================
# Setup and Installation
# =============================================================================

# Install required packages (run this in terminal if needed):
# pip install langgraph langchain langchain-openai langchain-google-genai python-dotenv

# Import necessary libraries
import os
from typing import TypedDict, Annotated, List, Dict, Optional
from datetime import datetime, timedelta
import operator

from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# LLM Configuration
# =============================================================================

# Choose your LLM provider
USE_OPENAI = True  # Set to False to use Google Gemini

if USE_OPENAI:
    # Set OPENAI_API_KEY in your environment or .env file
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
else:
    # Set GOOGLE_API_KEY in your environment or .env file
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    pass

# =============================================================================
# State Definition
# =============================================================================

class SwimmerState(TypedDict):
    """State object that maintains swimmer information throughout the workflow"""

    # Swimmer Profile
    swimmer_name: str
    age: int
    gender: str
    experience_level: str  # beginner, intermediate, advanced, elite
    preferred_stroke: str

    # Current Performance
    current_pb_100m: float  # Personal best in seconds
    current_volume_weekly: int  # meters per week
    recent_times: List[float]  # Recent 100m times

    # Goals
    goal_type: str  # fitness, technique, race
    goal_pb_100m: Optional[float]  # Target time if racing
    race_distance: Optional[int]  # 50, 100, 200, 400, 800, 1500
    race_date: Optional[str]  # ISO format date
    weeks_to_goal: int

    # Availability
    sessions_per_week: int
    minutes_per_session: int
    preferred_training_days: List[str]

    # Training Context
    current_week: int
    training_phase: str  # base, build, peak, taper, recovery

    # Agent Communications
    user_query: str
    query_type: str  # workout_request, technique_question, progress_check, schedule_plan

    # Outputs from each agent
    coordinator_response: str
    training_plan: Dict
    technique_analysis: str
    performance_feedback: str

    # Final response
    final_response: str

    # Messages for routing
    messages: Annotated[List[str], operator.add]

# =============================================================================
# Agent 1: Coordinator Agent
# =============================================================================

def coordinator_agent(state: SwimmerState) -> SwimmerState:
    """
    Analyzes the user query and determines which specialized agent(s) should handle it.
    Routes queries about:
    - Training plans → Training Planner
    - Technique questions → Technique Analyst
    - Performance/progress → Performance Coach
    - Complex queries → Multiple agents
    """

    user_query = state["user_query"]
    swimmer_name = state["swimmer_name"]

    # Create prompt for query classification
    system_prompt = """You are a Swimming Coach Coordinator AI. Analyze the swimmer's query and classify it into one of these categories:

    1. workout_request: User wants a training plan, specific workout, or schedule
    2. technique_question: User asks about swimming technique, drills, or form improvement
    3. progress_check: User wants performance analysis, progress update, or feedback
    4. general_advice: General swimming questions or motivational support
    5. multi_agent: Query requires multiple agents (e.g., "Create a training plan and analyze my technique")

    Respond with ONLY the category name (lowercase, underscore-separated).
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Swimmer's query: {user_query}")
    ]

    response = llm.invoke(messages)
    query_type = response.content.strip().lower()

    # Generate coordinator response
    coordinator_response = f"Hello {swimmer_name}! I've analyzed your request. Let me get the right specialist to help you."

    state["query_type"] = query_type
    state["coordinator_response"] = coordinator_response
    state["messages"] = [f"Coordinator: Routing query as '{query_type}'"]

    return state

# =============================================================================
# Agent 2: Training Planner Agent
# =============================================================================

def training_planner_agent(state: SwimmerState) -> SwimmerState:
    """
    Creates personalized training plans using:
    - Swimmer's current level and goals
    - Training phase (base/build/peak/taper)
    - Available time and frequency
    - Workout database and personalization algorithm
    """

    # Extract relevant information
    swimmer_profile = f"""
    Swimmer: {state['swimmer_name']}
    Level: {state['experience_level']}
    Current PB (100m): {state['current_pb_100m']}s
    Goal: {state['goal_type']}
    Target PB: {state.get('goal_pb_100m', 'N/A')}s
    Race Distance: {state.get('race_distance', 'N/A')}m
    Weeks to Goal: {state['weeks_to_goal']}
    Sessions/Week: {state['sessions_per_week']}
    Minutes/Session: {state['minutes_per_session']}
    Current Week: {state['current_week']}
    Training Phase: {state['training_phase']}
    Preferred Stroke: {state['preferred_stroke']}
    """

    system_prompt = f"""
    You are an expert Swimming Training Planner AI. Based on the swimmer's profile,
    create a detailed training plan for this week.

    PERSONALIZATION GUIDELINES:
    1. Calculate pace zones based on current PB:
       - Easy pace: +20% slower than PB
       - Endurance pace: +12% slower
       - Threshold pace: +5% slower
       - Race pace: Goal PB time
       - Sprint pace: 5% faster than goal

    2. Adjust volume for training phase:
       - Base building: Higher volume (70% endurance)
       - Build: Mixed volume/intensity
       - Peak: Race-specific, moderate volume
       - Taper: 60% volume, maintain intensity

    3. Structure each workout:
       - Warmup (15% of session)
       - Main set (70% of session)
       - Cooldown (15% of session)

    4. Include:
       - Specific distances and intervals
       - Target paces for each set
       - Rest intervals
       - Focus points (technique cues)
       - Estimated RPE (1-10)

    Swimmer Profile:
    {swimmer_profile}

    Generate a detailed weekly training plan with {state['sessions_per_week']} sessions.
    Format each workout clearly with warmup, main set, and cooldown.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {state['user_query']}")
    ]

    response = llm.invoke(messages)
    training_plan_text = response.content

    # Parse into structured format (simplified)
    training_plan = {
        "week_number": state['current_week'],
        "phase": state['training_phase'],
        "plan_details": training_plan_text,
        "generated_at": datetime.now().isoformat()
    }

    state["training_plan"] = training_plan
    state["messages"] = ["Training Planner: Generated personalized training plan"]

    return state

# =============================================================================
# Agent 3: Technique Analyst Agent
# =============================================================================

def technique_analyst_agent(state: SwimmerState) -> SwimmerState:
    """
    Provides technique analysis and drill recommendations.
    Covers:
    - Stroke mechanics (catch, pull, recovery)
    - Body position and streamlining
    - Breathing patterns
    - Kick technique
    - Turns and underwater work
    """

    stroke = state['preferred_stroke']
    level = state['experience_level']

    system_prompt = f"""
    You are an expert Swimming Technique Coach AI specializing in {stroke}.

    TECHNIQUE KNOWLEDGE BASE:

    FREESTYLE:
    - Body position: Horizontal, head neutral, hips high
    - Catch: High elbow, fingertips down, press chest
    - Pull: Straight line pull, accelerate to hip
    - Recovery: Relaxed, elbow leads, hand enters thumb first
    - Kick: From hips, pointed toes, 2-beat or 6-beat
    - Breathing: Rotate to air pocket, one goggle in water

    BACKSTROKE:
    - Body position: Flat on back, hips high, slight lean
    - Arm entry: Pinky first, straight arm overhead
    - Pull: Bent elbow underwater, push to thigh
    - Recovery: Straight arm, shoulder roll
    - Kick: Flutter kick, toes pointed, from hips

    BREASTSTROKE:
    - Timing: Pull-breathe-kick-glide
    - Pull: Out-around-in, elbows high, hands don't pass shoulders
    - Kick: Heels to hips, turn feet out, narrow kick
    - Glide: Streamline position, maximize distance
    - Breathing: Lift chin, quick breath, head down in glide

    BUTTERFLY:
    - Body motion: Undulation from chest, wave motion
    - Arm pull: Simultaneous, keyhole pattern underwater
    - Recovery: Over water, relaxed, thumb leads entry
    - Kick: Two kicks per arm cycle, from core
    - Breathing: Forward, every 1-2 strokes

    DRILLS LIBRARY:
    - Catch-up drill: One arm at a time, focus on catch
    - Fingertip drag: Recovery with fingertips on water
    - Kick with board: Isolate kick technique
    - Single arm: Swim with one arm, other at side
    - Fist drill: Swim with closed fists, feel forearm
    - 6-kick switch: 6 kicks on side, switch arms

    Swimmer Level: {level}
    Preferred Stroke: {stroke}

    Provide specific, actionable technique advice based on the swimmer's question.
    Include:
    1. Technical explanation (2-3 key points)
    2. Common mistakes to avoid
    3. Specific drill recommendations (2-3 drills)
    4. Practice structure for improvement
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Swimmer's question: {state['user_query']}")
    ]

    response = llm.invoke(messages)
    technique_analysis = response.content

    state["technique_analysis"] = technique_analysis
    state["messages"] = ["Technique Analyst: Provided technique feedback and drills"]

    return state

# =============================================================================
# Agent 4: Performance Coach Agent
# =============================================================================

def performance_coach_agent(state: SwimmerState) -> SwimmerState:
    """
    Provides performance analysis and motivational coaching.
    - Analyzes recent times and trends
    - Provides feedback on progress
    - Suggests training adjustments
    - Offers motivation and mental strategies
    """

    # Analyze performance trend
    recent_times = state.get('recent_times', [])
    current_pb = state['current_pb_100m']
    goal_pb = state.get('goal_pb_100m')
    weeks_to_goal = state['weeks_to_goal']

    # Calculate trend
    if len(recent_times) >= 3:
        recent_avg = sum(recent_times[-3:]) / 3
        if len(recent_times) >= 6:
            previous_avg = sum(recent_times[-6:-3]) / 3
            improvement = ((previous_avg - recent_avg) / previous_avg) * 100
            trend = "improving" if improvement > 1 else ("declining" if improvement < -1 else "stable")
        else:
            trend = "insufficient_data"
    else:
        trend = "insufficient_data"

    performance_context = f"""
    Current PB: {current_pb}s
    Goal PB: {goal_pb}s (need to improve by {current_pb - goal_pb:.1f} seconds)
    Weeks remaining: {weeks_to_goal}
    Recent times: {recent_times}
    Performance trend: {trend}
    Training phase: {state['training_phase']}
    """

    system_prompt = f"""
    You are an expert Performance Coach AI. Provide analysis and coaching based on:

    PERFORMANCE ANALYSIS FRAMEWORK:
    1. Progress Assessment:
       - Compare recent times to PB and goal
       - Identify trends (improving/plateauing/declining)
       - Calculate required improvement rate

    2. Training Recommendations:
       - If improving: Maintain current approach
       - If plateauing: Suggest intensity/variety changes
       - If declining: Recommend recovery or volume reduction

    3. Mental Coaching:
       - Acknowledge progress and effort
       - Set realistic short-term goals
       - Provide race preparation tips if applicable
       - Build confidence and motivation

    4. Periodization Advice:
       - Base phase: Focus on volume and consistency
       - Build phase: Increase intensity, lactate work
       - Peak phase: Race-specific training
       - Taper: Reduce volume, maintain intensity

    Performance Context:
    {performance_context}

    Provide:
    1. Performance analysis (progress toward goal)
    2. Training adjustments if needed
    3. Motivational coaching message
    4. Next steps and focus areas
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Swimmer's request: {state['user_query']}")
    ]

    response = llm.invoke(messages)
    performance_feedback = response.content

    state["performance_feedback"] = performance_feedback
    state["messages"] = ["Performance Coach: Analyzed progress and provided coaching"]

    return state

# =============================================================================
# Response Synthesizer
# =============================================================================

def synthesize_response(state: SwimmerState) -> SwimmerState:
    """
    Combines outputs from multiple agents into a coherent final response.
    """

    query_type = state["query_type"]
    final_response = state["coordinator_response"] + "\n\n"

    # Add relevant agent outputs based on query type
    if query_type == "workout_request":
        if state.get("training_plan"):
            final_response += "## YOUR TRAINING PLAN\n\n"
            final_response += state["training_plan"]["plan_details"]

    elif query_type == "technique_question":
        if state.get("technique_analysis"):
            final_response += "## TECHNIQUE ANALYSIS\n\n"
            final_response += state["technique_analysis"]

    elif query_type == "progress_check":
        if state.get("performance_feedback"):
            final_response += "## PERFORMANCE ANALYSIS\n\n"
            final_response += state["performance_feedback"]

    elif query_type == "multi_agent":
        # Include all relevant outputs
        if state.get("training_plan"):
            final_response += "\n## TRAINING PLAN\n\n"
            final_response += state["training_plan"]["plan_details"]

        if state.get("technique_analysis"):
            final_response += "\n\n## TECHNIQUE GUIDANCE\n\n"
            final_response += state["technique_analysis"]

        if state.get("performance_feedback"):
            final_response += "\n\n## PERFORMANCE INSIGHTS\n\n"
            final_response += state["performance_feedback"]

    else:  # general_advice
        # Use LLM to provide general guidance
        system_prompt = f"""
        You are a helpful Swimming Coach AI. Provide friendly, encouraging advice
        for general swimming questions. Keep responses concise and actionable.

        Swimmer: {state['swimmer_name']}
        Level: {state['experience_level']}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state['user_query'])
        ]

        response = llm.invoke(messages)
        final_response += response.content

    state["final_response"] = final_response
    state["messages"] = ["Synthesizer: Combined all agent outputs"]

    return state

# =============================================================================
# Routing Logic
# =============================================================================

def route_query(state: SwimmerState) -> str:
    """
    Routes to appropriate agent based on query type.
    """
    query_type = state.get("query_type", "general_advice")

    routing_map = {
        "workout_request": "training_planner",
        "technique_question": "technique_analyst",
        "progress_check": "performance_coach",
        "multi_agent": "training_planner",  # Start with planner, will chain to others
        "general_advice": "synthesizer"
    }

    return routing_map.get(query_type, "synthesizer")

def should_analyze_technique(state: SwimmerState) -> str:
    """Check if technique analysis is needed."""
    if state["query_type"] == "multi_agent":
        return "technique_analyst"
    return "synthesizer"

def should_analyze_performance(state: SwimmerState) -> str:
    """Check if performance analysis is needed."""
    if state["query_type"] == "multi_agent":
        return "performance_coach"
    return "synthesizer"

# =============================================================================
# Build the LangGraph Workflow
# =============================================================================

# Create the state graph
workflow = StateGraph(SwimmerState)

# Add all nodes (agents)
workflow.add_node("coordinator", coordinator_agent)
workflow.add_node("training_planner", training_planner_agent)
workflow.add_node("technique_analyst", technique_analyst_agent)
workflow.add_node("performance_coach", performance_coach_agent)
workflow.add_node("synthesizer", synthesize_response)

# Set entry point
workflow.set_entry_point("coordinator")

# Add conditional edges from coordinator
workflow.add_conditional_edges(
    "coordinator",
    route_query,
    {
        "training_planner": "training_planner",
        "technique_analyst": "technique_analyst",
        "performance_coach": "performance_coach",
        "synthesizer": "synthesizer"
    }
)

# Add edges from specialized agents
workflow.add_conditional_edges(
    "training_planner",
    should_analyze_technique,
    {
        "technique_analyst": "technique_analyst",
        "synthesizer": "synthesizer"
    }
)

workflow.add_conditional_edges(
    "technique_analyst",
    should_analyze_performance,
    {
        "performance_coach": "performance_coach",
        "synthesizer": "synthesizer"
    }
)

workflow.add_edge("performance_coach", "synthesizer")

# Set finish point
workflow.add_edge("synthesizer", END)

# Compile the graph
app = workflow.compile()

# =============================================================================
# Sample Swimmer Profile
# =============================================================================

sample_swimmer = {
    "swimmer_name": "Alex",
    "age": 28,
    "gender": "male",
    "experience_level": "intermediate",
    "preferred_stroke": "freestyle",

    "current_pb_100m": 65.0,
    "current_volume_weekly": 10000,
    "recent_times": [66.0, 65.5, 64.8, 65.2, 64.5],

    "goal_type": "race",
    "goal_pb_100m": 58.0,
    "race_distance": 100,
    "race_date": "2025-03-15",
    "weeks_to_goal": 16,

    "sessions_per_week": 4,
    "minutes_per_session": 60,
    "preferred_training_days": ["Monday", "Wednesday", "Friday", "Saturday"],

    "current_week": 3,
    "training_phase": "base_building",

    "user_query": "",
    "query_type": "",
    "coordinator_response": "",
    "training_plan": {},
    "technique_analysis": "",
    "performance_feedback": "",
    "final_response": "",
    "messages": []
}

# =============================================================================
# Interactive Chat Interface
# =============================================================================

def swimming_coach_chat():
    """
    Interactive chat interface for the Swimming Coach AI
    """
    print("SWIMMING COACH AI SYSTEM")
    print("=" * 80)
    print("Welcome! I'm your AI swimming coach.")
    print("I can help you with:")
    print("  - Training plans and workout recommendations")
    print("  - Technique analysis and drill suggestions")
    print("  - Performance tracking and progress analysis")
    print("  - Motivation and race preparation")
    print("\nType 'quit' to exit")
    print("=" * 80)

    # Initialize swimmer state
    swimmer_state = sample_swimmer.copy()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThanks for training with me! Keep swimming strong!")
            break

        if not user_input:
            continue

        # Update query and reset outputs
        swimmer_state["user_query"] = user_input
        swimmer_state["messages"] = []
        swimmer_state["final_response"] = ""

        # Invoke the agent system
        result = app.invoke(swimmer_state)

        # Display response
        print("\nCoach AI:")
        print("-" * 80)
        print(result["final_response"])
        print("-" * 80)

        # Update state for next iteration
        swimmer_state = result

# =============================================================================
# Example Usage Functions
# =============================================================================

def run_example_1():
    """Example 1: Request Training Plan"""
    swimmer_state = sample_swimmer.copy()
    swimmer_state["user_query"] = "Can you create my training plan for this week? I want to focus on building endurance."

    result = app.invoke(swimmer_state)

    print("=" * 80)
    print("QUERY:", result["user_query"])
    print("=" * 80)
    print("\n" + result["final_response"])
    print("\n" + "=" * 80)

def run_example_2():
    """Example 2: Technique Question"""
    swimmer_state = sample_swimmer.copy()
    swimmer_state["user_query"] = "How can I improve my freestyle catch? I feel like I'm not getting enough power in my pull."

    result = app.invoke(swimmer_state)

    print("=" * 80)
    print("QUERY:", result["user_query"])
    print("=" * 80)
    print("\n" + result["final_response"])
    print("\n" + "=" * 80)

def run_example_3():
    """Example 3: Performance Check"""
    swimmer_state = sample_swimmer.copy()
    swimmer_state["user_query"] = "How am I progressing toward my goal? Am I on track to hit 58 seconds by March?"

    result = app.invoke(swimmer_state)

    print("=" * 80)
    print("QUERY:", result["user_query"])
    print("=" * 80)
    print("\n" + result["final_response"])
    print("\n" + "=" * 80)

def run_example_4():
    """Example 4: Multi-Agent Query"""
    swimmer_state = sample_swimmer.copy()
    swimmer_state["user_query"] = """I need a complete assessment:
1. Create this week's training plan
2. Analyze my freestyle technique (especially my breathing)
3. Tell me how I'm progressing toward my 58-second goal
"""

    result = app.invoke(swimmer_state)

    print("=" * 80)
    print("QUERY:", result["user_query"])
    print("=" * 80)
    print("\n" + result["final_response"])
    print("\n" + "=" * 80)

# =============================================================================
# Database Integration Examples (Pseudo-code)
# =============================================================================

def load_swimmer_profile(swimmer_id: str) -> SwimmerState:
    """
    Load swimmer profile from database.
    In production, this would connect to your PostgreSQL database.
    """
    # Pseudo-code for database query
    # conn = psycopg2.connect(DATABASE_URL)
    # cursor = conn.cursor()
    # cursor.execute("SELECT * FROM swimmers WHERE id = %s", (swimmer_id,))
    # profile = cursor.fetchone()

    # For demo, return sample profile
    return sample_swimmer.copy()

def retrieve_workout_from_db(profile: SwimmerState, training_phase: str):
    """
    Retrieve appropriate workout from 1000-workout database.
    """
    # Pseudo-code for workout retrieval
    # query = """
    # SELECT * FROM workouts
    # WHERE target_level = %s
    #   AND target_goal = %s
    #   AND training_phase = %s
    # ORDER BY avg_rating DESC
    # LIMIT 1
    # """
    # cursor.execute(query, (profile['experience_level'], profile['goal_type'], training_phase))
    # workout = cursor.fetchone()

    pass

def log_workout_completion(swimmer_id: str, workout_id: int, actual_distance: int, completion_rate: float):
    """
    Log completed workout to track history.
    """
    # INSERT INTO user_workouts (swimmer_id, workout_id, completed_date, actual_distance, completion_rate)
    # VALUES (%s, %s, NOW(), %s, %s)
    pass

def sync_watch_data(swimmer_id: str):
    """
    Import swim data from Garmin/Apple Watch.
    """
    # Connect to Garmin API or Apple HealthKit
    # Fetch recent swim activities
    # Parse distance, duration, stroke count, SWOLF
    # Update swimmer profile with latest data
    pass

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            run_example_1()
        elif example == "2":
            run_example_2()
        elif example == "3":
            run_example_3()
        elif example == "4":
            run_example_4()
        else:
            print("Usage: python Swimming_Coach_AI_Assistant_LangGraph.py [1|2|3|4]")
            print("Or run without arguments for interactive chat")
    else:
        # Run interactive chat by default
        swimming_coach_chat()
