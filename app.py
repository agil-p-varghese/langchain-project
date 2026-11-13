import os
from typing import List, Dict, Any, TypedDict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
# Replaced OpenAI with Hugging Face LangChain integrations
#from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

#load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="LangGraph Resume Builder API (Hugging Face)")
templates = Jinja2Templates(directory="templates")

#------------------------------------------
#--------LLM initialization--------------
#-----------------------------------------



HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face API token is missing")

llm = ChatOpenAI(
    model_name="openai/gpt-oss-120b:fastest",
    openai_api_key=HF_TOKEN,
    openai_api_base="https://router.huggingface.co/v1",
    max_tokens=1024
)




# ==========================================
# 1. SHARED STATE DEFINITION
# ==========================================
class ResumeState(TypedDict):
    resume_text: str
    jd_text: str
    analysis: str
    advice_options: List[Dict[str, Any]]  # Multiple-choice options with text & id
    selected_modifications: List[str]     # User-checked modifications
    final_resume: str

# ==========================================
# 2. LANGGRAPH NODES
# ==========================================
def analyzer_node(state: ResumeState) -> ResumeState:
    """Node 1: Analyzes resume against JD and identifies gaps."""
    prompt = f"""
    You are an expert ATS and Technical Recruiter. Analyze the following resume against the Job Description (JD).
    
    JOB DESCRIPTION:
    {state['jd_text']}
    
    CANDIDATE RESUME:
    {state['resume_text']}
    
    Provide a detailed analysis covering:
    1. Missing Skills & Tech Stack items.
    2. Gaps in project quality/depth.
    3. Alignment score and overall critique.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["analysis"] = response.content
    return state

def advisor_node(state: ResumeState) -> ResumeState:
    """Node 2: Generates multi-select options for resume alignment."""
    prompt = f"""
    Based on the following analysis of gaps between a resume and JD, generate 4 to 6 concise, actionable modification options that the user can select via checkboxes.
    
    ANALYSIS:
    {state['analysis']}
    
    Return your response strictly as a bulleted list where each bullet represents a distinct modification choice starting with a short ID tag like [OPT1], [OPT2], etc.
    Example:
    - [OPT1] Add FastAPI asynchronous endpoint examples to your backend project section.
    - [OPT2] Highlight experience with Docker and PostgreSQL container orchestration.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse LLM response into structured selectable options
    options = []
    for line in response.content.split("\n"):
        if "[" in line and "]" in line:
            parts = line.split("]", 1)
            opt_id = parts[0].replace("[", "").replace("-", "").strip()
            opt_text = parts[1].strip("- ").strip()
            options.append({"id": opt_id, "text": opt_text})
            
    # Fallback if parsing fails
    if not options:
        options = [{"id": "OPT1", "text": "Incorporate key missing tech stack terms from JD into project descriptions."}]
        
    state["advice_options"] = options
    return state


def writer_node(state: ResumeState) -> ResumeState:
    """Node 3: Rewrites the resume using only selected advisor recommendations."""

    # Get advisor options from shared state
    advice_options = state.get("advice_options", [])

    # Get selected option IDs from shared state
    selected_ids = state.get("selected_modifications", [])

    # Convert selected IDs into recommendation descriptions
    selected_descriptions = [
        option["text"]
        for option in advice_options
        if option["id"] in selected_ids
    ]

    if not selected_descriptions:
        raise ValueError("No valid modifications selected.")

    modifications = "\n".join(
        f"- {description}" for description in selected_descriptions
    )

    prompt = f"""
    You are an expert technical resume writer.

    Rewrite the candidate's original resume using only the selected
    recommendations while keeping all information factually accurate.

    SELECTED MODIFICATIONS:
    {modifications}

    ORIGINAL RESUME:
    {state['resume_text']}

    JOB DESCRIPTION:
    {state['jd_text']}

    RESUME ANALYSIS:
    {state['analysis']}

    STRICT RULES:
    1. Never invent skills, experience, projects or achievements.
    2. Never fabricate phone numbers, email addresses or URLs.
    3. Never claim experience with technologies not present
       in the original resume.
    4. Never invent percentages, performance metrics or numbers.
    5. Only incorporate selected modifications where supported
       by the original resume.
    6. Preserve the candidate's original factual information.
    7. Improve wording, formatting and relevance to the JD.
    8. Return the complete resume in professional Markdown.
    9. Do not include explanations or commentary outside the resume.

    Generate the final resume now.
    """

    print("Calling LLM for resume generation...")

    response = llm.invoke([HumanMessage(content=prompt)])

    print("========== LLM RESPONSE DEBUG ==========")
    print("Response content:", repr(response.content))
    print("Response metadata:", response.response_metadata)
    #print("Token usage:", response.usage_metadata)
    print("========================================")

    if not response.content:
        raise ValueError("LLM returned an empty response.")

    state["final_resume"] = response.content

    return state


# ==========================================
# 3. BUILD LANGGRAPH WORKFLOW
# ==========================================
workflow = StateGraph(ResumeState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("advisor", advisor_node)
#workflow.add_node("writer", writer_node)

workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "advisor")
# We stop/interrupt after advisor to allow user checkbox selection on frontend
workflow.add_edge("advisor", END)

app_graph = workflow.compile()

# Global in-memory session store for demo persistence between HTTP calls
session_store = {}

# ==========================================
# 4. API & FRONTEND ROUTES
# ==========================================
class Step1Request(BaseModel):
    resume_text: str
    jd_text: str

class Step2Request(BaseModel):
    session_id: str
    selected_modifications: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def run_analysis(payload: Step1Request):
    import uuid
    import traceback

    print("1. Analyze endpoint called")

    try:
        session_id = str(uuid.uuid4())

        initial_state = {
            "resume_text": payload.resume_text,
            "jd_text": payload.jd_text,
            "analysis": "",
            "advice_options": [],
            "selected_modifications": [],
            "final_resume": ""
        }

        print("2. Starting LangGraph execution")

        final_state = app_graph.invoke(initial_state)

        print("3. LangGraph execution completed")

        session_store[session_id] = final_state

        return {
            "session_id": session_id,
            "analysis": final_state["analysis"],
            "advice_options": final_state["advice_options"]
        }

    except Exception as e:
        print("ERROR DURING ANALYSIS:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def run_writer(payload: Step2Request):

    if payload.session_id not in session_store:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired."
        )

    try:
        state = session_store[payload.session_id]

        print("\n========== WRITER DEBUG ==========")
        print("Received session ID:", payload.session_id)
        print("Received selected IDs:", payload.selected_modifications)

        state["selected_modifications"] = payload.selected_modifications

        updated_state = writer_node(state)

        session_store[payload.session_id] = updated_state

        return {
            "final_resume": updated_state["final_resume"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)