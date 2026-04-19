import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
import datetime

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="AI FPGI Monitor", layout="wide")
st.title("⚖️ FPGI Monitor: The AI Precaution Benchmark")
st.markdown("""
This tool autonomously calculates the **Foreseeability–Precaution Gap Index (FPGI)** for AI incidents.
It helps resolve Triad tensions by measuring if a harm was legally foreseeable based on prior signals.
""")

# --- CALCULATOR LOGIC (BASED ON WEIL & CLASS NOTES) ---
# Formula: FPGI = (S + C) - P
# S = Warning Signals, C = Control, P = Precaution
def calculate_fpgi(signals, control, precaution):
    """Calculates the FPGI score based on the proposed formula."""
    return (signals + control) - precaution

# --- AGENTIC CLASSIFICATION ENGINE ---
def analyze_incident_with_agent(incident_text):
    """
    Uses an LLM agent to 'code' the incident according to the project's legal framework.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Define the schema for the agent's 'thought' process
    response_schemas = [
        ResponseSchema(name="S", description="Warning Signals score (0-5). High if red-teaming/evals flagged this capability."),
        ResponseSchema(name="C", description="Control score (0-5). High if closed API, low if open-weight/unsupervised."),
        ResponseSchema(name="P", description="Precaution score (0-5). High if staged rollout, kill-switches, or human review used."),
        ResponseSchema(name="analysis", description="A brief legal analysis applying Ryobi/Kuehn logic.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_template(
        "You are a legal-tech agent specializing in AI tort law. Analyze the following incident:\n"
        "Incident: {incident}\n\n"
        "Apply these principles:\n"
        "1. Warning Signals (S): Did the developer have prior signals (failed evals/red-teaming)?\n"
        "2. Control (C): Did the developer maintain active oversight?\n"
        "3. Precaution (P): Were safeguards (kill-switches/staged rollout) active?\n\n"
        "{format_instructions}"
    )
    _input = prompt.format_prompt(incident=incident_text, format_instructions=format_instructions)
    output = llm(_input.to_messages())
    return output_parser.parse(output.content)

# --- SIDEBAR: ADD NEW INCIDENT ---
with st.sidebar:
    st.header("Ingest New Incident")
    new_title = st.text_input("Incident Title")
    new_text = st.text_area("Incident Description / News Snippet")
    if st.button("Run Agentic Analysis"):
        with st.spinner("Agent is coding the incident..."):
            result = analyze_incident_with_agent(new_text)
            fpgi_score = calculate_fpgi(int(result['S']), int(result['C']), int(result['P']))

            # Save to 'Database' (Session State)
            new_entry = {
                "Date": datetime.date.today(),
                "Incident": new_title,
                "S": result['S'],
                "C": result['C'],
                "P": result['P'],
                "FPGI": fpgi_score,
                "Analysis": result['analysis']
            }
            if 'data' not in st.session_state:
                st.session_state.data = []
            st.session_state.data.append(new_entry)

# --- MAIN DASHBOARD ---
if 'data' in st.session_state and st.session_state.data:
    df = pd.DataFrame(st.session_state.data)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average FPGI", round(df["FPGI"].mean(), 2))
    col2.metric("Highest Risk Case", df.loc[df['FPGI'].idxmax()]['Incident'])
    col3.metric("Total Incidents Tracked", len(df))

    # Visualization
    st.subheader("The Living Benchmark")
    st.line_chart(df.set_index("Date")["FPGI"])

    # Detailed Data
    st.write("### Incident Breakdown")
    st.table(df[["Date", "Incident", "FPGI", "S", "C", "P"]])

    for _, row in df.iterrows():
        with st.expander(f"Legal Analysis: {row['Incident']}"):
            st.write(row['Analysis'])
else:
    st.info("No incidents tracked yet. Use the sidebar to ingest a new AI failure incident.")

# --- FOOTER ---
st.markdown("---")
st.caption("Benchmark Methodology: FPGI = (Signals + Control) - Precaution. References: Weil (2024), Hood v. Ryobi (1999), Kuehn v. Inter-city (1979).")
