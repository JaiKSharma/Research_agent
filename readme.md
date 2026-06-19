# Research Agent

An AI-powered research agent built using LangGraph and Large Language Models (LLMs) that performs multi-step web research, synthesizes findings, and generates comprehensive answers to complex research questions.

## Features

* Agentic workflow powered by LangGraph
* Multi-step reasoning and planning
* Web search integration using DuckDuckGo (no search API key required)
* Support for multiple LLM providers
* Environment-based configuration
* Structured research and answer generation

---

## Tech Stack

* Python
* LangGraph
* LangChain
* Groq (default LLM provider)
* DuckDuckGo Search
* dotenv for configuration management

---

## Project Setup

### 1. Create and Activate Virtual Environment

Windows PowerShell:

```bash
python -m venv myenv
.\myenv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and add the required API keys.

Example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Agent

Example query:

```bash
python main.py "What caused the 2021 Texas grid failure and did the fixes hold?"
```

### Notes

* Search uses DuckDuckGo and does not require a separate API key.
* Ensure the appropriate API key is configured for the selected LLM provider.
* Change the provider in the `.env` file if multiple providers are supported.

---

## Example Workflow

```text
User Query
    │
    ▼
Planner Node
    │
    ▼
Web Research
    │
    ▼
Information Synthesis
    │
    ▼
Final Report Generation
```

---

## Project Structure

```text
Research_agent/
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## Sample Questions

* What caused the 2021 Texas grid failure and did the fixes hold?
* Compare LangGraph, CrewAI, and AutoGen for enterprise AI agents.
* What are the latest techniques for RAG evaluation?
* Summarize recent developments in agentic AI systems.

---

## Future Improvements

* Human-in-the-loop review
* Multi-agent collaboration
* Evaluation and benchmarking framework
* Long-term memory support
* Advanced tool routing
* Report export to PDF and Markdown

---

## License

This project is intended for educational, research, and experimentation purposes.
