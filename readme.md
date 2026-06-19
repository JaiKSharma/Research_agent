Steps to create the environment:
    - python -m venv myenv
    - .\myenv\Script\activate


Run:
    pip install -r requirements.txt
    put relevant api key .env file
    select relevant LLM provider (LLM_PROVIDER="groq")
    python main.py "What caused the 2021 Texas grid failure and did the fixes hold?"
    # search uses keyless DuckDuckGo — no second API key needed