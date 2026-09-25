# Day 1 — LLM Engineering Fundamentals

> Interview-ready notes covering the anatomy of an LLM call, project setup with `uv`, tokenization, and debugging.

---

## TL;DR (60-second revision)

| Concept | One-line summary |
|---|---|
| **venv** | Isolated project workroom — prevents dependency hell. |
| **uv** | Modern, fast replacement for `pip` + `venv` + `virtualenv`. |
| **API key** | Password to the LLM provider. Never hardcode → use `.env`. |
| **Client** | Object that manages the HTTP connection to the LLM API. |
| **Model** | Must be spelled *exactly* (e.g. `llama-3.3-70b-versatile`). |
| **Messages** | List of `{role, content}` dicts — carries the conversation. |
| **Roles** | `system` (behavior), `user` (input), `assistant` (reply). |
| **Response** | `choices[0].message.content` = primary answer, `usage` = token stats. |
| **Token** | ~4 chars ≈ ¾ of a word. You pay per million. |

---

## 1. Virtual Environments (`venv`)

### Why
Different projects need different library versions. Example:

- **Project A** requires `numpy==1.24`
- **Project B** requires `numpy==2.0`

A global install breaks one. A **virtual environment** isolates each project so they coexist peacefully.

> **Note:** `venv` isolates Python packages, not OS-level drivers.

### Classic `venv` commands

```bash
# Create
python -m venv venv

# Activate (macOS / Linux)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate
```

**Interview line:** *"I use virtual environments so each project has a reproducible, conflict-free dependency install."*

---

## 2. Project Setup with `uv` (recommended)

`uv` is a fast, modern all-in-one tool from Astral that replaces `pip`, `venv`, and `virtualenv`.

### Full flow

```bash
uv init day1                  # create a new Python project called "day1"
cd day1
uv venv --python 3.11         # create a .venv folder using Python 3.11
source .venv/bin/activate     # macOS/Linux
# or on Windows:  .\.venv\Scripts\Activate.ps1

code hello_llm.py             # open/create the script in VS Code
uv add groq python-dotenv     # install deps into the venv
python hello_llm.py           # run it
```

### What each package does
- **`groq`** — Groq's Python SDK to call the LLM API.
- **`python-dotenv`** — loads variables from `.env` into `os.environ` at runtime.

### Alternative (plain pip)
```bash
pip install groq python-dotenv
```

---

## 3. Anatomy of an LLM API Call

When asked *"walk me through what happens when you send a prompt to an LLM,"* answer in this order:

```text
1. Create client (with API key)
        ↓
2. Choose model
        ↓
3. Build messages (system + user + assistant history)
        ↓
4. Send request
        ↓
5. Receive response
        ↓
6. Extract answer: response.choices[0].message.content
```

### a) Authentication (API Key)
- Secret credential that proves *you* are the caller.
- **Rule:** Never hardcode. Store in `.env`, load via `python-dotenv`.
- Always add `.env` to `.gitignore` so keys never leave your machine.

### b) Client Initialization
The **client** is the object that handles the HTTP connection and request/response serialization.

```python
from groq import Groq
client = Groq(api_key=my_api_key)
```

### c) Model Selection
- Must be **exactly** the string the provider expects (e.g. `llama-3.3-70b-versatile`).
- Typo → connection failure. Always copy from official docs.
- Models differ in: reasoning ability, speed, cost, context window, multimodal support.

### d) Message Structure
The API expects a **list of dictionaries**, not a single string:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Who is Einstein?"},
    {"role": "assistant", "content": "A theoretical physicist..."},
    {"role": "user",      "content": "What was his age when he died?"}
]
```

Each message has two required fields:
- **`role`** — `system`, `user`, or `assistant`.
- **`content`** — the actual text.

### e) The three roles

| Role | Purpose | Example |
|---|---|---|
| `system` | High-level behavior, rules, persona | "You are a helpful ML tutor." |
| `user` | The human's input / question | "What is supervised learning?" |
| `assistant` | The model's previous reply (for context) | "Supervised learning is..." |

### f) Context (the "why" behind the list)
LLMs are **stateless** — they remember nothing between calls. You send the entire conversation every time so references like *"what was his age?"* still make sense.

---

## 4. Handling the Response

```python
response = client.chat.completions.create(model=model, messages=messages)

answer  = response.choices[0].message.content   # primary answer
prompt_tok  = response.usage.prompt_tokens
output_tok  = response.usage.completion_tokens
total_tok   = response.usage.total_tokens
```

- **`choices`** — list of possible outputs. Usually only `choices[0]` matters.
- **`usage`** — token counts. Critical for **cost tracking** and **rate limits**.

Response shape (conceptual):

```text
Response
├── choices[]
│   └── message.content   ← the answer
└── usage
    ├── prompt_tokens
    ├── completion_tokens
    └── total_tokens
```

---

## 5. Tokenization & Cost

- **Token** = fundamental unit LLMs process. Roughly **1 token ≈ 4 characters ≈ ¾ word**.
- A token can be a whole word, part of a word, punctuation, or a symbol.
- Every prompt in + every response out **consumes tokens**.
- Providers charge **per 1M tokens**, with input and output priced separately.
- Track `usage` in every response to monitor spend in production.

**Interview line:** *"I always log `response.usage` so I can attribute cost per feature and stay under rate limits."*

---

## 6. Working Example — `hello_llm.py`

### `.env` (never commit this)
```env
GROQ_API_KEY=your_api_key_here
```

### `hello_llm.py`
```python
import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from the .env file into os.environ
load_dotenv()

# Read the API key
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY is missing")

# Create the client
client = Groq(api_key=my_api_key)

# Choose the model
model = "llama-3.3-70b-versatile"

# Build the messages
messages = [
    {"role": "system", "content": "You are a concise ML tutor."},
    {"role": "user",   "content": "What is Machine Learning?"}
]

# Send the request
response = client.chat.completions.create(model=model, messages=messages)

# Extract the answer + token stats
answer = response.choices[0].message.content
print(answer)
print(f"\nTokens used: {response.usage.total_tokens}")
```

---

## 7. Debugging & Error Handling

When an API call fails, do not panic. Work through this checklist:

1. **Isolate** — comment out lines one at a time until the error disappears. That's your culprit.
2. **Read the traceback** — Python tells you the exact line and error type (`KeyError`, `TypeError`, `AuthenticationError`, etc.).
3. **Check the docs** — field names differ between providers. `content` vs `prompt`, `messages` vs `input`. Always verify.
4. **Common failures:**
   - Invalid / missing API key → check `.env` is loaded (`load_dotenv()` called).
   - Wrong model name → typo or deprecated version.
   - Wrong role name → must be exactly `system` / `user` / `assistant`.
   - Rate limit → back off and retry with exponential delay.
   - `.env` not found → make sure you're running the script from the folder containing `.env`.

---

## 8. Quick-Recall Flashcards

**Q1.** What is an API key?
**A.** A secret credential used to authenticate your application with the LLM provider.

**Q2.** Why send a list of messages instead of a single string?
**A.** LLMs are stateless. The list carries conversation history so follow-up references resolve correctly.

**Q3.** What are the two required fields in each message?
**A.** `role` and `content`.

**Q4.** What are the three standard roles?
**A.** `system` (behavior/rules), `user` (input), `assistant` (prior model replies).

**Q5.** What is a token?
**A.** The smallest unit of text an LLM processes — roughly 4 characters or ¾ of a word. Used for cost and context limits.

**Q6.** How do you estimate the cost of a request?
**A.** Read `response.usage.total_tokens` and multiply by the provider's per-million rate.

**Q7.** How do you extract the answer from the response?
**A.** `response.choices[0].message.content`.

**Q8.** Where do API keys go?
**A.** In `.env`, loaded via `python-dotenv`. Never in source code, never committed.

**Q9.** What's the difference between `venv` and `uv`?
**A.** `venv` is Python's built-in isolated environment tool. `uv` is a faster, all-in-one tool that also handles installs (like `pip`) and project scaffolding.

**Q10.** First step when an API call errors out?
**A.** Read the traceback line-by-line, then isolate by commenting out blocks.

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **API key** | Secret credential to authenticate with the API. |
| **Client** | SDK object that communicates with the LLM API. |
| **Model** | The specific LLM used to generate the response. |
| **Role** | Who sent the message: `system`, `user`, or `assistant`. |
| **Content** | The actual text of a message. |
| **Response** | The full object returned by the API. |
| **Choices** | Array of generated response options in the response. |
| **Usage** | Token count block in the response — used for cost. |
| **Token** | Smallest text unit processed by the model. |
