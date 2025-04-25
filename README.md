# Generative-AI
This repo contains examples and discussions about generatative AI models and subjects related to these models.

## Agentic AI vs. Traditional LLM
Here are the differences between agentic AI and traditional LLM. 

### 1. Traditional LLM (e.g., GPT-4)
* What it does: A large language model (LLM) like GPT-4 is a statistical predictor—it generates text by predicting the next most likely tokens (words/subwords) based on its training data.

- Key Traits:

    - Passive: Responds to prompts but doesn’t take initiative or "act" beyond generating text.

    - Stateless: Each interaction is independent (unless explicitly designed otherwise, like in a chat session with memory).

    - Single-step: Answers one query at a time without planning or breaking down tasks.

    - No autonomy: Cannot execute actions (e.g., browse the web, run code, or book a flight) unless integrated with external tools (like ChatGPT plugins).

#### Example:
You ask GPT-4, *"Write a Python script to analyze this CSV file."* It generates the code but doesn’t run it, debug errors, or refine it unless you manually iterate.


### 2. Agentic AI
- **What it does**: An agent is an AI system that **plans, acts, and iterates** to achieve goals autonomously or semi-autonomously. It often combines an LLM with tools (APIs, code execution, search), memory, and decision-making loops.

- **Key Traits:**

    - **Proactive:** Can break down tasks into subtasks, set goals, and take steps to achieve them.

    - **Stateful:** Remembers past interactions and learns from feedback within a session.

    - **Multi-step:** Can chain actions (e.g., search the web → summarize → draft an email → send it via API).

    - **Tool use:** Integrates with external systems (calculators, databases, browsers) to do things, not just talk about them.

    - **Self-correction:** Can critique and refine its own outputs (e.g., "This code failed; let me fix the error.").


#### Example:
You ask an agent, *"Find the latest research on LLM quantization techniques and summarize it for me."* The agent might:

1. Search the web for recent papers,

2. Extract key points,

3. Draft a summary,

4. Ask you if you’d like it formatted as a bullet list or a report.


### 3. Key Differences


| Feature          | Traditional LLM (e.g., GPT-4)       | Agentic AI                     |
|------------------|-------------------------------------|--------------------------------|
| **Autonomy**     | Reactive (responds to prompts)      | Proactive (plans and acts)     |
| **Memory**       | Limited/session-based context       | Long-term state retention      |
| **Task Scope**   | Single-step text generation         | Multi-step problem-solving     |
| **Tools**        | No native tool integration          | Uses APIs, code, search, etc.  |
| **Adaptation**   | Static responses                    | Self-corrects via feedback     |
| **Output**       | Text/code generation                | Actions (e.g., emails, bookings) |
| **Example Use**  | Answering Q&A, drafting content     | Booking flights, debugging code |


| Feature          | Traditional LLM (e.g., GPT-4)       | Agentic AI                     | GPT-4 Turbo           | Claude 3 Opus         | Gemini 1.5 Pro        |
|------------------|-------------------------------------|--------------------------------|-----------------------|-----------------------|-----------------------|
| **Autonomy**     | ❌ Reactive                         | ✅ Proactive (plans/acts)       | ❌ (unless plugins)    | ❌                     | ❌ (but API supports tools) |
| **Memory**       | ⏳ Short-term session               | 🗃️ Long-term state             | ⏳ (128k context)      | ⏳ (200k context)      | ⏳ (1M token context) |
| **Tool Use**     | ❌ (text-only)                      | ✅ APIs, search, code           | ✅ (via plugins)       | ❌                     | ✅ (Google ecosystem) |
| **Multi-Step**   | ❌ Single-step                      | ✅ Breaks down tasks            | ❌                     | ❌                     | ⚠️ (limited chaining) |
| **Self-Correct** | ❌                                  | ✅ Iterative refinement         | ❌ (manual prompts)    | ❌                     | ❌                     |
| **Open Source**  | ❌                                  | ✅ (e.g., AutoGPT, LangChain)   | ❌                     | ❌                     | ❌                     |
| **Strengths**    | High-quality text generation        | Real-world task automation     | Balanced performance  | Complex reasoning     | Multimodal (text/image) |

