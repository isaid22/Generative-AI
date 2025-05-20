## Transcript-Based QC Metrics

Below are some of metrics related to transcript of conversation:

| **Metric**                                 | **How to Compute or Estimate**                                                                                                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Script Adherence Score (%)**          | Use keyword or pattern matching (e.g., did the transcript include "Hello, thank you for calling"?) and check for required components. Build a checklist and count matched items. |
| **2. Correct Intent Classification (Y/N)** | Use a classifier (e.g., fine-tuned BERT or LLM) to label the intent and compare against human-annotated ground truth.                                                            |
| **3. Hallucination Count**                 | Run fact-checking or NLI (Natural Language Inference) tools; compare LLM output vs original transcript content. Flag generated statements that don’t match the source.           |
| **4. Question Resolution Rate (%)**        | Identify whether a customer’s question was followed by a direct, complete answer. Use a QA model or semantic entailment tools to judge answer completeness.                      |
| **5. Disposition Match Flag**              | Use text classification to assign a disposition from the transcript, then compare it to the expected CRM/system disposition (e.g., sale, no-sale, callback).                     |
| **6. PII/PCI Leakage Flags**               | Use regex + NER models (e.g., `spaCy`, AWS Comprehend) to detect SSN, DOB, credit card numbers, etc.                                                                             |
| **7. Customer Sentiment Classification**   | Use a sentiment analysis model (e.g., VADER, BERT-based classifiers) on customer utterances only.                                                                                |
| **8. Opt-Out Detection (Y/N)**             | Pattern match for phrases like “do not call”, “unsubscribe”, or use intent classification for opt-out detection.                                                                 |
| **9. Whisper Notes Accuracy (%)**          | Compare extracted metadata ("whisper notes") with transcript content using fuzzy string match or embedding similarity (e.g., cosine similarity of sentence embeddings).          |
| **10. Response Lag Estimate (ms)**         | Requires timestamped transcript. Calculate lag between customer’s last word and agent’s first word. Normalize per exchange and average.                                          |


### Implementation sugestions

#### Tools & Libraries:
Text Classification: scikit-learn, Hugging Face Transformers

NER/PII Detection: spaCy, Presidio, AWS Comprehend, regex

Sentiment Analysis: VADER, TextBlob, fine-tuned distilBERT

Hallucination Detection: No turnkey solution; use LLM + reference checking or entailment (e.g., DeBERTa-based NLI)

Keyword Matching: Python’s re, or flashtext for efficient keyword search

Timestamp Handling: If timestamps are included in JSON or CSV, use pandas to compute time gaps


