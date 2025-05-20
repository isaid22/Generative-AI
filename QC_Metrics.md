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

For Whisper Note Accureacy, following are some realistic means of calculation:

🔹 Keyword or Phrase Overlap
Method: Use TF-IDF or bag-of-words models to detect if the same key concepts appear in both.

Useful for: Call center contexts (e.g., “close account”, “credit card limit”).

Tool: sklearn.feature_extraction.text.TfidfVectorizer

🔹 Semantic Similarity (Embeddings)
Method: Convert both the whisper note and transcript excerpt into embeddings using:

Sentence-BERT

OpenAI embeddings

Universal Sentence Encoder

Compute cosine similarity between them.

Pros: Captures true paraphrasing.

Threshold: Typically 0.75–0.85.

Example Tools: sentence-transformers, openai, tensorflow-hub

## Audio Recording QC Metrics

| **Metric**                          | **Description**                                   | **How to Compute (Tools/Techniques)**                                                                                                      |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Average Response Lag (ms)**       | Time between customer utterance end & agent start | Use diarization + timestamps (e.g., via [pyannote-audio](https://github.com/pyannote/pyannote-audio)), compute gaps between speaker turns. |
| **Max Response Lag (ms)**           | Longest pause between speaker turns               | Same as above, but take the max lag between utterances.                                                                                    |
| **% Turns with Lag > Threshold**    | % of speaker turns with lag > e.g. 1500ms         | Count how many gaps exceed threshold / total turns.                                                                                        |
| **Silence Events Count**            | Count of silent periods > set length              | Use `pydub.silence.detect_silence` or `webrtcvad` to find long silent segments (e.g., > 2 sec).                                            |
| **Audio Quality Score (1–10)**      | Composite score for clarity, dropout, noise       | Can use pre-trained models (e.g., [DNSMOS](https://github.com/microsoft/DNS-Challenge)) or define heuristics.                              |
| **Overlapping Speech Events**       | # of times speakers talk over each other          | Use speaker diarization; check for overlapping timestamps.                                                                                 |
| **Voice Agent Tone Classification** | Polite, Robotic, Natural, etc.                    | Extract tone features (e.g., pitch, energy) + classify using a trained model or zero-shot LLM classification.                              |
| **Customer Emotion Detection**      | Happy / Neutral / Angry...                        | Use audio emotion detection (e.g., `SpeechBrain`, `pyAudioAnalysis`, `opensmile`).                                                         |
| **PCI Data Spoken (Y/N)**           | Did customer say sensitive PCI data?              | Transcribe audio (if not already) and search for regex patterns or NER (e.g., card numbers, CVV).                                          |
| **Call Engagement Duration (s)**    | Duration excluding silence/hold                   | Total call duration - silent/hold segments. Use diarization + silence detection.                                                           |
| **Background Noise Events**         | Count of external spikes (dogs, TVs, etc.)        | Use audio classification models (e.g., YAMNet or VGGish) or simple amplitude thresholding + classifiers.                                   |
| **Immediate Disconnect Flag**       | Customer picked up & hung up quickly              | From call metadata: check if call lasted < threshold (e.g., < 5 sec).                                                                      |
| **HLA Transfer Accuracy**           | Was transfer needed, triggered, and correct       | Requires event logs + audio alignment to detect handoff success; custom rule-based or intent models may help.                              |


### Key libraries

| **Purpose**              | **Library/Tool**                                 |
| ------------------------ | ------------------------------------------------ |
| Speaker diarization      | `pyannote.audio`, `resemblyzer`                  |
| Speech-to-text           | `whisper`, `AWS Transcribe`, `Google Speech API` |
| Silence detection        | `pydub`, `webrtcvad`, `librosa`                  |
| Emotion & tone detection | `SpeechBrain`, `opensmile`, `pyAudioAnalysis`    |
| Audio classification     | `YAMNet`, `VGGish`, `torchcrepe`                 |
| Sensitive data detection | Regex + `spaCy`, `Presidio`, LLM                 |
| Audio quality score      | `DNSMOS`, signal-to-noise ratio (SNR)            |

