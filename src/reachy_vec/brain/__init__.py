"""Reasoning layer: conversation loop, intent routing, and RAG prompting.

Modules:
- intents: classify each utterance as Q&A, note-taking, or messaging
- rag:     retrieve context from docs + memories, build the LLM prompt
- loop:    the conversation state machine tying it all together
"""
