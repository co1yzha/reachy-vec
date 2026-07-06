"""Conversation loop and intent routing.

Routes each utterance to one of: Q&A (RAG over docs+memories), note-taking
("note that ..."), or messaging ("tell <person> ..."), then prompts Claude
with retrieved context and the recognized speaker's identity.
"""
