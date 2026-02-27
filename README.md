# LLM-based Analysis of Mountain Value Chain Narratives (MOVING Project)

This repository contains experimental code for analyzing 454 narratives describing mountain value chains collected in the European research project MOVING (Mountain Valorisation through Interconnectedness and Green Growth).

The goal is to automatically identify narratives that satisfy specific use-case conditions (e.g., PDO cheese production, milk source, sheep-based production) using small open-source Large Language Models (LLMs) running locally via Ollama.

Two different approaches are implemented:

1. In-Context Learning

Each narrative is directly passed to the LLM together with a system prompt describing the use case.
The model analyzes the narrative and returns a JSON output indicating whether the narrative satisfies the required condition.

2. Retrieval-Augmented Filtering (Vector RAG)

Narratives are first embedded and stored in a Chroma vector database.
For each use case:

A semantic search retrieves the most relevant narratives.

The LLM then evaluates each retrieved narrative to verify whether it actually satisfies the conditions described in the prompt.

Both approaches are tested with multiple small open-source LLMs (e.g., Gemma, LLaMA, Mistral, Phi, DeepSeek) to compare their effectiveness in structured narrative analysis.

The outputs are stored as JSON files listing the IDs of narratives that match each use case.
