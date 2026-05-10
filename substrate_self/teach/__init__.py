"""Teach — corpus generation pipeline that uses LLM teachers to produce
training data for the substrate's own from-scratch language model.

The architectural commitment: the LLM is a TEACHER during corpus generation,
not a runtime dependency. The model that actually speaks at inference time
lives in `substrate_self.model` and has no LLM dependency.

Pipeline:
  1. Teacher (LLM) generates substrate-conditioned dialogue
  2. Corpus is saved as JSONL (one example per line)
  3. Tokenizer is fit to the corpus
  4. Model is trained on the corpus (substrate_self.model.train)
  5. Trained model becomes the runtime voice (substrate_self.model.generate)

After training, the LLM teacher can be removed entirely — the substrate
speaks with its own learned language faculty.
"""

from substrate_self.teach.corpus import generate_corpus, CorpusExample

__all__ = ["generate_corpus", "CorpusExample"]
