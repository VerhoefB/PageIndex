import string

import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 lexical retriever.

    Preprocessing:
    - lowercase text
    - remove punctuation
    - remove stopwords
    - whitespace tokenization
    - no stemming
    - no lemmatization
    """

    def __init__(self, chunks, k1=1.2, b=0.75):
        nltk.download("stopwords", quiet=True)

        # Use only canonical retrieval chunks if duplicate flags exist.
        self.chunks = [
            chunk for chunk in chunks
            if not chunk.get("is_duplicate_chunk", False)
        ]

        self.stop_words = set(stopwords.words("english"))

        self.tokenized_chunks = [
            self._tokenize(self._get_text(chunk))
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks,
            k1=k1,
            b=b
        )

    def _get_text(self, chunk):
        return chunk.get("text") or chunk.get("page_content") or ""

    def _get_chunk_id(self, chunk):
        return (
            chunk.get("retrieval_chunk_id")
            or chunk.get("canonical_chunk_id")
            or chunk.get("chunk_id")
        )

    def _tokenize(self, text):
        text = str(text or "").lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words]

        return tokens

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks for a query across the full corpus.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]

            results.append({
                "rank": rank,
                "chunk_id": str(self._get_chunk_id(chunk)),
                "source_chunk_id": str(chunk.get("chunk_id")),
                "bank_name": chunk.get("bank_name") or chunk.get("doc_name"),
                "title": chunk.get("title", ""),
                "heading": chunk.get("heading", ""),
                "score": float(scores[idx]),
                "text": self._get_text(chunk),
            })

        return results