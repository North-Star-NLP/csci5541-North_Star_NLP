import os
import json
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
CHUNK_SIZE = 512

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_knowledge_base(
    chunk_size: int,
    knowledge_base: list[Document],
    tokenizer_name: str = EMBEDDING_MODEL_NAME,
) -> list[Document]:
    """
    Split knowledge base documents into chunks of maximum size `chunk_size` tokens and return a list of unique documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def load_knowledge_base(
    docs_processed=None,
    faiss_index_path: str = "faiss_index",
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
):
    """
    Load an existing FAISS index from disk, or build one from processed documents and save it.

    Args:
        docs_processed: List of LangChain Documents (required if no saved index exists).
        faiss_index_path: Path to save/load the FAISS index.
        embedding_model_name: HuggingFace embedding model name.

    Returns:
        (KNOWLEDGE_VECTOR_DATABASE, embedding_model) tuple.
    """
    device = _get_device()
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(faiss_index_path):
        print(f"Loading existing FAISS index from '{faiss_index_path}'...")
        knowledge_base = FAISS.load_local(
            faiss_index_path, embedding_model, allow_dangerous_deserialization=True
        )
    else:
        if docs_processed is None:
            raise ValueError("No saved FAISS index found and no docs_processed provided to build one.")
        print("Building FAISS index from scratch...")
        knowledge_base = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_base.save_local(faiss_index_path)
        print(f"FAISS index saved to '{faiss_index_path}'")

    return knowledge_base, embedding_model



def _load_json_files(path: str) -> list[dict]:
    """Load JSON data from a single file or all .json files in a directory."""
    if os.path.isfile(path):
        if not path.endswith(".json"):
            raise ValueError(f"Expected a .json file, got: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    if os.path.isdir(path):
        all_data = []
        json_files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in directory: {path}")
        for fp in json_files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
        return all_data

    raise FileNotFoundError(f"Path does not exist: {path}")


def _records_to_documents(records: list[dict], text_key: str = "text", source_key: str = "source") -> list[Document]:
    """Convert JSON records to LangChain Documents, preserving id and is_poison metadata."""
    docs = []
    for record in records:
        content = record.get(text_key, "")
        if not content:
            content = json.dumps(record, ensure_ascii=False)
        
        # Build metadata, preserving all fields except text_key
        metadata = {k: v for k, v in record.items() if k != text_key}
        
        # Ensure source_key is present
        if source_key not in metadata:
            metadata[source_key] = "json_import"
        
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _split_json_documents(docs: list[Document], chunk_size: int = CHUNK_SIZE, tokenizer_name: str = EMBEDDING_MODEL_NAME) -> list[Document]:
    """Split documents into chunks using a tokenizer-aware splitter."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(docs)


def add_json_to_knowledge_base(
    path: str,
    knowledge_base: FAISS = None,
    text_key: str = "text",
    source_key: str = "source",
    chunk_size: int = CHUNK_SIZE,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> FAISS:
    """
    Load JSON files from a file or directory and add them to a FAISS knowledge base.

    Args:
        path: Path to a single .json file or a directory containing .json files.
        knowledge_base: Existing FAISS vector store to add to. If None, creates a new one.
        text_key: Key in each JSON record to use as document content.
        source_key: Key to use as the source metadata field.
        chunk_size: Token chunk size for splitting documents.
        embedding_model_name: HuggingFace embedding model name.

    Returns:
        The updated (or newly created) FAISS knowledge base.
    """
    records = _load_json_files(path)
    docs = _records_to_documents(records, text_key=text_key, source_key=source_key)
    docs_processed = _split_json_documents(docs, chunk_size=chunk_size, tokenizer_name=embedding_model_name)

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
    )

    if knowledge_base is None:
        knowledge_base = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
    else:
        knowledge_base.add_documents(docs_processed)

    print(f"Added {len(docs_processed)} chunks from {len(records)} records to the knowledge base.")
    return knowledge_base
