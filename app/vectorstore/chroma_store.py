import os
import re
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, cast
from app.vectorstore.base import VectorStoreBase
from app.schemas.document import DocumentChunk
from app.schemas.metadata import ChunkMetadata, TopicHierarchy, ChapterNode, TopicNode
from app.core.config import settings
from app.core.logging import logger


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _safe_int(val: Any, default: int = 1) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _is_valid_catalog_entry(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 60:
        return False
    # Reject standalone numbers / measurements
    if re.match(r'^[0-9\s,.\-+=/\\()<>*#@$%^&]+$', name):
        return False
    if re.match(r'^\d+\s*(?:m|cm|mm|km|kg|lakh|crore|billion|million)?$', name, re.IGNORECASE):
        return False
    # Reject questions and question option markers
    if '?' in name or re.match(r'^(?:\(?\d+\)?|\(?[a-zA-Z]\)[\s.:\-])', name):
        return False
    # Reject generic section keywords
    if name.lower().strip(' !.?-_') in {
        'answer', 'answers', 'question', 'questions', 'learning objectives', 'note', 'notes',
        'summary', 'thank you', 'review', 'recall', 'activity', 'overview', 'tens', 'ones',
        'classroom', 'a broom', 'tidy', 'write your answer'
    }:
        return False
    return True


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB Vector Store Implementation with strict metadata filtering."""

    def __init__(self, collection_name: str = "topic_rag_chunks", persist_directory: Optional[str] = None):
        path = persist_directory or settings.CHROMA_PATH
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaVectorStore at '{path}', collection '{collection_name}'")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata.to_dict() for c in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=texts,
            metadatas=cast(Any, metadatas)
        )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")

    def search(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 5
    ) -> List[DocumentChunk]:
        where_clause = self._build_where_clause(metadata_filter)

        try:
            results = self.collection.query(
                query_embeddings=cast(Any, [query_embedding]),
                n_results=top_k,
                where=where_clause if where_clause else None
            )

            chunks: List[DocumentChunk] = []
            if (
                results
                and results.get("documents")
                and results.get("metadatas")
                and results.get("ids")
            ):
                docs = results["documents"]
                metas = results["metadatas"]
                ids = results["ids"]

                if docs and metas and ids and docs[0] is not None and metas[0] is not None and ids[0] is not None:
                    retrieved_docs = docs[0]
                    retrieved_metas = metas[0]
                    retrieved_ids = ids[0]

                    for doc_id, text, meta in zip(retrieved_ids, retrieved_docs, retrieved_metas):
                        if meta and isinstance(meta, dict):
                            meta_obj = ChunkMetadata(
                                chunk_id=_safe_str(meta.get("chunk_id"), doc_id),
                                subject=_safe_str(meta.get("subject")),
                                class_level=_safe_str(meta.get("class_level"), "General"),
                                chapter=_safe_str(meta.get("chapter")),
                                topic=_safe_str(meta.get("topic")),
                                subtopic=_safe_str(meta.get("subtopic"), "General"),
                                source_document=_safe_str(meta.get("source_document")),
                                page_number=_safe_int(meta.get("page_number"), 1),
                                language=_safe_str(meta.get("language"), "English"),
                            )
                            chunks.append(DocumentChunk(chunk_id=doc_id, text=text, metadata=meta_obj))

            logger.info(f"Retrieved {len(chunks)} chunks with filter {where_clause}")
            return chunks
        except Exception as e:
            logger.error(f"Error during ChromaDB vector search: {e}")
            return []

    def get_topics_hierarchy(self, source_document: Optional[str] = None) -> TopicHierarchy:
        where_clause = self._build_where_clause({"source_document": source_document}) if source_document else None
        try:
            data = self.collection.get(where=where_clause if where_clause else None, include=["metadatas"])
        except Exception as e:
            logger.error(f"Error fetching from Chroma collection in get_topics_hierarchy: {e}")
            data = None

        subject = "General Subject"
        hierarchy_dict: Dict[str, Dict[str, List[str]]] = {}

        if data and "metadatas" in data and data["metadatas"]:
            for meta in data["metadatas"]:
                if not meta or not isinstance(meta, dict):
                    continue
                subj = _safe_str(meta.get("subject"), "General Subject")
                chap = _safe_str(meta.get("chapter"), "General Chapter")
                top = _safe_str(meta.get("topic"), "General Topic")
                subtop = _safe_str(meta.get("subtopic"), "General")

                if not _is_valid_catalog_entry(chap) or not _is_valid_catalog_entry(top):
                    continue

                if subj != "General Subject":
                    subject = subj

                if chap not in hierarchy_dict:
                    hierarchy_dict[chap] = {}
                if top not in hierarchy_dict[chap]:
                    hierarchy_dict[chap][top] = []
                if subtop and subtop not in hierarchy_dict[chap][top]:
                    hierarchy_dict[chap][top].append(subtop)

        chapters_list: List[ChapterNode] = []
        for chap, topics in sorted(hierarchy_dict.items()):
            topic_nodes: List[TopicNode] = []
            for top, subtopics in sorted(topics.items()):
                topic_nodes.append(TopicNode(name=top, subtopics=subtopics))
            chapters_list.append(ChapterNode(name=chap, topics=topic_nodes))

        return TopicHierarchy(subject=subject, chapters=chapters_list)

    def get_all_hierarchies(self) -> List[TopicHierarchy]:
        """Returns structured hierarchy trees for all distinct subjects in vector store."""
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as e:
            logger.error(f"Error fetching all topics hierarchy: {e}")
            return []

        raw_metas = (data.get("metadatas") or []) if data else []
        if not raw_metas:
            return []

        subject_map: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

        for meta in raw_metas:
            if not meta or not isinstance(meta, dict):
                continue
            subj = _safe_str(meta.get("subject"), "General Subject")
            chap = _safe_str(meta.get("chapter"), "General Chapter")
            top = _safe_str(meta.get("topic"), "General Topic")
            subtop = _safe_str(meta.get("subtopic"), "General")

            if not _is_valid_catalog_entry(subj) or not _is_valid_catalog_entry(chap) or not _is_valid_catalog_entry(top):
                continue

            if subj not in subject_map:
                subject_map[subj] = {}
            if chap not in subject_map[subj]:
                subject_map[subj][chap] = {}
            if top not in subject_map[subj][chap]:
                subject_map[subj][chap][top] = []
            if subtop and subtop not in subject_map[subj][chap][top]:
                subject_map[subj][chap][top].append(subtop)

        result: List[TopicHierarchy] = []
        for subj, chapters in sorted(subject_map.items()):
            chap_nodes: List[ChapterNode] = []
            for chap, topics in sorted(chapters.items()):
                top_nodes: List[TopicNode] = []
                for top, subtopics in sorted(topics.items()):
                    top_nodes.append(TopicNode(name=top, subtopics=subtopics))
                chap_nodes.append(ChapterNode(name=chap, topics=top_nodes))
            result.append(TopicHierarchy(subject=subj, chapters=chap_nodes))

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection statistics and distinct metadata count."""
        try:
            count = self.collection.count()
            data = self.collection.get(include=["metadatas"])
            raw_metas = (data.get("metadatas") or []) if data else []
            subjects = set()
            chapters = set()
            topics = set()
            documents = set()
            for m in raw_metas:
                if m and isinstance(m, dict):
                    if m.get("subject"):
                        subjects.add(_safe_str(m.get("subject")))
                    if m.get("chapter"):
                        chapters.add(_safe_str(m.get("chapter")))
                    if m.get("topic"):
                        topics.add(_safe_str(m.get("topic")))
                    if m.get("source_document"):
                        documents.add(_safe_str(m.get("source_document")))
            return {
                "total_chunks": count,
                "total_subjects": len(subjects),
                "total_chapters": len(chapters),
                "total_topics": len(topics),
                "total_documents": len(documents),
                "subjects": sorted(list(subjects)),
                "status": "ready"
            }
        except Exception as e:
            return {"total_chunks": 0, "status": "error", "error": str(e)}

    def delete_document(self, source_document: str) -> None:
        self.collection.delete(where={"source_document": source_document})
        logger.info(f"Deleted chunks for source document '{source_document}'")

    def clear(self) -> None:
        # Delete and recreate collection
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store collection.")

    def _build_where_clause(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs ChromaDB compatible `$and` query filter."""
        conditions = []
        for k, v in metadata_filter.items():
            if v is not None and str(v).strip():
                conditions.append({k: {"$eq": str(v).strip()}})

        if not conditions:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
