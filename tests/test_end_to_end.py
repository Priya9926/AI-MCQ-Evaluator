import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.vectorstore.chroma_store import ChromaVectorStore

client = TestClient(app)


def test_end_to_end_topic_isolated_question_generation(tmp_path):
    # Prepare sample educational text document containing multiple chapters and topics
    sample_doc_content = (
        "Subject: Physics\n\n"
        "Chapter: Mechanics\n\n"
        "Topic: Velocity\n"
        "Velocity is defined as the rate of change of displacement with respect to time. "
        "The formula for average velocity is v = delta_s / delta_t. It is a vector quantity having both magnitude and direction.\n\n"
        "Topic: Acceleration\n"
        "Acceleration is defined as the rate of change of velocity with respect to time. "
        "The formula for acceleration is a = delta_v / delta_t. "
        "Uniform acceleration occurs when an object changes velocity at a constant rate over equal time intervals. "
        "Non-uniform acceleration occurs when velocity changes at varying rates.\n\n"
        "Chapter: Thermodynamics\n\n"
        "Topic: Heat\n"
        "Heat is energy in transit transferred between systems due to a temperature difference. "
        "The SI unit of heat is the Joule. Heat transfer can occur via conduction, convection, and radiation.\n"
    )

    doc_file = tmp_path / "physics_sample.txt"
    doc_file.write_text(sample_doc_content, encoding="utf-8")

    # Step 1: Upload document via API
    with open(doc_file, "rb") as f:
        upload_res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("physics_sample.txt", f, "text/plain")}
        )

    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document_id"]
    assert upload_data["total_chunks"] > 0
    assert upload_data["detected_hierarchy"]["subject"] == "Physics"

    # Step 2: Retrieve document topic hierarchy
    topics_res = client.get(f"/api/v1/documents/{doc_id}/topics")
    assert topics_res.status_code == 200
    hierarchy = topics_res.json()
    chapter_names = [c["name"] for c in hierarchy["chapters"]]
    assert "Mechanics" in chapter_names or any("Mechanics" in c for c in chapter_names)

    # Step 3: Request Question Generation strictly for Mechanics -> Acceleration
    gen_payload = {
        "document_id": doc_id,
        "subject": "Physics",
        "chapter": "Mechanics",
        "topic": "Acceleration",
        "question_type": "mcq",
        "difficulty": "medium",
        "count": 5
    }

    gen_res = client.post("/api/v1/questions/generate", json=gen_payload)
    assert gen_res.status_code == 200, f"Error: {gen_res.text}"
    gen_data = gen_res.json()

    # Step 4: Validate strict topic isolation and groundedness assertions
    assert gen_data["total_generated"] > 0
    assert gen_data["topic"] == "Acceleration"

    questions = gen_data["questions"]
    for q in questions:
        # 1. Topic must be Acceleration
        assert q["topic"] == "Acceleration"
        assert q["subject"] == "Physics"

        # 2. Options structure
        assert "A" in q["options"]
        assert "B" in q["options"]
        assert "C" in q["options"]
        assert "D" in q["options"]
        assert q["correct_answer"] in ["A", "B", "C", "D"]

        # 3. Traceability to source chunk IDs
        assert len(q["source_chunk_ids"]) > 0
        for cid in q["source_chunk_ids"]:
            assert "acceleration" in cid.lower()
            assert "heat" not in cid.lower()
            assert "thermodynamics" not in cid.lower()

        # 4. Content must be about Acceleration, NOT Heat or Thermodynamics
        q_str = f"{q['question']} {q['explanation']}".lower()
        assert "thermodynamics" not in q_str
        assert "conduction" not in q_str


def test_root_dashboard():
    res = client.get("/")
    assert res.status_code == 200
    assert "Topic-Aware RAG Engine" in res.text


def test_topics_all_and_stats():
    res = client.get("/api/v1/documents/topics/all")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    stats_res = client.get("/api/v1/documents/stats")
    assert stats_res.status_code == 200
    assert "total_chunks" in stats_res.json()

