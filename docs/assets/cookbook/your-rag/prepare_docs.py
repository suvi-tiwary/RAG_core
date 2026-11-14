"""Find empty, tiny and duplicated documents before you write questions."""

from pathlib import Path

from rag_titan.evaluation import profile_documents, sample_documents

# 1. collect your documents as {name: text} (from disk, S3, a database ...)
documents = {path.name: path.read_text(encoding="utf-8")
             for path in sorted(Path("documents").glob("*.md"))}

# 2. profile them
profile = profile_documents(documents, min_words=20)
print(f"Documents: {profile['document_count']}")
for name, issues in profile["issues"].items():
    print(f"  problem in {name}: {', '.join(issues)}")

# 3. keep the clean ones, then pick the documents to write questions for
clean = {name: text for name, text in documents.items() if name not in profile["issues"]}
picked = sample_documents(clean, size=1, strategy="random", seed=7)
print(f"Clean documents: {len(clean)}")
print(f"Write golden questions for: {', '.join(picked)}")
