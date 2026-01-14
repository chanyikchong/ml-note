#!/usr/bin/env python3
"""
Q&A Tool for ML Study Notes

Ask questions about ML concepts and get answers from the documentation.
Optionally generates patch proposals for improving content.

Usage:
    python -m qa.ask "What is gradient descent?"
    python -m qa.ask "How does PCA work?" --language en
    python -m qa.ask "What is regularization?" --propose-patch
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .index import DocumentIndex, Document


def format_answer(query: str, results: List[tuple], language: str = "en") -> Dict[str, Any]:
    """
    Generate a structured answer from search results.

    Returns a dictionary with:
    - direct_answer: Main answer
    - reasoning: Step-by-step explanation
    - alternatives: Alternative perspectives
    - action_plan: Suggested next steps
    - sources: Referenced documents
    """
    if not results:
        return {
            "direct_answer": "No relevant documentation found for this query.",
            "reasoning": "The search did not return any matching sections.",
            "alternatives": [],
            "action_plan": "Consider rephrasing the question or checking the available topics.",
            "sources": []
        }

    # Get top result
    top_doc, top_score = results[0]

    # Extract key information
    sources = []
    relevant_content = []

    for doc, score in results[:3]:
        sources.append({
            "path": doc.path,
            "title": doc.title,
            "section": doc.section,
            "score": round(score, 2)
        })
        relevant_content.append(doc.content[:500] + "..." if len(doc.content) > 500 else doc.content)

    # Generate structured answer
    answer = {
        "query": query,
        "language": language,
        "timestamp": datetime.now().isoformat(),
        "direct_answer": f"Based on the documentation in '{top_doc.title}', section '{top_doc.section}':\n\n{relevant_content[0][:300]}...",
        "reasoning": f"""
1. Found {len(results)} relevant sections matching your query
2. Most relevant: {top_doc.title} > {top_doc.section} (score: {top_score:.2f})
3. The documentation covers this topic with mathematical details and examples
4. See the Quiz section in the source for self-assessment questions
""".strip(),
        "alternatives": [
            f"Also see: {r[0].title} > {r[0].section}" for r in results[1:3]
        ],
        "action_plan": f"""
1. Read the full section: {top_doc.path}
2. Study the mathematical derivations
3. Complete the quiz questions
4. Try the related code examples
5. Cross-reference with the Interview Questions page
""".strip(),
        "sources": sources
    }

    return answer


def generate_patch_proposal(query: str, answer: Dict[str, Any], confusion_detected: bool = False) -> Dict[str, Any]:
    """
    Generate a patch proposal for content improvement.

    If confusion_detected is True, also proposes quiz additions.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    patch = {
        "id": f"patch_{timestamp}",
        "query": query,
        "type": "content_clarification",
        "status": "proposed",
        "created_at": datetime.now().isoformat(),
        "affected_files": [s["path"] for s in answer.get("sources", [])],
        "description": f"Clarification needed based on user query: {query}",
        "proposed_changes": []
    }

    if answer.get("sources"):
        top_source = answer["sources"][0]
        patch["proposed_changes"].append({
            "file": top_source["path"],
            "section": top_source["section"],
            "change_type": "clarification",
            "rationale": f"User asked: {query}. Current content may need additional explanation.",
            "suggested_addition": f"""
<!--
Suggested clarification for: {query}

Consider adding:
1. More explicit definition
2. Additional example
3. Common misconception note
-->
"""
        })

    # Propose quiz additions if confusion detected
    if confusion_detected:
        patch["proposed_changes"].append({
            "file": answer["sources"][0]["path"] if answer.get("sources") else "unknown",
            "section": "Quiz",
            "change_type": "quiz_addition",
            "rationale": f"User confusion detected. Suggest adding quiz question about: {query}",
            "suggested_addition": f"""
<details>
<summary><strong>Q (New):</strong> {query}</summary>

**Answer**: [Add clear answer here]

**Explanation**: [Add detailed explanation]

**Key equation**: [Add relevant equation if applicable]

**Common pitfall**: [Add common misconception]
</details>
"""
        })

    return patch


def save_patch(patch: Dict[str, Any]):
    """Save patch to both .claude/patches (temp) and proposals/ (final)."""
    # Temp location
    temp_dir = Path(".claude/patches")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{patch['id']}.json"

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    # Final location
    final_dir = Path("proposals")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{patch['id']}.json"

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    return str(final_path)


def detect_confusion(query: str) -> bool:
    """
    Simple heuristic to detect if query indicates confusion.

    Returns True if the query suggests the user is confused about a concept.
    """
    confusion_indicators = [
        "don't understand",
        "confused",
        "what is the difference",
        "why does",
        "how come",
        "doesn't make sense",
        "不明白",
        "不理解",
        "为什么",
        "区别是什么",
        "搞不清"
    ]

    query_lower = query.lower()
    return any(indicator in query_lower for indicator in confusion_indicators)


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about ML concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m qa.ask "What is gradient descent?"
    python -m qa.ask "Explain PCA" --language en
    python -m qa.ask "I don't understand regularization" --propose-patch
    python -m qa.ask "SVM vs logistic regression" --top-k 5
        """
    )

    parser.add_argument("query", help="Your question about ML")
    parser.add_argument("--language", "-l", choices=["en", "zh"], default=None,
                        help="Limit search to specific language")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--propose-patch", "-p", action="store_true",
                        help="Generate a patch proposal")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output in JSON format")
    parser.add_argument("--docs-dir", default="docs",
                        help="Documentation directory")

    args = parser.parse_args()

    # Initialize index
    index = DocumentIndex(args.docs_dir)
    if not index.load_index():
        print("Building index (first run)...", file=sys.stderr)
        index.build_index()

    # Search
    results = index.search(args.query, top_k=args.top_k, language=args.language)

    # Generate answer
    answer = format_answer(args.query, results, args.language or "en")

    # Handle patch proposal
    patch_path = None
    if args.propose_patch:
        confusion = detect_confusion(args.query)
        patch = generate_patch_proposal(args.query, answer, confusion)
        patch_path = save_patch(patch)
        answer["patch_proposal"] = patch_path

    # Output
    if args.json:
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    else:
        print("\n" + "="*60)
        print("QUERY:", answer["query"])
        print("="*60)

        print("\n📚 DIRECT ANSWER:")
        print("-"*40)
        print(answer["direct_answer"])

        print("\n🔍 REASONING:")
        print("-"*40)
        print(answer["reasoning"])

        if answer["alternatives"]:
            print("\n📖 SEE ALSO:")
            print("-"*40)
            for alt in answer["alternatives"]:
                print(f"  • {alt}")

        print("\n📝 ACTION PLAN:")
        print("-"*40)
        print(answer["action_plan"])

        print("\n📎 SOURCES:")
        print("-"*40)
        for src in answer["sources"]:
            print(f"  • {src['path']} > {src['section']} (score: {src['score']})")

        if patch_path:
            print(f"\n✅ PATCH PROPOSAL: {patch_path}")

        print()


if __name__ == "__main__":
    main()
