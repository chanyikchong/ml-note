#!/usr/bin/env python3
"""
Skills Extraction System
========================

Extracts and tracks ML skills from documentation.
Generates a skills report for interview preparation.
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class SkillsExtractor:
    """Extract skills from ML documentation."""

    def __init__(self, template_path: str = "skills/template.yaml"):
        """Load skills template."""
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = yaml.safe_load(f)

        self.docs_dir = Path("docs")
        self.skills_found: Dict[str, Dict] = {}
        self.doc_coverage: Dict[str, Set[str]] = defaultdict(set)

    def extract_skills_from_file(self, file_path: Path) -> None:
        """Extract skills mentioned in a documentation file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        rel_path = str(file_path.relative_to(self.docs_dir))

        for category in self.template["categories"]:
            for skill in category["skills"]:
                skill_id = skill["id"]
                keywords = skill["keywords"]

                # Check if any keyword is mentioned
                for keyword in keywords:
                    if keyword.lower() in content:
                        self.skills_found[skill_id] = {
                            **skill,
                            "category": category["name"],
                        }
                        self.doc_coverage[skill_id].add(rel_path)
                        break

    def extract_all(self) -> None:
        """Extract skills from all documentation files."""
        print("Extracting skills from documentation...")

        for lang in ["en", "zh"]:
            lang_dir = self.docs_dir / lang
            if lang_dir.exists():
                for md_file in lang_dir.rglob("*.md"):
                    self.extract_skills_from_file(md_file)

    def generate_report(self, output_path: str = "skills/extracted_skills.yaml") -> None:
        """Generate skills extraction report."""
        report = {
            "summary": {
                "total_skills_defined": sum(
                    len(cat["skills"]) for cat in self.template["categories"]
                ),
                "skills_found": len(self.skills_found),
                "coverage_percentage": round(
                    len(self.skills_found)
                    / sum(len(cat["skills"]) for cat in self.template["categories"])
                    * 100,
                    1,
                ),
            },
            "skills_by_category": {},
            "skills_by_difficulty": defaultdict(list),
            "skills_by_relevance": defaultdict(list),
            "missing_skills": [],
        }

        # Organize by category
        for category in self.template["categories"]:
            cat_name = category["name"]
            report["skills_by_category"][cat_name] = {
                "found": [],
                "missing": [],
            }

            for skill in category["skills"]:
                skill_id = skill["id"]
                if skill_id in self.skills_found:
                    skill_info = {
                        "id": skill_id,
                        "name": skill["name"],
                        "difficulty": skill["difficulty"],
                        "interview_relevance": skill["interview_relevance"],
                        "documented_in": list(self.doc_coverage[skill_id]),
                    }
                    report["skills_by_category"][cat_name]["found"].append(skill_info)
                    report["skills_by_difficulty"][skill["difficulty"]].append(
                        skill["name"]
                    )
                    report["skills_by_relevance"][skill["interview_relevance"]].append(
                        skill["name"]
                    )
                else:
                    report["skills_by_category"][cat_name]["missing"].append(
                        skill["name"]
                    )
                    report["missing_skills"].append(
                        {"name": skill["name"], "category": cat_name}
                    )

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)

        print(f"\nSkills report written to: {output_path}")

    def print_summary(self) -> None:
        """Print skills extraction summary."""
        total = sum(len(cat["skills"]) for cat in self.template["categories"])
        found = len(self.skills_found)

        print("\n" + "=" * 50)
        print("SKILLS EXTRACTION SUMMARY")
        print("=" * 50)
        print(f"\nTotal skills defined: {total}")
        print(f"Skills documented: {found}")
        print(f"Coverage: {found / total * 100:.1f}%")

        print("\nBy Category:")
        for category in self.template["categories"]:
            cat_name = category["name"]
            cat_total = len(category["skills"])
            cat_found = sum(
                1 for skill in category["skills"] if skill["id"] in self.skills_found
            )
            print(f"  {cat_name}: {cat_found}/{cat_total}")

        print("\nBy Difficulty:")
        for difficulty in ["beginner", "intermediate", "advanced"]:
            skills = [
                s
                for s in self.skills_found.values()
                if s["difficulty"] == difficulty
            ]
            print(f"  {difficulty.capitalize()}: {len(skills)}")

        print("\nBy Interview Relevance:")
        for relevance in ["high", "medium", "low"]:
            skills = [
                s
                for s in self.skills_found.values()
                if s["interview_relevance"] == relevance
            ]
            print(f"  {relevance.capitalize()}: {len(skills)}")

        if total > found:
            missing = total - found
            print(f"\n⚠ {missing} skill(s) not yet documented")


def main():
    """Main entry point for skills extraction."""
    parser = argparse.ArgumentParser(description="Extract ML skills from documentation")
    parser.add_argument(
        "--template",
        default="skills/template.yaml",
        help="Path to skills template",
    )
    parser.add_argument(
        "--output",
        default="skills/extracted_skills.yaml",
        help="Path to output report",
    )
    args = parser.parse_args()

    extractor = SkillsExtractor(args.template)
    extractor.extract_all()
    extractor.generate_report(args.output)
    extractor.print_summary()


if __name__ == "__main__":
    main()
