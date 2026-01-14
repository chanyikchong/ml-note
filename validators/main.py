#!/usr/bin/env python3
"""
Documentation Validators
========================

Validates:
1. Required sections in each topic page
2. Minimum quiz questions (>=5)
3. Bilingual consistency (EN/ZH structure matches)
4. Code examples exist
"""

import os
import re
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class ValidationError:
    """Represents a validation error."""

    def __init__(self, file: str, message: str, severity: str = "error"):
        self.file = file
        self.message = message
        self.severity = severity  # 'error' or 'warning'

    def __str__(self):
        return f"[{self.severity.upper()}] {self.file}: {self.message}"


class DocValidator:
    """Validates ML Study Notes documentation."""

    REQUIRED_SECTIONS = [
        "Interview Summary",
        "Core Definitions",
        "Math and Derivations",
        "Algorithm Sketch",
        "Common Pitfalls",
        "Mini Example",
        "Quiz",
        "References",
    ]

    REQUIRED_SECTIONS_ZH = [
        "面试摘要",
        "核心定义",
        "数学与推导",
        "算法框架",
        "常见陷阱",
        "迷你示例",
        "测验",
        "参考文献",
    ]

    MIN_QUIZ_QUESTIONS = 5

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.docs_dir = Path(self.config["docs"]["source_dir"])
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate_sections(self, content: str, file_path: str, lang: str) -> None:
        """Check that all required sections are present."""
        sections = self.REQUIRED_SECTIONS if lang == "en" else self.REQUIRED_SECTIONS_ZH

        for section in sections:
            # Look for section as h2 header
            pattern = rf"^##\s+\d*\.?\s*{re.escape(section)}"
            if not re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                self.errors.append(
                    ValidationError(file_path, f"Missing required section: '{section}'")
                )

    def validate_quiz_count(self, content: str, file_path: str) -> None:
        """Check that there are at least MIN_QUIZ_QUESTIONS quiz questions."""
        # Count <details> blocks (used for quiz questions)
        quiz_count = len(re.findall(r"<details>", content))

        if quiz_count < self.MIN_QUIZ_QUESTIONS:
            self.errors.append(
                ValidationError(
                    file_path,
                    f"Insufficient quiz questions: {quiz_count} < {self.MIN_QUIZ_QUESTIONS}",
                )
            )

    def validate_math_rendering(self, content: str, file_path: str) -> None:
        """Check for potential math rendering issues."""
        # Check for unescaped underscores in math mode
        if re.search(r"\$[^$]*(?<!\\)_(?!\{)[^$]*\$", content):
            self.warnings.append(
                ValidationError(
                    file_path,
                    "Potential unescaped underscore in math mode",
                    severity="warning",
                )
            )

    def validate_code_blocks(self, content: str, file_path: str) -> None:
        """Check for code blocks in example sections."""
        # Check if Mini Example section has code
        example_match = re.search(
            r"##\s+\d*\.?\s*(Mini Example|迷你示例)(.+?)(?=^##|\Z)",
            content,
            re.MULTILINE | re.DOTALL,
        )

        if example_match:
            example_content = example_match.group(2)
            if "```" not in example_content:
                self.warnings.append(
                    ValidationError(
                        file_path,
                        "Mini Example section has no code blocks",
                        severity="warning",
                    )
                )

    def validate_file(self, file_path: Path, lang: str) -> None:
        """Validate a single documentation file."""
        # Skip index files
        if file_path.name == "index.md":
            return

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = str(file_path.relative_to(self.docs_dir))

        self.validate_sections(content, rel_path, lang)
        self.validate_quiz_count(content, rel_path)
        self.validate_math_rendering(content, rel_path)
        self.validate_code_blocks(content, rel_path)

    def validate_bilingual_consistency(self) -> None:
        """Check that EN and ZH docs have matching structure."""
        en_dir = self.docs_dir / "en"
        zh_dir = self.docs_dir / "zh"

        if not en_dir.exists() or not zh_dir.exists():
            return

        en_files = set(
            str(f.relative_to(en_dir)) for f in en_dir.rglob("*.md")
        )
        zh_files = set(
            str(f.relative_to(zh_dir)) for f in zh_dir.rglob("*.md")
        )

        # Check for files in EN but not ZH
        for f in en_files - zh_files:
            self.errors.append(
                ValidationError(f"en/{f}", "Missing corresponding Chinese translation")
            )

        # Check for files in ZH but not EN
        for f in zh_files - en_files:
            self.errors.append(
                ValidationError(f"zh/{f}", "Missing corresponding English version")
            )

    def validate_all(self) -> Tuple[int, int]:
        """Validate all documentation files."""
        print("Validating ML Study Notes Documentation")
        print("=" * 50)

        # Validate English docs
        en_dir = self.docs_dir / "en"
        if en_dir.exists():
            print("\nValidating English (en) docs...")
            for md_file in en_dir.rglob("*.md"):
                self.validate_file(md_file, "en")

        # Validate Chinese docs
        zh_dir = self.docs_dir / "zh"
        if zh_dir.exists():
            print("Validating Chinese (zh) docs...")
            for md_file in zh_dir.rglob("*.md"):
                self.validate_file(md_file, "zh")

        # Check bilingual consistency
        print("Checking bilingual consistency...")
        self.validate_bilingual_consistency()

        return len(self.errors), len(self.warnings)

    def report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)

        if self.errors:
            print(f"\n{len(self.errors)} ERROR(S):")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n{len(self.warnings)} WARNING(S):")
            for warning in self.warnings:
                print(f"  {warning}")

        if not self.errors and not self.warnings:
            print("\n✓ All validations passed!")

        print("\n" + "=" * 50)
        print(f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings")


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(description="Validate ML Study Notes documentation")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    args = parser.parse_args()

    validator = DocValidator(args.config)
    n_errors, n_warnings = validator.validate_all()
    validator.report()

    # Exit with error code if validation failed
    if n_errors > 0 or (args.strict and n_warnings > 0):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
