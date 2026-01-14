#!/usr/bin/env python3
"""
Static Site Generator for ML Study Notes
========================================

Converts markdown documentation to a static HTML site with:
- Bilingual support (EN/ZH)
- KaTeX math rendering
- Collapsible quiz sections
- Navigation sidebar
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import markdown
import yaml
from jinja2 import Environment, FileSystemLoader


class SiteGenerator:
    """Generate static site from markdown documentation."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.docs_dir = Path(self.config["docs"]["source_dir"])
        self.output_dir = Path(self.config["docs"]["output_dir"])
        self.languages = self.config["project"]["languages"]

        # Markdown processor
        self.md = markdown.Markdown(
            extensions=[
                "tables",
                "fenced_code",
                "codehilite",
                "toc",
                "attr_list",
            ]
        )

    def get_template(self) -> str:
        """Return the HTML template."""
        return '''<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - ML Study Notes</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <style>
        :root {
            --primary-color: #2563eb;
            --bg-color: #ffffff;
            --text-color: #1f2937;
            --border-color: #e5e7eb;
            --code-bg: #f3f4f6;
            --sidebar-width: 280px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-color);
        }

        .container {
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            background: #f8fafc;
            border-right: 1px solid var(--border-color);
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }

        .sidebar h2 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }

        .sidebar nav ul {
            list-style: none;
        }

        .sidebar nav li {
            margin: 0.5rem 0;
        }

        .sidebar nav a {
            color: var(--text-color);
            text-decoration: none;
            display: block;
            padding: 0.3rem 0;
        }

        .sidebar nav a:hover {
            color: var(--primary-color);
        }

        .sidebar nav a.active {
            color: var(--primary-color);
            font-weight: 600;
        }

        /* Language Toggle */
        .lang-toggle {
            display: flex;
            gap: 10px;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .lang-toggle a {
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            color: var(--text-color);
            background: white;
            border: 1px solid var(--border-color);
        }

        .lang-toggle a.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        /* Main Content */
        .main-content {
            margin-left: var(--sidebar-width);
            padding: 40px;
            max-width: 900px;
            flex: 1;
        }

        .main-content h1 {
            font-size: 2rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }

        .main-content h2 {
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            color: var(--primary-color);
        }

        .main-content h3 {
            font-size: 1.2rem;
            margin: 1.5rem 0 0.5rem;
        }

        .main-content p {
            margin: 1rem 0;
        }

        .main-content ul, .main-content ol {
            margin: 1rem 0;
            padding-left: 2rem;
        }

        .main-content li {
            margin: 0.5rem 0;
        }

        /* Code */
        .main-content code {
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9rem;
        }

        .main-content pre {
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .main-content pre code {
            background: none;
            padding: 0;
        }

        /* Tables */
        .main-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        .main-content th, .main-content td {
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            text-align: left;
        }

        .main-content th {
            background: var(--code-bg);
        }

        /* Quiz (details/summary) */
        .main-content details {
            margin: 0.5rem 0;
            padding: 1rem;
            background: #f0f9ff;
            border-radius: 5px;
            border-left: 4px solid var(--primary-color);
        }

        .main-content summary {
            cursor: pointer;
            font-weight: 600;
        }

        .main-content details[open] summary {
            margin-bottom: 0.5rem;
        }

        /* Math */
        .katex-display {
            overflow-x: auto;
            overflow-y: hidden;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            .main-content {
                margin-left: 0;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <div class="lang-toggle">
                {% for language in languages %}
                <a href="{{ language.code }}/index.html"
                   class="{{ 'active' if language.code == lang else '' }}">
                    {{ language.flag }} {{ language.name }}
                </a>
                {% endfor %}
            </div>
            <h2>ML Study Notes</h2>
            <nav>
                {{ navigation | safe }}
            </nav>
        </aside>
        <main class="main-content">
            {{ content | safe }}
        </main>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\\\[", right: "\\\\]", display: true},
                    {left: "\\\\(", right: "\\\\)", display: false}
                ]
            });
        });
    </script>
</body>
</html>'''

    def build_navigation(self, lang: str, current_file: str = "") -> str:
        """Build navigation HTML for sidebar."""
        # Define navigation structure with folder mappings
        nav_sections = [
            ("fundamentals", "Fundamentals" if lang == "en" else "基础知识", [
                ("learning-paradigms", "Learning Paradigms" if lang == "en" else "学习范式"),
                ("data-splits", "Data Splits" if lang == "en" else "数据划分"),
                ("bias-variance", "Bias-Variance" if lang == "en" else "偏差-方差"),
                ("loss-functions", "Loss Functions" if lang == "en" else "损失函数"),
                ("optimization", "Optimization" if lang == "en" else "优化算法"),
                ("regularization", "Regularization" if lang == "en" else "正则化"),
                ("generalization", "Generalization" if lang == "en" else "泛化"),
            ]),
            ("models", "Models" if lang == "en" else "核心模型", [
                ("linear-regression", "Linear Regression" if lang == "en" else "线性回归"),
                ("logistic-regression", "Logistic Regression" if lang == "en" else "逻辑回归"),
                ("svm", "SVM" if lang == "en" else "支持向量机"),
                ("knn", "k-NN" if lang == "en" else "k近邻"),
                ("naive-bayes", "Naive Bayes" if lang == "en" else "朴素贝叶斯"),
                ("decision-trees", "Decision Trees" if lang == "en" else "决策树"),
                ("ensemble", "Ensemble Methods" if lang == "en" else "集成方法"),
                ("clustering", "Clustering" if lang == "en" else "聚类"),
                ("dimensionality-reduction", "Dim. Reduction" if lang == "en" else "降维"),
            ]),
            ("deep-learning", "Deep Learning" if lang == "en" else "深度学习", [
                ("nn-fundamentals", "NN Fundamentals" if lang == "en" else "神经网络基础"),
                ("cnn", "CNN" if lang == "en" else "卷积神经网络"),
                ("rnn", "RNN" if lang == "en" else "循环神经网络"),
                ("normalization", "Normalization" if lang == "en" else "归一化"),
                ("training", "Training" if lang == "en" else "训练技巧"),
            ]),
            ("practical", "Practical ML" if lang == "en" else "实用机器学习", [
                ("metrics", "Metrics" if lang == "en" else "评估指标"),
                ("class-imbalance", "Class Imbalance" if lang == "en" else "类别不平衡"),
                ("feature-engineering", "Feature Engineering" if lang == "en" else "特征工程"),
                ("interpretability", "Interpretability" if lang == "en" else "可解释性"),
                ("data-centric", "Data-Centric" if lang == "en" else "数据中心"),
                ("mlops", "MLOps" if lang == "en" else "MLOps"),
            ]),
            ("interview", "Interview Prep" if lang == "en" else "面试准备", [
                ("common-questions", "Common Questions" if lang == "en" else "常见问题"),
                ("quick-reference", "Quick Reference" if lang == "en" else "快速参考"),
            ]),
        ]

        html = "<ul>"
        for folder, section_title, items in nav_sections:
            html += f"<li><strong>{section_title}</strong><ul>"
            for file_name, title in items:
                href = f"../{folder}/{file_name}.html"
                active = "active" if file_name in current_file else ""
                html += f'<li><a href="{href}" class="{active}">{title}</a></li>'
            html += "</ul></li>"
        html += "</ul>"
        return html

    def convert_markdown(self, content: str) -> str:
        """Convert markdown to HTML."""
        self.md.reset()
        return self.md.convert(content)

    def process_file(self, md_path: Path, lang: str) -> None:
        """Process a single markdown file."""
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convert markdown to HTML
        html_content = self.convert_markdown(content)

        # Extract title from first h1
        title_match = re.search(r"<h1[^>]*>(.+?)</h1>", html_content)
        title = title_match.group(1) if title_match else md_path.stem

        # Build navigation
        navigation = self.build_navigation(lang, md_path.stem)

        # Render template
        template = self.get_template()
        from jinja2 import Template
        t = Template(template)
        html = t.render(
            title=title,
            content=html_content,
            navigation=navigation,
            lang=lang,
            languages=self.languages,
        )

        # Write output
        rel_path = md_path.relative_to(self.docs_dir / lang)
        out_path = self.output_dir / lang / rel_path.with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"  Generated: {out_path}")

    def build(self) -> None:
        """Build the entire site."""
        print("Building ML Study Notes Site...")
        print("=" * 50)

        # Clean output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        # Process each language
        for language in self.languages:
            lang = language["code"]
            lang_dir = self.docs_dir / lang

            if not lang_dir.exists():
                print(f"Warning: {lang_dir} does not exist, skipping")
                continue

            print(f"\nProcessing {language['name']} ({lang})...")

            # Process all markdown files
            for md_file in lang_dir.rglob("*.md"):
                self.process_file(md_file, lang)

        print("\n" + "=" * 50)
        print(f"Site generated in {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build ML Study Notes static site")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start local server after build"
    )
    args = parser.parse_args()

    generator = SiteGenerator(args.config)
    generator.build()

    if args.serve:
        import http.server
        import socketserver

        os.chdir(generator.output_dir)
        PORT = 8000
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\nServing at http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
