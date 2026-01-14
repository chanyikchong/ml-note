"""
Patch Management for ML Study Notes

Apply, review, and manage content patches.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


def list_patches(directory: str = "proposals") -> List[Dict]:
    """List all pending patches."""
    patches_dir = Path(directory)
    if not patches_dir.exists():
        return []

    patches = []
    for patch_file in patches_dir.glob("*.json"):
        try:
            with open(patch_file, 'r', encoding='utf-8') as f:
                patch = json.load(f)
                patch['_file'] = str(patch_file)
                patches.append(patch)
        except Exception as e:
            print(f"Error reading {patch_file}: {e}")

    return sorted(patches, key=lambda x: x.get('created_at', ''), reverse=True)


def show_patch(patch_id: str, directory: str = "proposals") -> Optional[Dict]:
    """Show details of a specific patch."""
    patches_dir = Path(directory)
    patch_file = patches_dir / f"{patch_id}.json"

    if not patch_file.exists():
        # Try finding by partial ID
        for f in patches_dir.glob(f"*{patch_id}*.json"):
            patch_file = f
            break

    if not patch_file.exists():
        return None

    with open(patch_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_patch(patch_id: str, directory: str = "proposals", dry_run: bool = True) -> bool:
    """
    Apply a patch to the documentation.

    Currently only supports showing what would be changed (dry run).
    Actual application requires manual editing.
    """
    patch = show_patch(patch_id, directory)
    if not patch:
        print(f"Patch not found: {patch_id}")
        return False

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Applying patch: {patch['id']}")
    print(f"Query: {patch['query']}")
    print(f"Type: {patch['type']}")
    print(f"Affected files: {', '.join(patch['affected_files'])}")

    for change in patch.get('proposed_changes', []):
        print(f"\n--- Change to: {change['file']} ---")
        print(f"Section: {change['section']}")
        print(f"Type: {change['change_type']}")
        print(f"Rationale: {change['rationale']}")
        print(f"\nSuggested addition:")
        print(change['suggested_addition'])

    if dry_run:
        print("\n[DRY RUN] No changes made. Remove --dry-run to apply.")
    else:
        # Mark as applied
        patch['status'] = 'applied'
        patch['applied_at'] = datetime.now().isoformat()

        patch_file = Path(directory) / f"{patch['id']}.json"
        with open(patch_file, 'w', encoding='utf-8') as f:
            json.dump(patch, f, indent=2, ensure_ascii=False)

        print("\nPatch marked as applied. Manual edits required.")

    return True


def reject_patch(patch_id: str, reason: str = "", directory: str = "proposals") -> bool:
    """Reject a patch."""
    patch = show_patch(patch_id, directory)
    if not patch:
        print(f"Patch not found: {patch_id}")
        return False

    patch['status'] = 'rejected'
    patch['rejected_at'] = datetime.now().isoformat()
    patch['rejection_reason'] = reason

    patch_file = Path(directory) / f"{patch['id']}.json"
    with open(patch_file, 'w', encoding='utf-8') as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    print(f"Patch rejected: {patch_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Manage content patches")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all patches")
    list_parser.add_argument("--status", choices=["proposed", "applied", "rejected"],
                            help="Filter by status")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show patch details")
    show_parser.add_argument("patch_id", help="Patch ID or partial ID")

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a patch")
    apply_parser.add_argument("patch_id", help="Patch ID")
    apply_parser.add_argument("--dry-run", action="store_true", default=True,
                             help="Show what would be changed (default)")
    apply_parser.add_argument("--force", action="store_true",
                             help="Actually apply the patch")

    # Reject command
    reject_parser = subparsers.add_parser("reject", help="Reject a patch")
    reject_parser.add_argument("patch_id", help="Patch ID")
    reject_parser.add_argument("--reason", "-r", default="", help="Rejection reason")

    args = parser.parse_args()

    if args.command == "list":
        patches = list_patches()
        if args.status:
            patches = [p for p in patches if p.get('status') == args.status]

        if not patches:
            print("No patches found.")
        else:
            print(f"\n{'ID':<25} {'Status':<12} {'Type':<20} {'Query':<30}")
            print("-" * 90)
            for p in patches:
                query = p.get('query', '')[:28]
                print(f"{p['id']:<25} {p.get('status', 'unknown'):<12} {p.get('type', ''):<20} {query:<30}")

    elif args.command == "show":
        patch = show_patch(args.patch_id)
        if patch:
            print(json.dumps(patch, indent=2, ensure_ascii=False))
        else:
            print(f"Patch not found: {args.patch_id}")

    elif args.command == "apply":
        apply_patch(args.patch_id, dry_run=not args.force)

    elif args.command == "reject":
        reject_patch(args.patch_id, args.reason)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
