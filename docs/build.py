#!/usr/bin/env python3
"""
Simple build script for ChemographyKit documentation.

This script provides a convenient way to build the documentation
with proper error handling and cleanup.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def clean_build(build_dir):
    """Clean the build directory."""
    if build_dir.exists():
        print(f"Cleaning {build_dir}")
        shutil.rmtree(build_dir)


def build_docs(source_dir, build_dir, builder="html", clean=False):
    """Build the documentation."""
    if clean:
        clean_build(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sphinx-build",
        "-b",
        builder,
        "-W",  # Treat warnings as errors
        str(source_dir),
        str(build_dir / builder),
    ]

    result = run_command(cmd, check=False)

    if result.returncode == 0:
        print(f"\n✅ Documentation built successfully!")
        print(f"📁 Output: {build_dir / builder}")
        if builder == "html":
            index_file = build_dir / builder / "index.html"
            print(f"🌐 Open: {index_file}")
    else:
        print(f"\n❌ Documentation build failed!")
        sys.exit(1)


def serve_docs(build_dir, port=8000):
    """Serve the documentation locally."""
    html_dir = build_dir / "html"
    if not html_dir.exists():
        print("❌ HTML documentation not found. Build it first with: python build.py")
        sys.exit(1)

    print(f"🚀 Serving documentation at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        cmd = ["python", "-m", "http.server", str(port)]
        subprocess.run(cmd, cwd=html_dir, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


def check_links(source_dir, build_dir):
    """Check for broken links in the documentation."""
    print("🔍 Checking links...")

    cmd = [
        "sphinx-build",
        "-b",
        "linkcheck",
        str(source_dir),
        str(build_dir / "linkcheck"),
    ]

    result = run_command(cmd, check=False)

    if result.returncode == 0:
        print("✅ All links are valid!")
    else:
        print("⚠️  Some links may be broken. Check the output above.")


def main():
    parser = argparse.ArgumentParser(description="Build ChemographyKit documentation")
    parser.add_argument(
        "command",
        choices=["build", "clean", "serve", "linkcheck", "livehtml"],
        help="Command to run",
    )
    parser.add_argument(
        "--builder",
        "-b",
        default="html",
        choices=["html", "pdf", "epub", "latex"],
        help="Sphinx builder to use",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port for serving documentation"
    )

    args = parser.parse_args()

    # Set up paths
    docs_dir = Path(__file__).parent
    source_dir = docs_dir
    build_dir = docs_dir / "_build"

    if args.command == "build":
        build_docs(source_dir, build_dir, args.builder)

    elif args.command == "clean":
        clean_build(build_dir)
        print("✅ Build directory cleaned")

    elif args.command == "serve":
        serve_docs(build_dir, args.port)

    elif args.command == "linkcheck":
        check_links(source_dir, build_dir)

    elif args.command == "livehtml":
        print("🔄 Starting live HTML build (auto-reload on changes)")
        print("Press Ctrl+C to stop")

        cmd = [
            "sphinx-autobuild",
            str(source_dir),
            str(build_dir / "html"),
            "--port",
            str(args.port),
            "--open-browser",
        ]

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n👋 Live build stopped")


if __name__ == "__main__":
    main()
