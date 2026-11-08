#!/bin/bash
set -e

echo "Setting up pre commit hooks for clearing notebooks"
uv run pre-commit install

echo "Done!"
