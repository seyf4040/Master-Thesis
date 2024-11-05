import re
from pathlib import Path

# Regular expression to match the transclusion syntax
INCLUDE_PATTERN = r"\{\{include (.+?)\}\}"

def process_file(filepath):
    """Processes a single Markdown file for transclusions."""
    content = filepath.read_text()

    # Search for all placeholders in the file
    def replace_include(match):
        include_path = match.group(1).strip()
        include_file = Path(include_path)

        # Ensure the included file exists
        if include_file.is_file():
            return include_file.read_text()
        else:
            print(f"Warning: {include_path} does not exist.")
            return f"*Error: File '{include_path}' not found.*"

    # Substitute all placeholders with the content of the specified files
    new_content = re.sub(INCLUDE_PATTERN, replace_include, content)

    # Only write back if there's a change
    if new_content != content:
        filepath.write_text(new_content)
        print(f"Processed {filepath}")

def main():
    # Process all Markdown files in the repository
    for filepath in Path(".").rglob("*.md"):
        process_file(filepath)

if __name__ == "__main__":
    main()
