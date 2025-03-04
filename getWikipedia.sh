#!/bin/bash
# getWikipedia.sh
# This script downloads the latest English Wikipedia pages-articles dump,
# extracts the text using WikiExtractor, and organizes the output into a directory.

# Set variables
WIKI_DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
DUMP_FILE="enwiki-latest-pages-articles.xml.bz2"
EXTRACTED_DIR="wikipedia_extracted"
OUTPUT_DIR="wikipedia_text"

# Download the Wikipedia dump if not already downloaded
if [ ! -f "$DUMP_FILE" ]; then
    echo "Downloading Wikipedia dump from $WIKI_DUMP_URL..."
    wget --continue "$WIKI_DUMP_URL"
else
    echo "Wikipedia dump already downloaded."
fi

# Create directory for extracted files if it doesn't exist
if [ ! -d "$EXTRACTED_DIR" ]; then
    mkdir "$EXTRACTED_DIR"
fi

# Run WikiExtractor using the pip-installed executable
echo "Extracting Wikipedia dump using wikiextractor..."
wikiextractor -o "$EXTRACTED_DIR" --no-templates "$DUMP_FILE"

# (Optional) Combine extracted files into a single file for easier processing.
echo "Combining extracted text files into a single file..."
mkdir -p "$OUTPUT_DIR"
cat $(find "$EXTRACTED_DIR" -type f -name "*.txt") > "$OUTPUT_DIR/wikipedia_combined.txt"

echo "Wikipedia extraction and combination complete. The combined file is located at: $OUTPUT_DIR/wikipedia_combined.txt"
