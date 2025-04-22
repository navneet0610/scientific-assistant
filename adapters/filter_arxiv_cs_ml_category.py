import json

# Paths
INPUT_FILE = "datasets/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "datasets/filtered_arxiv_cs_statml.json"

# Filter prefixes
VALID_PREFIXES = ("cs.", "stat.ML")
BATCH_SIZE = 1000

def filter_and_write_batch(batch, outfile):
    for line in batch:
        try:
            record = json.loads(line)
            categories = record.get("categories", "")
            category_list = categories.split()

            # Apply category filter
            if not any(cat.startswith(prefix) for prefix in VALID_PREFIXES for cat in category_list):
                continue

            # Extract relevant fields
            filtered_record = {
                "arxivid": record.get("id", ""),
                "title": record.get("title", ""),
                "abstract": record.get("abstract", ""),
                "authors": record.get("authors", ""),
                "journal": record.get("journal-ref", ""),
                "license": record.get("license", ""),
                "categories": categories
            }

            json.dump(filtered_record, outfile)
            outfile.write("\n")

        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Batch processing loop
line_count = 0
with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    batch = []
    for line in infile:
        line_count += 1
        batch.append(line)

        # Log progress every 10k lines
        if line_count % 10000 == 0:
            print(f"Processed {line_count:,} lines...")

        if len(batch) == BATCH_SIZE:
            filter_and_write_batch(batch, outfile)
            batch.clear()

    # Final batch
    if batch:
        filter_and_write_batch(batch, outfile)

print(f" Done! Total lines processed: {line_count:,}")
