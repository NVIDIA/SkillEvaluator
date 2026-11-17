---
name: csv-cleaner
description: Use when cleaning CSV files, such as removing duplicate rows or normalizing column values.
---

# CSV Cleaner

Clean CSV files with Python's `csv` module.

## Instructions

1. Read the input CSV and keep its header.
2. Normalize the key column before comparing rows: trim spaces and compare case-insensitively.
3. Keep the first occurrence of each key and write the result to a new file.
4. Report how many rows were removed.
