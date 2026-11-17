# Rubric

Score each item 0 or 1.

1. `customers_clean.csv` exists and has the header `id,name,email`.
2. Every email appears once, compared case-insensitively after trimming spaces.
3. The first occurrence of each email is the row that is kept.
4. No unique customer row is lost.
5. The answer reports that 2 rows were removed.
