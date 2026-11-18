`code/totals.py` crashes on `code/fixtures/orders.csv`. Fix it so malformed
rows are skipped with a warning on stderr and the grand total is printed as
`TOTAL 42.50`.
