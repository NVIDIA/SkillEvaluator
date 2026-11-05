# Offline report chart resources

These assets are public upstream software, included in the Python wheel and
source distribution so Tier 3 HTML reports work without network access. They
are read from fixed package resource names, cached, and embedded only when a
report has chart panels. No JavaScript package manager is needed at runtime.

## Chart.js 4.5.1

- Upstream: <https://github.com/chartjs/Chart.js/tree/v4.5.1>
- Original bundle: <https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js>
- License source: `package/LICENSE.md` in
  <https://registry.npmjs.org/chart.js/-/chart.js-4.5.1.tgz>, retained verbatim as
  `Chart.js-LICENSE.md` (MIT).
- Original SHA-384 (verified against the report's previous SRI before editing):
  `jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ`
- Original SHA-256:
  `48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a`
  (208,522 bytes).
- Vendored `chart.umd.min.js` SHA-256:
  `31afbecadaef6da89edf023df400e08b3dfb44dd099c57e711d5e69c4c731784`
  (208,481 bytes).

The sole modification is removal of the 41 ASCII bytes
`//# sourceMappingURL=chart.umd.min.js.map`. Both surrounding newlines are
retained. This prevents developer tools from requesting a missing source map.
The MIT header and bundled dependency notice remain intact. To reproduce the
asset, download the original URL, verify the original SHA-384 above, remove
that exact directive, and verify the vendored SHA-256. Do not re-minify it.

## Bundled @kurkle/color 0.3.2

The bundle's own header identifies version 0.3.2, regardless of versions that
newer dependency resolution might select from Chart.js's version range.

- Upstream: <https://github.com/kurkle/color/tree/v0.3.2>
- License source: `package/LICENSE.md` in
  <https://registry.npmjs.org/@kurkle/color/-/color-0.3.2.tgz>, retained verbatim as
  `kurkle-color-LICENSE.md` (MIT).

Both full license texts also appear in generated chart-bearing reports and
the distribution's `THIRD_PARTY_NOTICES.md`.
