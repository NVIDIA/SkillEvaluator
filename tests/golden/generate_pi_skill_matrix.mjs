#!/usr/bin/env node
// Regenerate tests/golden/pi_skill_matrix.json: run the pinned pi skills loader
// against the fixture cases and print recorded outcomes as JSON.
//
// Dev-only (CI has no node). Usage:
//   node tests/golden/generate_pi_skill_matrix.mjs tests/golden/pi_skill_matrix_cases.json \
//     > tests/golden/pi_skill_matrix.json
// PI below must point at the installed release matching pi_version in the golden file;
// bump it and regenerate whenever the pinned pi version changes.
//
// Each case: {id, files: {relpath: content|null(dir)}, targets: [dir to load, relative to root]}
// Output per case: {id, skills: [...], diagnostics: [...]} with the sandbox root replaced by <root>.
import { readFileSync, mkdtempSync, writeFileSync, mkdirSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join, relative } from "path";
import { createRequire } from "module";

const PI = "/Users/sergeykostrov/.pi/agent/install/releases/1.6.0/node_modules/@earendil-works/pi-coding-agent";
const require = createRequire(PI + "/package.json");
const { version } = require(PI + "/package.json");
const { loadSkills } = await import(PI + "/dist/core/skills.js");

const casesPath = process.argv[2];
if (!casesPath) {
  console.error("usage: node generate_pi_skill_matrix.mjs <pi_skill_matrix_cases.json>");
  process.exit(2);
}
const { cases } = JSON.parse(readFileSync(casesPath, "utf8"));

const results = [];
for (const c of cases) {
  const root = mkdtempSync(join(tmpdir(), "pi-probe-"));
  const agentDir = mkdtempSync(join(tmpdir(), "pi-probe-agent-")); // empty user dir: no outside skills
  try {
    for (const [rel, content] of Object.entries(c.files)) {
      const p = join(root, rel);
      if (content === null) mkdirSync(p, { recursive: true });
      else {
        mkdirSync(p.slice(0, p.lastIndexOf("/")), { recursive: true });
        writeFileSync(p, content);
      }
    }
    const res = loadSkills({
      cwd: root,
      agentDir,
      skillPaths: c.targets.map((t) => join(root, t)),
      includeDefaults: false,
    });
    const scrub = (x) => JSON.parse(JSON.stringify(x).split(root).join("<root>"));
    results.push({
      id: c.id,
      pi_version: version,
      skills: res.skills.map((s) => ({
        name: s.name,
        description: s.description,
        disableModelInvocation: s.disableModelInvocation ?? null,
        filePath: relative(root, s.filePath),
        source: s.source,
      })),
      diagnostics: scrub(res.diagnostics),
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
    rmSync(agentDir, { recursive: true, force: true });
  }
}
console.log(JSON.stringify({ generator: "generate_pi_skill_matrix.mjs", pi_version: results[0]?.pi_version, cases: results }, null, 2));
