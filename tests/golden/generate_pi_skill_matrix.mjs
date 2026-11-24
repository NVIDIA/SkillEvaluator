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
// Two recording paths:
// - A target that is a pi default root (its path ends in .pi/skills or .agents/skills) is
//   recorded through normal discovery: the real DefaultResourceLoader with a trusted project
//   and cwd = the project directory around it (pi reads <cwd>/.pi/skills and walks up from
//   cwd for .agents/skills), so collectSkillEntries' root/mode rule applies (package-manager.js:
//   loose .md files load at a .pi root, nested under .agents/skills).
//   The loader also reads this machine's ~/.agents/skills, so only skills and diagnostics
//   under the sandbox root are recorded.
// - Any other target is passed to loadSkills as an explicit skillPaths directory, which
//   loadSkillsFromDirInternal (skills.js) scans with loose .md files at its own root.
// A case whose files are not stored under their exact spelling (a case-folding filesystem
// merged SKILL.md and skill.md into one file) is refused before anything is recorded.
// Output per case: {id, skills: [...], diagnostics: [...]} with the sandbox root replaced by <root>.
import { readFileSync, readdirSync, mkdtempSync, writeFileSync, mkdirSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join, dirname, relative, basename } from "path";
import { createRequire } from "module";

const PI = "/Users/sergeykostrov/.pi/agent/install/releases/1.6.0/node_modules/@earendil-works/pi-coding-agent";
const require = createRequire(PI + "/package.json");
const { version } = require(PI + "/package.json");
const { loadSkills } = await import(PI + "/dist/core/skills.js");
const { DefaultResourceLoader } = await import(PI + "/dist/core/resource-loader.js");
const { SettingsManager } = await import(PI + "/dist/core/settings-manager.js");

const casesPath = process.argv[2];
if (!casesPath) {
  console.error("usage: node generate_pi_skill_matrix.mjs <pi_skill_matrix_cases.json>");
  process.exit(2);
}
const { cases } = JSON.parse(readFileSync(casesPath, "utf8"));

// On a case-folding filesystem the second spelling overwrites the first, so the record
// would describe a tree that does not exist; refuse it instead.
function refuseFoldedNames(c, written) {
  for (const p of written) {
    const held = readdirSync(dirname(p));
    if (!held.includes(basename(p))) {
      console.error(`${c.id}: ${p} is not stored under that spelling; ${dirname(p)} holds: ${held.join(", ")}`);
      process.exit(1);
    }
  }
}

const isDefaultRoot = (p) => p.endsWith("/.pi/skills") || p.endsWith("/.agents/skills");
const under = (root) => (p) => {
  const rel = relative(root, p);
  return rel !== "" && !rel.startsWith("..");
};

const results = [];
for (const c of cases) {
  const root = mkdtempSync(join(tmpdir(), "pi-probe-"));
  const agentDir = mkdtempSync(join(tmpdir(), "pi-probe-agent-")); // empty user dir: no outside skills
  try {
    const written = [];
    for (const [rel, content] of Object.entries(c.files)) {
      const p = join(root, rel);
      written.push(p);
      if (content === null) mkdirSync(p, { recursive: true });
      else {
        mkdirSync(p.slice(0, p.lastIndexOf("/")), { recursive: true });
        writeFileSync(p, content);
      }
    }
    refuseFoldedNames(c, written);
    const targets = c.targets.map((t) => join(root, t));
    let res;
    if (targets.length === 1 && isDefaultRoot(targets[0])) {
      const loader = new DefaultResourceLoader({
        cwd: dirname(dirname(targets[0])),
        agentDir,
        settingsManager: SettingsManager.inMemory({}, { projectTrusted: true }),
        noExtensions: true,
        noPromptTemplates: true,
        noThemes: true,
        noContextFiles: true,
      });
      await loader.reload();
      const inRoot = under(root);
      const all = loader.getSkills();
      res = { skills: all.skills.filter((s) => inRoot(s.filePath)), diagnostics: all.diagnostics.filter((d) => inRoot(d.path)) };
    } else {
      res = loadSkills({
        cwd: root,
        agentDir,
        skillPaths: targets,
        includeDefaults: false,
      });
    }
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
