/* Original local workbench client. SPDX-License-Identifier: Apache-2.0 */
"use strict";

(() => {
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const json = (value) => JSON.stringify(value, null, 2);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const activeStatus = (status) => ["running", "stopping", "pending", "queued"].includes(status);
  const capability = (value) => typeof value === "object" && value !== null ? Boolean(value.available) : Boolean(value);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const number = (value, digits = 3) => finite(value) ? value.toFixed(digits).replace(/\.?0+$/, "") : "Not available";
  const scalar = (value) => finite(value) ? number(value) : value == null ? "Not available" : typeof value === "string" ? value : "See retained report";
  const time = (value) => value ? new Date(value).toLocaleString(undefined, {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}) : "Time not recorded";
  const actionNames = {draft:"Dataset draft",validate:"Dataset validation",check:"Skill checks",evaluate:"Agent evaluation",improve:"Proposing changes",compare_improvement:"Comparing results"};
  const ui = {state:null,skillId:null,stage:"skill",view:"workspace",draft:null,draftMode:null,datasetDirty:false,guidance:{task:"",success:"",avoid:""},generation:null,jobId:null,selectedCase:0,arm:"with_skill",attempt:0,detail:"checks",pollTimer:null,toastTimer:null,busy:new Set(),evalOptions:{env_mode:"local",n_attempts:1,n_concurrent:1,skip_baseline:false,agent:"",model:"",timeout_multiplier:3,grading_mode:"default",tier1:false,tier1_llm:false,dedup:false},candidateContent:null,pasteContent:"",addingSkill:false,returnFocus:null,caseRevision:0,draftStartRevision:null,pendingDraft:null,improvement:{},logViews:{}};
  const selectedSkill = () => ui.state?.skills?.find((s) => s.id === ui.skillId) || null;
  const jobById = (id) => ui.state?.jobs?.find((job) => job.id === id);
  const selectedJob = () => jobById(ui.jobId);
  const latestJob = (skillId, action) => [...(ui.state?.jobs || [])].filter((j) => j.skill_id === skillId && (!action || j.action === action)).sort((a,b) => String(b.started_at || "").localeCompare(String(a.started_at || "")))[0];
  const skillJobs = () => (ui.state?.jobs || []).filter((j) => j.skill_id === ui.skillId);
  const themeKey = "skillevaluator.studio.theme";
  const systemAppearance = window.matchMedia("(prefers-color-scheme: dark)");
  let themePreference = "system";
  try { const saved=localStorage.getItem(themeKey); if(["system","light","dark"].includes(saved))themePreference=saved; } catch {}
  function applyTheme(preference, persist=false) {
    themePreference=["system","light","dark"].includes(preference)?preference:"system";
    document.documentElement.dataset.theme=themePreference==="system"?(systemAppearance.matches?"dark":"light"):themePreference;
    document.documentElement.dataset.themePreference=themePreference;
    const control=$("#theme-select");if(control)control.value=themePreference;
    if(persist){try{localStorage.setItem(themeKey,themePreference);}catch{}}
  }
  applyTheme(themePreference);
  const followSystem=()=>{if(themePreference==="system")applyTheme("system");};
  if(systemAppearance.addEventListener)systemAppearance.addEventListener("change",followSystem);else systemAppearance.addListener(followSystem);
  window.addEventListener("storage",event=>{if(event.key===themeKey)applyTheme(event.newValue);});
  const modeKey="skillevaluator.studio.mode";
  function applyMode(mode,persist=false) {
    const advanced=mode==="advanced";
    document.documentElement.dataset.mode=advanced?"advanced":"simple";
    $$("[name=display_mode]").forEach(input=>{input.checked=input.value===(advanced?"advanced":"simple");});
    if(persist){try{localStorage.setItem(modeKey,advanced?"advanced":"simple");}catch{}}
  }
  try{applyMode(localStorage.getItem(modeKey));}catch{applyMode("simple");}
  window.addEventListener("storage",event=>{if(event.key===modeKey)applyMode(event.newValue);});
  function hasUnsavedEdits() {
    return ui.datasetDirty||Boolean(ui.candidateContent!==null&&ui.candidateContent!==selectedSkill()?.content)||Boolean(ui.pasteContent.trim());
  }
  function allowSkillSwitch() {
    return !hasUnsavedEdits()||window.confirm("You have unsaved changes. Discard them and switch skills? Choose Cancel to stay here and save your work.");
  }
  function applyDraft(job) {
    ui.draft=(job.result.cases || []).map(value=>({value:clone(value),selected:true}));
    ui.draftMode=job.result.generation_mode;ui.datasetDirty=true;ui.pendingDraft=null;ui.caseRevision++;
  }
  function renderDraftReady() {
    if(!ui.pendingDraft)return "";
    return '<div class="note attention draft-ready-note"><strong>Your new draft is ready. Your edits are safe.</strong><br>Review the completed draft when you’re ready. Your current cases stay here until you choose to replace them.<div class="button-row"><button class="button secondary small" data-action="review-ready-draft">Review new draft</button><button class="text-button" data-action="keep-current-cases">Keep current cases</button></div></div>';
  }
  window.addEventListener("beforeunload",event=>{if(hasUnsavedEdits()){event.preventDefault();event.returnValue="";}});


  async function api(path, options = {}) {
    const method = options.method || "GET";
    const headers = {"Accept":"application/json",...(options.headers || {})};
    if (options.body !== undefined) headers["Content-Type"] = "application/json";
    if (method !== "GET" && method !== "HEAD") headers["X-Studio-Token"] = ui.state?.token || "";
    const response = await fetch(path, {...options,method,headers,cache:"no-store",body:options.body === undefined ? undefined : JSON.stringify(options.body)});
    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json") ? await response.json() : {error:await response.text()};
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : data.error?.message || data.message || `Request failed (${response.status}).`);
    return data;
  }

  function notify(message, error = false) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.className = `toast${error ? " error" : ""}`;
    toast.hidden = false;
    clearTimeout(ui.toastTimer);
    ui.toastTimer = setTimeout(() => { toast.hidden = true; }, error ? 10000 : 5500);
  }

  function showActionError(message, button) {
    const scope=button?.closest("form,section,article,.draft-actions") || $("#workspace-content");
    if(!scope)return;
    let error=scope.querySelector("[data-action-error]");
    if(!error){error=document.createElement("p");error.className="banner error section-gap";error.dataset.actionError="";error.setAttribute("role","alert");scope.append(error);}
    error.textContent=message;error.hidden=!message;
  }

  async function doAction(key, task, button) {
    if (ui.busy.has(key)) return;
    ui.busy.add(key);
    button?.closest("form,section,article,.draft-actions")?.querySelector("[data-action-error]")?.remove();
    if (button) {button.disabled = true;button.setAttribute("aria-busy","true");}
    try { return await task(); }
    catch (error) { const message=error.message || "The action could not be completed.";showActionError(message,button);notify(message,true); }
    finally {ui.busy.delete(key);if(button?.isConnected){button.disabled=false;button.removeAttribute("aria-busy");}}
  }

  function connection(ok, message = "") {
    $("#connection-label").textContent = ok ? "Local server connected" : "Server disconnected";
    $("#connection-dot").classList.toggle("offline", !ok);
    const banner = $("#connection-error");
    banner.hidden = ok;
    if (!ok) banner.innerHTML = `<strong>Could not reach the local server.</strong> ${esc(message)} <button class="text-button" data-action="refresh">Try again</button>`;
  }

  async function refreshState() {
    try {
      const previousJobs=ui.state?.jobs || [];
      ui.state = await api("/api/state");
      ui.state.skills ||= [];
      ui.state.jobs ||= [];
      ui.state.jobs=ui.state.jobs.map((job)=>{const previous=previousJobs.find((j)=>j.id===job.id);return previous?.result?{...previous,...job,result:previous.result,logs:previous.logs,events:previous.events}:job;});
      connection(true);
      if (ui.skillId && !selectedSkill()) ui.skillId = null;
      if (!ui.skillId && !ui.addingSkill && ui.state.skills.length) selectSkill(ui.state.skills[ui.state.skills.length - 1].id, false);
      if(ui.jobId)await fetchJob(ui.jobId);
      await loadComparisonJobs();
      if (!ui.evalOptions.agent) ui.evalOptions.agent = ui.state.capabilities?.agents?.find((a) => a.available)?.id || ui.state.capabilities?.agents?.[0]?.id || "opencode";
      if (!ui.evalOptions.model) ui.evalOptions.model = ui.state.provider?.model || "";
      if (capability(ui.state.capabilities?.docker) && !capability(ui.state.capabilities?.local)) ui.evalOptions.env_mode = "docker";
      renderAll();
      schedulePoll();
      return ui.state;
    } catch (error) {
      connection(false,error.message);
      if (!ui.state) $("#workspace-content").innerHTML = `<div class="surface empty-state"><h2>Your skills are waiting.</h2><p>Keep the local SkillEvaluator server running, then reconnect here.</p><button class="button primary" data-action="refresh">Reconnect</button></div>`;
      throw error;
    }
  }

  function updateSkill(skill) {
    const i = ui.state.skills.findIndex((s) => s.id === skill.id);
    if (i < 0) ui.state.skills.push(skill); else ui.state.skills[i] = skill;
  }

  async function fetchJob(id) {
    const job=await api(`/api/jobs/${encodeURIComponent(id)}`);
    const index=ui.state.jobs.findIndex((item)=>item.id===job.id);
    if(index<0)ui.state.jobs.push(job);else ui.state.jobs[index]=job;
    return job;
  }

  async function loadComparisonJobs() {
    const skill=selectedSkill();
    if(!skill?.parent_id)return;
    const ids=[latestJob(skill.parent_id,"evaluate")?.id,latestJob(skill.id,"evaluate")?.id].filter(Boolean);
    const updates=await Promise.allSettled(ids.map(fetchJob));
    updates.forEach((result)=>{if(result.status==="rejected")notify(`Could not load comparison evidence: ${result.reason.message}`,true);});
  }

  function selectSkill(id, render = true) {
    ui.skillId = id;
    ui.addingSkill=false;ui.pasteContent="";
    ui.draft = null;
    ui.datasetDirty = false;
    ui.draftMode = null;
    ui.caseRevision=0;ui.draftStartRevision=null;ui.pendingDraft=null;
    ui.guidance = {task:"",success:"",avoid:""};
    ui.candidateContent = null;
    ui.improvement={};
    ui.jobId = latestJob(id)?.id || null;
    ui.selectedCase = 0;
    ui.attempt = 0;
    if (render) renderAll();
  }

  function jobStage(job) {
    if(job.validation_of||["improve","compare_improvement"].includes(job.action))return "improve";
    if(["draft","validate"].includes(job.action))return "cases";
    return job.action==="check"&&!job.options?.tier1&&!job.options?.dedup?"skill":"evaluate";
  }

  function setStage(stage) {
    if (stage !== "skill" && !selectedSkill()) {
      notify("Add a skill first, or try the working sample.");
      stage = "skill";
    }
    ui.stage = stage;
    ui.view = "workspace";
    renderAll();
    if(stage==="improve")loadImprovement();
    if(stage==="evaluate"){
      const job=latestJob(ui.skillId,"evaluate"),skillId=ui.skillId;
      if(job&&!job.result)fetchJob(job.id).then(()=>{if(ui.skillId===skillId&&ui.stage==="evaluate")renderAll();}).catch(error=>notify(error.message,true));
    }
    $("#workspace-content").scrollIntoView({block:"nearest"});
  }

  function renderChrome() {
    const skill = selectedSkill();
    const titles={skill:skill?"Test your skill.":"Start with a skill.",cases:"Review your test cases.",evaluate:"See how your skill performs.",improve:"Review a possible improvement."};
    const guidance={skill:skill?"Review the examples you want to test, then run an evaluation.":"Add your instructions. We’ll help you build a small set of examples.",cases:"Check the task and expected result. Expand a case to edit it.",evaluate:ui.state.provider?.configured?"Run your saved cases, then inspect the actual responses and checks.":"Connect a model to run your saved cases. Optional checks are included below.",improve:"Use a completed evaluation to propose a change, then test the proposed copy."};
    $(".heading-copy h1").textContent=titles[ui.stage];
    $(".heading-copy .lede").textContent=guidance[ui.stage];
    $("#skill-breadcrumb").textContent = skill?.name || "New skill";
    $("#view-breadcrumb").textContent = ui.view === "runs" ? "Run history" : "Skills";
    $("#app-version").textContent = ui.state?.version ? `· ${ui.state.version}` : "";
    $("#local-address").textContent = location.host;
    const running = (ui.state?.jobs || []).filter((j) => activeStatus(j.status)).length;
    $("#running-count").textContent = String(running);
    $("#running-count").hidden = !running;
    $$(".nav-item[data-view]").forEach((node) => {const active=node.dataset.view===ui.view;node.classList.toggle("active",active);if(active)node.setAttribute("aria-current","page");else node.removeAttribute("aria-current");});
    $$(".stage").forEach((node) => {const active=node.dataset.stage===ui.stage;node.classList.toggle("active",active);if(active)node.setAttribute("aria-current","step");else node.removeAttribute("aria-current");});
    $("#workspace-view").hidden = ui.view !== "workspace";
    $("#runs-view").hidden = ui.view !== "runs";
    const picker = $("#skill-picker");
    $("#skill-picker-wrap").hidden = !ui.state?.skills?.length;
    $("#add-skill-button").hidden = !ui.state?.skills?.length;
    const runningJob=(ui.state?.jobs || []).find(job=>activeStatus(job.status));
    $("#background-task-banner").hidden=!runningJob;
    if(runningJob)$("#background-task-banner").innerHTML=`<div><strong>${esc(actionNames[runningJob.action] || "A task")} for ${esc(ui.state.skills.find(item=>item.id===runningJob.skill_id)?.name || "a saved skill")} ${runningJob.status==="stopping"?"is stopping":"is running"}.</strong><p>One background task at a time in this app. You can keep editing or add another skill. An evaluation can include multiple cases and concurrent trials.</p></div><button class="button secondary small" data-action="open-job" data-id="${esc(runningJob.id)}">Open task</button>`;
    picker.innerHTML = (!ui.skillId?'<option value="">Choose a saved skill…</option>':"")+(ui.state?.skills || []).map((s) => `<option value="${esc(s.id)}"${s.id===ui.skillId ? " selected" : ""}>${esc(s.name)}${s.parent_id ? " · candidate" : ""}</option>`).join("");
  }

  function renderAll() {
    if (!ui.state) return;
    const focused=document.activeElement?.id;
    rememberJobLog();
    renderChrome();
    if (ui.view === "runs") {renderHistory();return;}
    const content = $("#workspace-content");
    if (ui.stage === "skill") content.innerHTML = renderSkill();
    if (ui.stage === "cases") content.innerHTML = renderDataset();
    if (ui.stage === "evaluate") content.innerHTML = renderEvaluate();
    if (ui.stage === "improve") content.innerHTML = renderImprove();
    renderJobRegion();
    syncDraftMethod();
    syncEvaluationAction();
    if(focused)document.getElementById(focused)?.focus({preventScroll:true});
  }

  function skillContext(skill) {
    return `<aside class="surface skill-context" aria-label="Selected skill"><div class="skill-identity"><span class="document-glyph mono" aria-hidden="true">MD</span><div><h3>${esc(skill.name)}</h3><p>SKILL.md · ${(skill.files || []).length} files</p></div></div><span class="badge ${skill.parent_id ? "attention" : "good"}">${skill.parent_id ? "Candidate copy" : "Local copy"}</span><p class="skill-description">${esc(skill.description || "Your skill instructions, ready for a small test set.")}</p><pre class="source-excerpt" tabindex="0" aria-label="Skill source excerpt">${esc((skill.content || "").slice(0,370))}${(skill.content || "").length>370 ? "\n…" : ""}</pre><div class="skill-stats"><div><span>Saved test cases</span><strong>${skill.cases?.length || 0}</strong></div><div><span>Local runs</span><strong>${skillJobs().length}</strong></div></div><div class="button-row"><button class="button secondary small" data-action="source">View source</button><a class="button secondary small" href="/api/skills/${encodeURIComponent(skill.id)}/export">Download skill (.zip)</a></div><p class="fine-print section-gap">Downloads include saved files and test cases. Save case edits first.</p><p class="hash-note">Source <code>${esc((skill.source_hash || "not recorded").slice(0,12))}</code></p><p class="fine-print section-gap">We keep a separate local copy. Your original files stay unchanged.</p></aside>`;
  }

  function renderSkill() {
    const skill=selectedSkill();
    const evaluation=skill&&latestJob(skill.id,"evaluate"),current=evaluation?.status==="succeeded"&&evaluation.source_hash&&evaluation.dataset_hash&&evaluation.source_hash===skill?.source_hash&&evaluation.dataset_hash===skill?.dataset_hash;
    const ready=skill?.cases?.length&&skill.cases.every(c=>c.authoring_status==="qualified");
    const next=current?`data-action="open-job" data-id="${esc(evaluation.id)}"`:ready||skill?.native_custom_only?'data-stage="evaluate"':'data-stage="cases"';
    const label=current?"View evaluation results":ready||skill?.native_custom_only?"Evaluate skill":skill?.cases?.length?"Review test cases":"Create test cases";
    if(skill)return `<div class="work-grid"><div class="advanced-only"><p class="section-label">YOUR SKILL</p>${skillContext(skill)}</div><div><section class="surface panel"><h2>${esc(skill.name)}</h2><p class="muted help-caption">${esc(skill.description || "Your skill is ready for test cases.")}</p>${skill.parent_id?'<p class="note attention section-gap">Candidate copy · evaluate before deciding to keep it.</p>':""}<p class="fine-print section-gap">Your original files stay unchanged.</p><div class="form-actions"><span>${skill.cases?.length || 0} saved test cases</span><button class="button primary" ${next}>${label} →</button></div></section><details class="surface panel section-gap"><summary>Skill files</summary><div class="button-row section-gap"><button class="button secondary small" data-action="source">View source</button><a class="button secondary small" href="/api/skills/${encodeURIComponent(skill.id)}/export">Download skill (.zip)</a></div><p class="fine-print section-gap">Exports the saved skill and test cases. Save edits first.</p></details><details class="surface panel section-gap advanced-only"><summary>Basic checks</summary><div class="button-row section-gap"><button class="button secondary small" data-job="check">Check structure and quality</button><button class="button secondary small" data-job="validate">Validate dataset</button></div></details><div id="job-region"></div></div></div>`;
    return `<section class="surface import-panel simple-width"><h2>Add a skill.</h2><p class="local-copy-note">We keep a local copy. Your original files stay unchanged.</p><div id="dropzone" class="dropzone"><span class="upload-symbol" aria-hidden="true">↥</span><strong>Choose or drop your skill</strong><p class="field-hint">SKILL.md or .zip · up to 12 MB</p><label class="sr-only" for="skill-file">Add a skill from a file</label><input id="skill-file" type="file" accept=".md,.zip,text/markdown,application/zip"></div><details id="paste-details"${ui.pasteContent?" open":""}><summary>Or paste SKILL.md</summary><form id="paste-form"><label class="sr-only" for="skill-text">Skill document</label><textarea id="skill-text" class="code-input" rows="6" placeholder="---&#10;name: my-skill&#10;description: What this skill helps with&#10;---&#10;&#10;Your instructions…" required>${esc(ui.pasteContent)}</textarea><div class="form-actions"><button type="submit" class="button primary">Add this skill →</button></div></form></details><button class="text-button" data-action="sample">Try a working sample</button></section>`;
  }

  function draftRecords() {
    if (!ui.draft) ui.draft = (selectedSkill()?.cases || []).map((value) => ({value:clone(value),selected:true}));
    return ui.draft;
  }

  function renderDataset() {
    const skill=selectedSkill(),records=draftRecords(),selected=records.filter(r=>r.selected).length,job=selectedJob(),generation=ui.generation ?? (ui.state.provider?.configured?"llm":"template");
    return `<div class="work-grid"><div class="advanced-only"><p class="section-label">YOUR SKILL</p>${skillContext(skill)}</div><div><details class="surface panel draft-builder"${records.length||ui.datasetDirty?"":" open"}><summary>${records.length?"Draft more test cases":"Create test cases"}</summary><form id="draft-form"><label for="guidance-task">What should the skill do?</label><textarea id="guidance-task" name="task" rows="3" required placeholder="For example: write a short release note from merged changes, keeping features and fixes separate.">${esc(ui.guidance.task)}</textarea><div class="advanced-only"><label for="guidance-success">What should a good result include? <span class="optional">optional</span></label><textarea id="guidance-success" name="success" rows="2">${esc(ui.guidance.success)}</textarea><label for="guidance-avoid">What should it avoid? <span class="optional">optional</span></label><input id="guidance-avoid" name="avoid" value="${esc(ui.guidance.avoid)}"></div><label class="field-label">Create examples with</label><div class="inline-choice"><label class="choice"><input type="radio" name="generation" value="template"${generation==="template"?" checked":""}>Starter template · no key</label><label class="choice"><input type="radio" name="generation" value="llm"${generation==="llm"?" checked":""}>Model assistance</label></div><div class="form-actions"><span class="fine-print" id="draft-method-note">${generation==="template"?"Template draft · no key needed":"Model-assisted draft"}</span><button id="draft-submit" class="button primary" type="submit"${job?.action==="draft"&&activeStatus(job.status)?" disabled":""}>${generation==="template"?"Create starter cases":"Draft with model"} →</button></div></form><details class="section-gap advanced-only"><summary>Import a dataset</summary><label for="dataset-file">JSON cases</label><input id="dataset-file" type="file" accept=".json,application/json"><p class="field-hint">Accepts an array, or an object containing cases or evals. Imported cases need review and Save.</p></details></details><div id="job-region"></div><div id="dataset-review">${renderDraftReady()}<div class="review-heading"><div><h3>${records.length?"Review each task and expected result":ui.datasetDirty?"All test cases removed":"Or write a case yourself"}</h3>${records.length?`<p>${selected} of ${records.length} cases included</p>`:ui.datasetDirty?"<p>Save changes to remove them from the saved dataset.</p>":""}</div><button class="button secondary small" data-action="add-case">+ Add case</button></div>${ui.draftMode&&records.length?`<p class="generation-note">${ui.draftMode==="template"?"Starter template cases · review required":ui.draftMode==="llm"?"Model-assisted draft · review required":"Imported cases · review required"}</p>`:""}${records.length||ui.datasetDirty?`<div class="case-list">${records.map(renderCaseEditor).join("")}</div><div class="draft-actions"><span id="dataset-dirty" class="${ui.datasetDirty?"unsaved":""}">${ui.datasetDirty?"Unsaved changes":`${skill.cases?.length || 0} cases saved`}</span><div class="button-row">${records.length?`<button class="button secondary advanced-only" data-action="save-cases">Save cases</button><button class="button primary" data-action="continue-evaluate">${ui.datasetDirty?"Save and continue":"Continue to evaluation"} →</button>`:'<button class="button primary" data-action="save-cases">Save changes</button>'}</div></div>`:""}</div></div></div>`;
  }

  function syncDraftMethod() {
    const form=$("#draft-form");if(!form)return;
    const model=$("[name=generation]:checked",form)?.value==="llm",connect=model&&!ui.state.provider?.configured,button=$("#draft-submit",form);
    $("#draft-method-note",form).textContent=model?"Uses your skill and instructions. Review the generated examples.":"Starter cases to edit yourself. No model call.";
    button.type=connect?"button":"submit";
    if(connect)button.dataset.action="settings";else delete button.dataset.action;
    button.textContent=connect?"Connect a model →":model?"Suggest test cases →":"Create starter cases →";
  }

  function renderCaseEditor(record,i) {
    const c=record.value,checks=c.expected_behavior ?? c.assertions ?? [],behavior=Array.isArray(checks)?checks.join("\n"):String(checks);
    const reviewed=c.authoring_status==="qualified",requirements=Array.isArray(c.unresolved_requirements)&&c.unresolved_requirements.length?c.unresolved_requirements:(Array.isArray(c.reviewed_requirements)?c.reviewed_requirements:[]);
    const splits=[["discovery","Use to suggest changes"],["holdout","Reserve for validation"]];
    if(c.split&&!splits.some(([value])=>value===c.split))splits.push([c.split,`Imported role: ${c.split}`]);
    return `<article class="case-editor${record.selected?"":" excluded"}" data-case-index="${i}">
      <div class="case-editor-head"><label class="check-label"><input type="checkbox" data-case-field="selected" data-index="${i}"${record.selected?" checked":""}>Include case ${i+1}</label><div class="case-card-actions"><span class="badge ${reviewed?"good":"attention"}" data-review-badge>${reviewed?"Reviewed":"Needs review"}</span><button class="text-button case-remove" type="button" data-action="remove-case" data-index="${i}" aria-label="Remove test case ${i+1}">Remove</button></div></div>
      <div class="case-preview"><div><span class="compact-label">Task</span><p data-case-preview="question">${esc(c.question ?? c.prompt ?? "") || "Add the task you want to test."}</p></div><div><span class="compact-label">Expected result</span><p data-case-preview="ground_truth">${esc(c.ground_truth ?? c.expected_output ?? "") || "Describe what a good result should look like."}</p></div></div>
      <details class="case-fields" data-case-editor="${i}"${record.editing||!String(c.question ?? c.prompt ?? "").trim()?" open":""}><summary>Edit task and checks</summary><div>
      <label class="compact-label" for="case-question-${i}">Task</label><textarea id="case-question-${i}" data-case-field="question" data-index="${i}" rows="2">${esc(c.question ?? c.prompt ?? "")}</textarea>
      <label class="compact-label" for="case-truth-${i}">Expected result</label><textarea id="case-truth-${i}" data-case-field="ground_truth" data-index="${i}" rows="2">${esc(c.ground_truth ?? c.expected_output ?? "")}</textarea>
      <details class="case-checks"><summary>What should we check?${checks.length?` (${checks.length})`:""}</summary><label class="sr-only" for="case-behavior-${i}">Behavior checks, one per line</label><textarea id="case-behavior-${i}" data-case-field="expected_behavior" data-index="${i}" rows="2">${esc(behavior)}</textarea></details></div></details>
      ${requirements.length?`<div class="note attention section-gap"><strong>${reviewed?"Reviewed requirements":"Also verify before using this case"}</strong><ul>${requirements.map(item=>`<li>${esc(typeof item==="string"?item:json(item))}</li>`).join("")}</ul></div>`:""}
      <label class="check-label case-review"><input type="checkbox" data-case-field="reviewed" data-index="${i}"${reviewed?" checked":""}><span>I verified the inputs${requirements.length?", expected result, and listed requirements":" and expected result"}.</span></label>
      <div class="advanced-only"><label for="case-split-${i}">Case purpose</label><select id="case-split-${i}" data-case-field="split" data-index="${i}">${splits.map(([value,label])=>`<option value="${esc(value)}"${(c.split || "discovery")===value?" selected":""}>${esc(label)}</option>`).join("")}</select><p class="field-hint">Reserved cases test the change; their feedback is kept out of proposals.</p></div>
      <label class="check-label${c.expected_skill===null?"":" advanced-only"}"><input type="checkbox" data-case-field="inactive" data-index="${i}"${c.expected_skill===null?" checked":""}>This skill should not activate for this task</label><details class="advanced-only"><summary>Case details</summary><pre class="raw-object">${esc(json(c))}</pre></details>
    </article>`;
  }

  function renderEvaluate() {
    const skill=selectedSkill(),caps=ui.state.capabilities || {},provider=ui.state.provider || {},count=skill.cases?.length || 0;
    const job=selectedJob()?.skill_id===skill.id&&selectedJob()?.action==="evaluate"?selectedJob():latestJob(skill.id,"evaluate"),mode=ui.evalOptions.env_mode,info=caps[mode];
    const agents=(caps.agents || []).map(a=>`<option value="${esc(a.id)}"${ui.evalOptions.agent===a.id?" selected":""}>${esc(a.id)}${a.available?"":" · unavailable"}</option>`).join("");
    const result=job?.result?.kind==="evaluation";
    return `${result?'<details class="surface panel rerun-config"><summary>Run again</summary>':""}<section class="surface evaluation-config"><h2>Run your test cases.</h2><p class="muted help-caption">${esc(skill.name)} · ${count} saved cases</p>${!caps.tier3?'<div class="note attention section-gap">Agent evaluation needs Tier 3 dependencies. You can still use local checks below.</div>':""}<form id="evaluate-form"><div id="case-readiness-note">${renderCaseReadiness(skill)}</div>${ui.datasetDirty?'<p class="unsaved section-gap">Unsaved case edits will not be used. <a href="#cases" data-stage="cases">Review and save them first.</a></p>':""}<div class="provider-card section-gap"><div class="provider-line"><strong>${esc(providerLabel(provider.name))}</strong><button type="button" class="text-button${provider.configured?"":" advanced-only"}" data-action="settings">${provider.configured?"Connection settings":"Connect"}</button></div><p>${provider.configured?"Key configured":"Connect a model to run an agent evaluation."}</p><p class="advanced-only">${esc(provider.model || "Use the provider default model.")}</p></div><p class="field-hint section-gap" id="environment-hint">${mode==="local"?"Runs on this computer.":"Runs in Docker."}</p><p class="${capability(info)?"advanced-only fine-print":"note attention"}" id="environment-state">${esc(info?.reason || "This execution environment is unavailable.")}</p><div class="advanced-only"><label class="field-label">Execution environment</label><div class="inline-choice"><label class="choice"><input type="radio" name="env_mode" value="local"${mode==="local"?" checked":""}>Local</label><label class="choice"><input type="radio" name="env_mode" value="docker"${mode==="docker"?" checked":""}>Docker</label></div></div><details class="section-gap advanced-only" id="advanced-options"><summary>Run options</summary><div class="field-grid"><div><label for="eval-agent">Agent</label><select id="eval-agent" name="agent">${agents || '<option value="opencode">opencode</option>'}</select></div><div><label for="eval-model">Model override <span class="optional">optional</span></label><input id="eval-model" name="model" value="${esc(ui.evalOptions.model)}" placeholder="Use provider model"></div><div><label for="eval-attempts">Attempts per case and arm</label><input id="eval-attempts" type="number" name="n_attempts" min="1" max="5" value="${ui.evalOptions.n_attempts}"></div><div><label for="eval-concurrency">Concurrent tasks</label><input id="eval-concurrency" type="number" name="n_concurrent" min="1" max="4" value="${ui.evalOptions.n_concurrent}"></div><div><label for="eval-timeout">Timeout multiplier</label><input id="eval-timeout" type="number" name="timeout_multiplier" min="1" max="5" step="1" value="${ui.evalOptions.timeout_multiplier}"></div><div><label for="eval-grading">Grading mode</label><select id="eval-grading" name="grading_mode"><option value="default"${ui.evalOptions.grading_mode==="default"?" selected":""}>Default checks</option><option value="default_plus_custom"${ui.evalOptions.grading_mode==="default_plus_custom"?" selected":""}>Default + custom grader</option><option value="custom_only"${ui.evalOptions.grading_mode==="custom_only"?" selected":""}>Custom grader only</option></select></div><label class="check-label field-span"><input type="checkbox" name="skip_baseline"${ui.evalOptions.skip_baseline?" checked":""}>Skip baseline comparison · lift unavailable</label></div></details><details class="section-gap extra-checks" id="optional-check-options"${ui.evalOptions.tier1||ui.evalOptions.dedup?" open":""}><summary>Include in this evaluation <span class="optional" id="optional-check-count">${Number(ui.evalOptions.tier1)+Number(ui.evalOptions.dedup) || ""}</span></summary><div><label class="check-label"><input type="checkbox" name="tier1"${ui.evalOptions.tier1?" checked":""}><span>Scan skill <span class="optional">Tier 1</span></span></label><p class="field-hint">Structure, scripts and safety checks. No model key needed.</p><div class="scan-ai-option" id="scan-ai-option"${ui.evalOptions.tier1?"":" hidden"}><label class="check-label"><input type="checkbox" name="tier1_llm"${ui.evalOptions.tier1_llm?" checked":""}>Add AI security review <span class="optional">optional</span></label><p class="field-hint">Requires a model key.</p></div><label class="check-label"><input type="checkbox" name="dedup"${ui.evalOptions.dedup?" checked":""}>Check repeated content</label><p class="field-hint">Within this skill; no database setup. Requires an NVIDIA or OpenAI key.</p><button class="button secondary small advanced-only" id="run-selected-checks" type="button" data-action="run-selected-checks"${ui.evalOptions.tier1||ui.evalOptions.dedup?"":" disabled"}>Run checks only</button><p class="fine-print advanced-only">Runs without an agent or test dataset.</p></div></details><p class="field-hint section-gap" id="pipeline-note"${ui.evalOptions.tier1||ui.evalOptions.dedup?"":" hidden"}>Selected checks run first, then the agent runs your test cases.</p><div class="form-actions"><div class="run-count" id="run-count">${renderRunCount(count)}</div><button id="evaluation-submit" class="button primary" type="submit">Run evaluation →</button></div></form><details class="section-gap advanced-only"><summary>Basic checks · no model key needed</summary><div class="button-row section-gap"><button class="button secondary small" data-job="validate">Validate dataset</button><button class="button secondary small" data-job="check">Check structure and quality</button></div></details></section>${result?"</details>":""}<div id="job-region"></div><div id="evaluation-results">${result?renderResults(job):""}</div>`;
  }

  function usesNativeTasks(skill=selectedSkill(),options=ui.evalOptions) {
    return Boolean(skill?.native_custom_only)&&options.grading_mode==="custom_only"&&!skill.cases?.length;
  }

  function syncEvaluationAction() {
    const button=$("#evaluation-submit"),skill=selectedSkill();if(!button||!skill)return;
    const needsCases=!skill.cases?.length&&!usesNativeTasks(skill),needsReview=skill.cases?.some(c=>c.authoring_status==="draft_requires_oracle_review");
    delete button.dataset.action;delete button.dataset.stage;
    button.type=needsCases||needsReview||!ui.state.provider?.configured?"button":"submit";
    if(needsCases||needsReview)button.dataset.stage="cases";else if(!ui.state.provider?.configured)button.dataset.action="settings";
    button.textContent=(needsCases?"Create test cases":needsReview?"Review test cases":ui.state.provider?.configured?"Run evaluation":"Connect a model")+" →";
  }

  function renderCaseReadiness(skill) {
    if(skill.cases?.some(c=>c.authoring_status==="draft_requires_oracle_review"))return '<p class="note attention">Review the inputs and expected result for each draft case before evaluating.</p>';
    if(skill.cases?.length)return "";
    if(usesNativeTasks(skill))return '<p>Native task package selected. JSON dataset rows are not required; the backend validates the package.</p>';
    if(skill.native_custom_only)return '<p>Native custom-only tasks are available. Choose “Bundled custom grader only” in Advanced run options, or save JSON cases for another grading mode.</p>';
    return '<p class="field-hint">Save at least one reviewed test case before running an evaluation.</p>';
  }

  function renderRunCount(count) {
    if(usesNativeTasks())return '<span class="simple-only fine-print">Uses the bundled task package.</span><span class="advanced-only">Native task count resolved by the backend. Attempt and baseline settings apply.</span>';
    const attempts=Number(ui.evalOptions.n_attempts)||1,arms=ui.evalOptions.skip_baseline?1:2;
    return `<span class="simple-only fine-print">${arms===1?"No baseline comparison.":"Compares with and without the skill."}${attempts>1?` ${attempts} attempts per case and arm.`:""}</span><span class="advanced-only"><strong>${count * attempts * arms} planned attempts</strong> · ${count} cases × ${attempts} attempts × ${arms} ${arms===1?"arm":"arms"}<br><span class="fine-print">Runtime and provider cost depend on the tasks and model.</span></span>`;
  }

  function providerLabel(name) {return ({nv_build:"NVIDIA API Catalog",openai:"OpenAI",anthropic:"Anthropic"})[name] || "Model provider";}

  function badgeStatus(status) {
    const normalized=String(status || "unknown").toLowerCase();
    const kind=["pass","passed","scored","succeeded","valid"].includes(normalized)?"good":["fail","failed","invalid","error"].includes(normalized)?"bad":["running","stopping","pending","queued"].includes(normalized)?"running":"attention";
    const label=({succeeded:"Completed",failed:"Failed",scored:"Scored",unscored:"Unscored",stopping:"Cancellation requested",interrupted:"Interrupted",unknown:"Not assessed"})[normalized] || normalized[0]?.toUpperCase()+normalized.slice(1);
    return `<span class="badge ${kind}">${esc(label || "Not assessed")}</span>`;
  }

  function trialLabel(trial) {
    if (trial.status!=="scored") return trial.status || "Unknown";
    const failed=trial.checks?.some((check) => ["fail","failed","error"].includes(String(check.status).toLowerCase()));
    return failed ? "Scored · failed checks" : "Scored";
  }

  function caseAttention(c) {
    const trials=Object.values(c.arms || {}).flat();
    return !trials.length||trials.some((trial) => trial.status!=="scored" || trial.checks?.some((check) => ["fail","failed","error"].includes(String(check.status).toLowerCase())));
  }

  function renderResults(job) {
    const r=job.result,s=r.summary || {},cases=r.cases || [];
    ui.selectedCase=Math.min(Math.max(0,ui.selectedCase),Math.max(0,cases.length-1));
    const error=r.error || job.error,showError=error&&(selectedJob()?.id!==job.id||jobStage(job)!==ui.stage);
    const count=cases.filter(caseAttention).length,stopped=(job.status==="failed"||r.execution_status==="failed")&&!cases.some(c=>Object.values(c.arms || {}).flat().length);
    return `<section aria-label="Evaluation results">${showError?`<div class="banner error section-gap">${esc(typeof error==="string"?error:json(error))}</div>`:""}<div class="result-header"><div><p class="eyebrow">RETAINED EVALUATION RESULTS</p><h2>${stopped?"Evaluation stopped before results were available.":`${cases.length} ${cases.length===1?"case":"cases"} with retained evidence.`}</h2><p>${esc(r.execution_status || job.status)} · ${count} ${count===1?"case needs":"cases need"} a closer look · ${time(job.started_at)} · ${esc(s.scored_attempts ?? "Unknown")} / ${esc(s.expected_attempts ?? "unknown")} attempts scored</p><p class="fine-print">${esc(actionNames[job.action] || job.action)}</p></div><div class="button-row"><button class="button secondary small" data-action="download-result" data-id="${esc(job.id)}">Download report (.json)</button><a class="button secondary small" href="/api/jobs/${encodeURIComponent(job.id)}/download?file=output.log">Download log</a><button class="button secondary small" data-action="improve-from-run" data-id="${esc(job.id)}">Improve skill (optional)</button></div></div><div class="result-summary advanced-only"><div><small>SCORING COVERAGE</small><strong>${esc(s.scored_attempts ?? "Unknown")} / ${esc(s.expected_attempts ?? "unknown")} attempts scored</strong><p>Failed or unscored attempts remain visible.</p></div><div><small>LIFT UNDER THIS RUBRIC</small><strong>${finite(s.lift) ? `${s.lift>0?"+":""}${number(s.lift,4)}` : "Not available"}</strong><p>With skill ${esc(scalar(s.with_skill))} · baseline ${esc(scalar(s.without_skill))}</p></div><div><small>INTERPRETATION</small><strong>Read the case evidence.</strong><p>Lift alone does not establish a better task result or justify an edit.</p></div></div>${cases.length ? `<div class="result-grid"><aside class="surface result-cases" aria-label="Evaluation cases"><h3>Test cases</h3>${cases.map((c,i) => `<button class="case-row${i===ui.selectedCase?" active":""}${caseAttention(c)?" attention":""}" data-action="select-case" data-index="${i}"${i===ui.selectedCase?' aria-current="true"':""}><strong>${esc(c.id || `Unidentified case ${i+1}`)}</strong><p>${esc((c.question || "No task text retained.").slice(0,125))}${(c.question || "").length>125?"…":""}</p><span class="badge${caseAttention(c)?" attention":" good"}">${Object.values(c.arms || {}).flat().length?(caseAttention(c)?"Review evidence":"Scored evidence"):"No trial results"}</span></button>`).join("")}</aside><article class="surface case-detail" id="case-detail">${renderCaseDetail(cases[ui.selectedCase])}</article></div>` : '<div class="note attention">No case-level trials were normalized for this result. Inspect the retained report and logs. No missing outcome is counted as a pass.</div>'}${r.suggestions?.length ? `<section class="surface suggestions"><h3>Reported findings and suggestions</h3><p class="fine-print">These are the evaluator’s reported recommendations. They are not validated skill edits.</p>${r.suggestions.map((item)=>`<div class="suggestion-item">${esc(typeof item==="string"?item:json(item))}</div>`).join("")}</section>` : ""}${r.artifacts?.length ? `<div class="artifact-links advanced-only">${r.artifacts.map((a)=>`<a href="/api/jobs/${encodeURIComponent(job.id)}/download?file=${encodeURIComponent(a.path)}">${esc(a.name || a.path)}</a>`).join("")}</div>` : ""}<details class="section-gap advanced-only"><summary>Result provenance and complete normalized data</summary><div><p class="fine-print">Source ${esc(job.source_hash || "not recorded")}<br>Dataset ${esc(job.dataset_hash || "not recorded")}</p><pre class="raw-object">${esc(json(r))}</pre></div></details></section>`;
  }

  function renderCaseDetail(c) {
    if (!c) return "";
    const arms=c.arms || {}, withTrials=arms.with_skill || [], withoutTrials=arms.without_skill || [];
    const trials=arms[ui.arm] || [];
    ui.attempt=Math.min(Math.max(0,ui.attempt),Math.max(0,trials.length-1));
    const trial=trials[ui.attempt];
    return `<div class="case-detail-head"><h3>${esc(c.id || "Case identity not retained")}</h3>${trial ? badgeStatus(trial.status) : '<span class="badge attention">No retained attempt</span>'}</div><p class="case-question">${esc(c.question || "No task text retained.")}</p><div class="arm-tabs" role="group" aria-label="Comparison arm"><button class="${ui.arm==="with_skill"?"active":""}" data-action="select-arm" data-arm="with_skill" aria-pressed="${ui.arm==="with_skill"}">With skill · ${withTrials.length}</button><button class="${ui.arm==="without_skill"?"active":""}" data-action="select-arm" data-arm="without_skill" aria-pressed="${ui.arm==="without_skill"}">Baseline · ${withoutTrials.length}</button></div>${trials.length>1?`<div class="trial-picker"><label for="trial-picker">Attempt</label><select id="trial-picker">${trials.map((t,i)=>`<option value="${i}"${ui.attempt===i?" selected":""}>${i+1} · ${esc(trialLabel(t))}</option>`).join("")}</select></div>`:""}<div class="detail-tabs" role="group" aria-label="Evidence view">${[ ["checks","Checks"],["response","Response"],["trajectory","Evidence"] ].map(([id,label])=>`<button class="${ui.detail===id?"active":""}${id==="trajectory"?" advanced-only":""}" data-action="select-detail" data-detail="${id}" aria-pressed="${ui.detail===id}">${label}</button>`).join("")}</div>${trial ? renderTrial(trial) : `<div class="note attention">No ${ui.arm==="with_skill"?"skill-enabled":"baseline"} attempt is retained for this case. A missing result is not zero and is not a pass.</div>`}`;
  }

  function renderNativeEvidence(trial) {
    const job=selectedJob()?.result?.kind==="evaluation"?selectedJob():latestJob(ui.skillId,"evaluate");
    const group=job?.result?.improvement_signals?.agents?.[trial.agent];
    const c=group?.cases?.find(c=>c.entry_id===trial.case_id),attempts=c?.[trial.arm==="without_skill"?"baseline_attempts":"attempts"];
    const attempt=attempts?.find(a=>a.trial_id===(trial.physical_trial_id || trial.id));
    if(!attempt)return "";
    const findings=attempt.findings || [],execution=attempt.execution_evidence,artifacts=attempt.artifact_evidence;
    return `<details class="section-gap"><summary>Evidence for improving this case</summary><p class="fine-print">${esc(c.split || "discovery")} case · ${esc(attempt.skill_activation?.reason || "Skill use was not established.")}</p>${findings.map(f=>`<details><summary>${esc(f.label || f.metric || "Finding")}</summary><p>${esc((f.reasons || []).join(" "))}</p>${f.diagnostic?.text?`<pre class="raw-object">${esc(f.diagnostic.text)}</pre>${f.diagnostic.truncated?'<p class="fine-print">Diagnostic shortened; omissions are not evidence of success.</p>':""}`:""}</details>`).join("")}${execution?`<details><summary>Observed tool calls and results</summary><pre class="raw-object">${esc(execution.summary || execution.reason || "Execution evidence is unavailable.")}</pre></details>`:""}${artifacts?`<details><summary>Original and delivered files</summary><p class="fine-print">Only declared files are compared. Changed bytes do not establish a correct result.</p><pre class="raw-object">${esc(json(artifacts))}</pre></details>`:""}${attempt.missing_scores?.length?`<p class="note attention">Missing checks: ${esc(attempt.missing_scores.join(", "))}</p>`:""}</details>`;
  }

  function renderTrial(trial) {
    const id=trial.id ? `<p class="fine-print advanced-only">Trial <code>${esc(trial.id)}</code></p>` : '<p class="fine-print">Trial identity was not retained.</p>';
    const error=trial.error?`<div class="banner error">${esc(typeof trial.error==="string"?trial.error:json(trial.error))}</div>`:"";
    if(ui.detail==="response") return `${error}${trial.final_response ? `<pre class="response-view" tabindex="0" aria-label="Complete retained agent response">${esc(trial.final_response)}</pre>` : '<div class="note">No final response was retained for this attempt.</div>'}<div class="section-gap">${id}</div>`;
    if(ui.detail==="trajectory") return `${error}<p class="simple-only note">Switch to <button class="text-button" data-action="advanced-mode">Advanced</button> for detailed evidence.</p><p class="fine-print advanced-only">Retained evidence from this attempt. Complete trajectories may be available in the report artifacts below.</p>${trial.evidence?.length ? trial.evidence.map((e)=>`<pre class="raw-object advanced-only">${esc(typeof e==="string"?e:json(e))}</pre>`).join("") : '<div class="note section-gap">No structured trajectory evidence was provided by this result. Inspect the saved artifacts and logs.</div>'}<div class="section-gap">${id}</div>`;
    const checks=trial.checks || [],metrics=Object.entries(trial.metrics || {}).filter(([,v])=>finite(v)||v===null);
    return `${error}${trial.status!=="scored"?'<div class="note attention">This attempt was not scored. Missing values are not counted as success or failure.</div>':""}<p class="simple-only fine-print">${checks.length} reported checks · ${checks.filter(c=>["fail","failed","error"].includes(String(c.status).toLowerCase())).length} failed. <button class="text-button" data-action="advanced-mode">Show all details</button></p>${checks.length ? checks.map((check)=>`<div class="check-result${["pass","passed","valid","succeeded"].includes(String(check.status).toLowerCase())?" advanced-only":""}"><div class="check-result-head"><strong>${esc(check.name || "Unnamed check")}</strong>${badgeStatus(check.status)}</div>${check.reason?`<p>${esc(check.reason)}</p>`:""}${check.evidence?.length ? `<pre class="check-evidence advanced-only">${esc(check.evidence.map((e)=>typeof e==="string"?e:json(e)).join("\n\n"))}</pre>`:""}</div>`).join("") : '<div class="note">No requirement-level checks were provided. The reported metrics below do not establish exact output compliance on their own.</div>'}${metrics.length ? `<h4 class="section-gap advanced-only">Reported metrics</h4><dl class="metric-list advanced-only">${metrics.map(([name,value])=>`<div><dt>${esc(name.replaceAll("_"," "))}</dt><dd>${number(value)}</dd></div>`).join("")}</dl>`:""}${renderNativeEvidence(trial)}<div class="note section-gap advanced-only"><strong>Skill attribution</strong><br>No causal skill diagnosis or validated edit is supplied by these scores alone.</div><div class="section-gap">${id}</div>`;
  }

  function improvementContext() {
    const skill=selectedSkill(),original=skill?.proposal_job_id?ui.state.skills.find(s=>s.id===skill.parent_id):skill;
    return {skill,original,proposal:jobById(ui.improvement.proposalId===undefined?skill?.proposal_job_id:ui.improvement.proposalId)};
  }

  function evaluationRuns(skillId) {
    return (ui.state.jobs || []).filter(j=>j.skill_id===skillId&&j.action==="evaluate"&&j.status==="succeeded").sort((a,b)=>String(b.started_at).localeCompare(String(a.started_at)));
  }

  function runOptions(jobs,selected) {
    return jobs.map(job=>`<option value="${esc(job.id)}"${job.id===selected?" selected":""}>${esc(time(job.started_at))} · ${esc(job.options?.agent || "Evaluation")} · ${esc(job.id.slice(0,8))}</option>`).join("");
  }

  async function loadImprovement() {
    const skillId=ui.skillId,state=ui.improvement,{skill,original}=improvementContext();
    if(!original)return;
    const requestToken=Symbol();state.requestToken=requestToken;state.loading=true;state.error=null;renderAll();
    try {
      const proposals=(ui.state.jobs || []).filter(j=>j.skill_id===original.id&&j.action==="improve").sort((a,b)=>String(b.started_at).localeCompare(String(a.started_at)));
      if(state.proposalId===undefined)state.proposalId=state.discoveryId?(proposals.find(j=>j.discovery_job_id===state.discoveryId)?.id || null):(skill.proposal_job_id || proposals[0]?.id || null);
      if(state.proposalId)await fetchJob(state.proposalId);
      const proposal=jobById(state.proposalId);
      const shown=selectedJob();
      state.discoveryId ||= proposal?.discovery_job_id || (shown?.skill_id===original.id&&shown.action==="evaluate"?shown.id:null) || evaluationRuns(original.id)[0]?.id || null;
      const originalSelection=!state.beforeId;
      state.beforeId ||= state.discoveryId;
      if(state.discoveryId){
        await fetchJob(state.discoveryId);
        const discoveryId=state.discoveryId;
        const readiness=await api(`/api/jobs/${encodeURIComponent(discoveryId)}/improvement?skill_id=${encodeURIComponent(original.id)}`);
        if(state.requestToken!==requestToken||state.discoveryId!==discoveryId)return;
        state.readiness=readiness;
      } else state.readiness=null;
      if(proposal?.result?.candidate_id){
        const candidate=await api(`/api/skills/${encodeURIComponent(proposal.result.candidate_id)}`);
        if(state.requestToken!==requestToken)return;
        updateSkill(candidate);
        const validations=evaluationRuns(candidate.id);
        state.afterId ||= validations.find(j=>j.validation_of===proposal.id)?.id || validations[0]?.id || null;
        if(state.afterId)await fetchJob(state.afterId);
      }
      const comparison=(ui.state.jobs || []).filter(j=>j.action==="compare_improvement"&&j.proposal_job_id===state.proposalId).sort((a,b)=>String(b.started_at).localeCompare(String(a.started_at)))[0];
      if(comparison&&state.loadedProposalId!==state.proposalId){
        await fetchJob(comparison.id);
        if(state.requestToken!==requestToken)return;
        state.comparisonId=comparison.id;state.beforeId=comparison.before_job_id;state.afterId=comparison.after_job_id;
        state.outcomeMetrics=(comparison.outcome_metrics || []).join(", ");
      } else if(originalSelection&&state.afterId){
        const validation=jobById(state.afterId);state.beforeId=validation?.before_job_id || state.beforeId;
        state.outcomeMetrics=(validation?.outcome_metrics || []).join(", ");
      }
      if(proposal?.result)state.loadedProposalId=state.proposalId;
    } catch(error){if(state.requestToken===requestToken)state.error=error.message;}
    finally {if(state.requestToken===requestToken){state.loading=false;if(ui.skillId===skillId&&ui.improvement===state)renderAll();}}
  }

  function proposalNeedsModel() {
    const summary=ui.improvement.readiness?.signals_summary;
    return !summary||Boolean(summary.findings||(ui.improvement.observations&&summary.observations));
  }

  function renderImprove() {
    const {skill,original,proposal}=improvementContext(),state=ui.improvement;
    if(!original)return '<div class="surface empty-state"><h2>Add a skill first.</h2><button class="button primary" data-stage="skill">Add skill</button></div>';
    const receipt=proposal?.result?.receipt,view=proposal?.result?.proposal_view,changed=receipt?.changed_files || [];
    const runs=evaluationRuns(original.id),ready=state.readiness,working=(ui.state.jobs || []).some(j=>activeStatus(j.status));
    const validation=jobById(state.afterId),comparisonJob=jobById(state.comparisonId);
    const selectedMetrics=JSON.stringify((state.outcomeMetrics || "").split(",").map(s=>s.trim()).filter(Boolean).sort());
    const matches=job=>Boolean(job)&&job.before_job_id===state.beforeId&&JSON.stringify([...(job.outcome_metrics || [])].sort())===selectedMetrics;
    const comparisonResult=comparisonJob?.proposal_job_id===proposal?.id&&comparisonJob?.after_job_id===state.afterId&&matches(comparisonJob)?comparisonJob.result:validation?.validation_of===proposal?.id&&matches(validation)?validation.result?.improvement_comparison:null;
    const comparison=comparisonResult?.proposal_view;
    const summary=receipt?.proposal_summary || (receipt?.status==="unchanged"?"The available evidence did not support a skill change.":"Review the proposed edits and their supporting findings before testing.");
    let content;
    if(state.loading&&!receipt)content='<section class="surface panel" aria-busy="true"><h2>Checking the evaluation…</h2><p class="muted help-caption">Looking for evidence that can support a change.</p><div class="loading-placeholder" aria-hidden="true"></div></section>';
    else if(!runs.length&&!receipt)content='<section class="surface panel"><h2>Start with an evaluation.</h2><p class="muted help-caption">A proposal needs results from your skill’s test cases.</p><div class="form-actions"><button class="button primary" data-stage="evaluate">Evaluate skill →</button></div></section>';
    else if(!receipt)content=`<section class="surface panel"><p class="section-label">IMPROVE · OPTIONAL</p><h2>Let the results guide the next change.</h2><p class="muted help-caption">Review a proposed copy of your skill, then test it. Keeping the original is always an option.</p><label for="improve-discovery">Use this evaluation</label><select id="improve-discovery">${runOptions(runs,state.discoveryId)}</select>${ready&&!ready.eligible?`<div class="note attention section-gap"><strong>One thing to resolve first</strong><ul>${(ready.reasons || []).map(reason=>`<li>${esc(reason)}</li>`).join("")}</ul><div class="button-row"><button class="text-button" data-stage="cases">Review test cases</button><button class="text-button" data-stage="evaluate">Run a new evaluation</button></div></div>`:`<p class="field-hint">${ready?(proposalNeedsModel()?"The retained evidence is ready for a proposal.":"No selected feedback is available for an edit. Review this result without calling a model."):"Select an evaluation to check its evidence."}</p>`}<div class="form-actions"><span class="fine-print">${esc(providerLabel(ui.state.provider?.name))}</span><button class="button primary" data-action="propose-change"${!ready?.eligible||working?" disabled":""}>${!proposalNeedsModel()?"Review without edits":ui.state.provider?.configured?"Propose changes":"Connect a model"} →</button></div></section>`;
    else if(receipt.status==="unchanged")content=`<section class="surface panel"><p class="section-label">REVIEW COMPLETE</p><h2>No change recommended.</h2><p class="muted help-caption">Keep the original. A proposal does not have to make an edit.</p>${renderProposalSummary(summary)}<div class="form-actions"><button class="button primary" data-action="original-results">Back to results</button></div></section>`;
    else if(receipt.status==="exported")content='<section class="surface panel"><h2>Evidence is ready to review.</h2><p class="muted help-caption">This export contains findings and editing instructions. It has not changed or tested the skill.</p><div class="form-actions"><button class="button primary" data-action="new-proposal">Propose a change →</button></div></section>';
    else if(receipt.status==="failed"||!receipt.candidate_ready)content=`<section class="surface panel"><h2>A proposal could not be prepared.</h2><p class="muted help-caption">${esc(receipt.error || proposal?.error || "Review the details below, then try again.")}</p><div class="form-actions"><button class="button primary" data-action="new-proposal">Review evaluation</button></div></section>`;
    else content=`<section class="surface proposal-review"><div class="proposal-heading"><div><p class="section-label">PROPOSED COPY</p><h2>${changed.length} ${changed.length===1?"file":"files"} to review.</h2><p class="muted">Your original skill is kept.</p></div><span class="badge attention">${comparison?"Test results below":"Not yet validated"}</span></div><div class="change-story"><section><h3>Why this change</h3>${renderProposalSummary(summary)}</section><section><h3>What changes</h3><p class="fine-print">${changed.map(name=>`<code>${esc(name)}</code>`).join(" · ")}</p><details class="change-diff"><summary>Review file changes</summary><pre tabindex="0" aria-label="Proposed file changes">${String(view?.diff || "No file diff was retained.").split("\n").map(line=>`<span class="${line.startsWith("+")&&!line.startsWith("+++")?"diff-add":line.startsWith("-")&&!line.startsWith("---")?"diff-remove":""}">${esc(line)}\n</span>`).join("")}</pre></details>${renderProposalEvidence(view)}</section><section><h3>What the tests show</h3>${view?.issues?.length?`<div class="note attention"><strong>Proposal evidence needs review</strong><ul>${view.issues.map(issue=>`<li>${esc(issue)}</li>`).join("")}</ul></div>`:""}${comparison?renderImprovementOutcome(comparison):comparisonResult?.error?`<p class="banner error" role="alert">Comparison could not complete: ${esc(comparisonResult.error)}</p>`:`<p class="muted">${validation?.status==="succeeded"?"Compare the selected runs to review the result.":"The proposed copy still needs testing against the original."}</p>`}${!comparison?`<div class="form-actions"><span class="fine-print">Uses the selected original run’s cases and settings.</span><button class="button primary" data-action="${validation?.status==="succeeded"?"compare-proposal":"validate-proposal"}"${working||!proposal.result.candidate_id?" disabled":""}>${working?"A task is running…":validation?.status==="succeeded"?"Compare results →":"Test proposed copy →"}</button></div>`:`<div class="button-row section-gap"><a class="button secondary" href="/api/skills/${encodeURIComponent(proposal.result.candidate_id)}/export">Download proposed skill (.zip)</a><button class="text-button" data-action="original-results">Keep original</button></div>`}</section></div></section>`;
    const manual=skill.proposal_job_id?"":renderManualEditor(skill);
    return `<div class="improve-flow">${content}${state.error?`<p class="banner error section-gap" role="alert">${esc(state.error)}</p>`:""}<div id="job-region"></div>${runs.length?renderImprovementAdvanced(original,proposal):""}${manual}</div>`;
  }

  function renderProposalSummary(summary) {
    return summary.length>240?`<p class="proposal-summary">${esc(summary.slice(0,240))}…</p><details><summary>Read full explanation</summary><p class="proposal-summary">${esc(summary)}</p></details>`:`<p class="proposal-summary">${esc(summary)}</p>`;
  }

  function renderProposalEvidence(view) {
    if(!view?.edits?.length)return "";
    return `<details class="section-gap"><summary>Evidence behind these edits</summary>${view.edits.map(edit=>`<div class="evidence-item"><strong>${esc(edit.path)}</strong><p>${esc(edit.reason || edit.explanation || "See the linked findings.")}</p>${(edit.evidence_ids || []).map(id=>{const finding=view.findings?.[id];return `<details><summary>${esc(finding?.case || "Finding")} · ${esc(finding?.label || finding?.metric || finding?.id || id)}</summary><p>${esc(finding?.kind==="agent_observation"?`Unverified agent report: ${finding.text || "No report text retained."}`:finding?.reasons?.join(" ") || finding?.reason || finding?.message || "No short explanation retained.")}</p>${finding?.kind==="agent_observation"&&finding.truncated?'<p class="fine-print">This unverified report was shortened.</p>':""}${finding?.diagnostic?.text?`<pre class="raw-object">${esc(finding.diagnostic.text)}</pre>${finding.diagnostic.truncated?'<p class="fine-print">This diagnostic was shortened.</p>':""}`:""}</details>`;}).join("")}</div>`).join("")}</details>`;
  }

  function renderImprovementOutcome(view) {
    const comparison=view.comparison,verified=view.association==="Verified runtime source match"&&!view.issues?.length;
    const labels={outcome_improvement:["Observed outcome gain","An explicitly selected outcome check improved in these tests."],signal_improvement:["Scores increased","A better task outcome has not been established."],regression:["Some results got worse","Review the declines before deciding whether to use this copy."],no_demonstrated_benefit:["No demonstrated benefit","These tests did not establish a gain from the proposed copy."],abstain:["Not enough comparable evidence","Resolve the issues below before drawing a conclusion."]};
    const [title,description]=!verified?["Results are not verified for this change","The saved runs could not both be matched to this proposal."]:labels[comparison?.verdict] || ["Not evaluated","Test the proposed copy to compare results."];
    const good=verified&&comparison?.verdict==="outcome_improvement",issues=[...(view.issues || []),...(comparison?.issues || [])];
    const cases=Object.entries(comparison?.agents || {}).flatMap(([agent,data])=>(data.cases || []).map(c=>({...c,agent})));
    const holdout=cases.length&&cases.every(c=>["holdout","heldout","held-out","validation","val","test","selection"].includes(c.split));
    return `<div class="comparison-verdict ${good?"positive":""}"><strong>${esc(title)}</strong><p>${esc(description)}</p></div>${issues.length?`<ul class="issue-list">${[...new Set(issues)].map(issue=>`<li>${esc(issue)}</li>`).join("")}</ul>`:""}${cases.length?`<p class="fine-print comparison-scope">${holdout?"Reserved validation cases":"Includes cases available during discovery"} · ${cases.length} ${cases.length===1?"case":"cases"}. These observations do not establish a general improvement.</p><div class="comparison-cases">${cases.map(c=>`<details><summary><span>${esc(c.entry_id)}</span><span class="badge ${c.status==="regressed"?"bad":""}">${esc(c.status || "Not assessed")}</span></summary><p class="fine-print">${esc(c.agent)} · ${esc(c.attempts)} attempts · ${esc(c.split || "discovery")}</p><div class="table-scroll"><table><caption class="sr-only">Original and proposed scores for ${esc(c.entry_id)}</caption><thead><tr><th scope="col">Check</th><th scope="col">Original</th><th scope="col">Proposed</th></tr></thead><tbody>${Object.entries(c.metrics || {}).map(([name,m])=>`<tr><th scope="row">${esc(name.replaceAll("_"," "))}${(comparison.outcome_metrics || []).includes(name)?' <span class="fine-print">(outcome)</span>':""}</th><td>${esc(scalar(m.before))}</td><td>${esc(scalar(m.after))}${m.delta<0?' <span class="decline">↓</span>':""}</td></tr>`).join("")}</tbody></table></div></details>`).join("")}</div>`:""}<details class="advanced-only section-gap"><summary>Source checks and full comparison</summary><p>${esc(view.association)}</p><pre class="raw-object">${esc(json(comparison))}</pre></details>`;
  }

  function renderImprovementAdvanced(original,proposal) {
    const state=ui.improvement,candidateId=proposal?.result?.candidate_id;
    return `<details id="improve-settings" class="surface panel section-gap advanced-only"${state.advancedOpen?" open":""}><summary>Proposal and validation settings</summary><div class="section-gap"><label class="check-label"><input id="improve-observations" type="checkbox"${state.observations?" checked":""}>Include agent observations</label><p class="field-hint">Agent reports remain unverified. Only eligible discovery evidence reaches the proposer; scanning and repeated-content checks are separate.</p><p class="fine-print">Editor: ${esc(providerLabel(ui.state.provider?.name))} · ${esc(ui.state.provider?.model)} <button class="text-button" data-action="settings">Change model</button></p><div class="button-row"><button class="button secondary small" data-action="export-evidence"${!state.discoveryId?" disabled":""}>Export evidence only</button>${proposal?'<button class="text-button" data-action="new-proposal">Start another proposal</button>':""}</div><label for="improve-edits">Import reviewed edits (.json)</label><input id="improve-edits" type="file" accept=".json,application/json"><p class="field-hint">Uses the native edit format with exact replacements and finding references. No model call.</p>${proposal?.result?`<button class="text-button" data-action="download-result" data-id="${esc(proposal.id)}">Download proposal report (.json)</button>`:""}${candidateId?`<div class="section-gap"><label for="improve-before">Original validation run</label><select id="improve-before">${runOptions(evaluationRuns(original.id),state.beforeId)}</select><label for="improve-after">Proposed-copy validation run</label><select id="improve-after"><option value="">Choose a completed run</option>${runOptions(evaluationRuns(candidateId),state.afterId)}</select><label for="improve-outcomes">Independent outcome checks <span class="optional">optional, comma-separated</span></label><input id="improve-outcomes" value="${esc(state.outcomeMetrics || "")}" placeholder="For example: helper_contract"><p class="field-hint">Name independently defined grader checks. Leaving this blank reports score gains only.</p><div class="button-row"><button class="button secondary" data-action="compare-proposal"${!state.afterId?" disabled":""}>Compare selected runs</button><button class="text-button" data-action="validate-proposal">Test again</button></div></div>`:""}</div></details>`;
  }

  function renderManualEditor(skill) {
    const content=ui.candidateContent ?? skill.content;
    return `<details class="surface panel section-gap advanced-only"><summary>Edit SKILL.md yourself</summary><form id="candidate-form"><label for="candidate-content">Candidate SKILL.md</label><textarea id="candidate-content" class="code-input" rows="12" required>${esc(content)}</textarea><p class="field-hint">Creates a separate copy with the saved assets and test cases.</p><div class="form-actions"><button class="button secondary" type="submit">Create manual copy</button></div></form></details>${skill.parent_id&&!skill.proposal_job_id?renderComparison(ui.state.skills.find(s=>s.id===skill.parent_id),skill):""}`;
  }

  async function proposeChange(mode="llm",edits) {
    const {original}=improvementContext(),state=ui.improvement;
    if(mode==="llm"&&proposalNeedsModel()&&!ui.state.provider?.configured){openSettings();return;}
    const job=await api("/api/jobs",{method:"POST",body:{skill_id:original.id,action:"improve",discovery_job_id:state.discoveryId,improvement:{mode,include_agent_observations:Boolean(state.observations),...(edits?{edits}:{})}}});
    state.proposalId=job.id;state.comparisonId=null;state.afterId=null;state.error=null;
    ui.state.jobs.push(job);ui.jobId=job.id;renderAll();schedulePoll();
  }

  async function validateProposal() {
    const {proposal}=improvementContext(),state=ui.improvement,selectedAtStart=ui.skillId;
    if(!proposal?.result?.candidate_id)throw new Error("Prepare a changed candidate first.");
    const before=await fetchJob(state.beforeId || state.discoveryId);
    const candidate=await api(`/api/skills/${encodeURIComponent(proposal.result.candidate_id)}`);updateSkill(candidate);
    if(ui.skillId!==selectedAtStart||ui.improvement!==state)throw new Error("The selected skill changed. Start the test again from the proposal you want.");
    if(hasUnsavedEdits())throw new Error("Save your edits before testing a proposed copy.");
    const job=await startJob("evaluate",before.options || {},null,{skill_id:candidate.id,validation_of:proposal.id,before_job_id:before.id,outcome_metrics:(state.outcomeMetrics || "").split(",").map(s=>s.trim()).filter(Boolean)});
    state.afterId=job.id;state.comparisonId=null;renderAll();
  }

  async function compareProposal() {
    const {proposal}=improvementContext(),state=ui.improvement;
    const job=await api("/api/jobs",{method:"POST",body:{skill_id:proposal.result.candidate_id,action:"compare_improvement",proposal_job_id:proposal.id,before_job_id:state.beforeId,after_job_id:state.afterId,outcome_metrics:(state.outcomeMetrics || "").split(",").map(s=>s.trim()).filter(Boolean)}});
    ui.state.jobs.push(job);ui.jobId=job.id;state.comparisonId=job.id;renderAll();schedulePoll();
  }

  function renderComparison(original,candidate) {
    const a=latestJob(original?.id,"evaluate"),b=latestJob(candidate.id,"evaluate");
    const eligible=(job,skill)=>job?.status==="succeeded"&&job.result?.kind==="evaluation"&&job.result.execution_status==="succeeded"&&Boolean(job.source_hash)&&job.source_hash===skill?.source_hash&&job.dataset_hash===skill?.dataset_hash;
    const matching=eligible(a,original)&&eligible(b,candidate)&&Boolean(a.dataset_hash)&&Boolean(b.dataset_hash)&&a.dataset_hash===b.dataset_hash&&Boolean(a.provider)&&Boolean(b.provider)&&json(a.provider)===json(b.provider)&&json(a.options || {})===json(b.options || {});
    return `<section class="surface comparison"><h3>Original and candidate runs</h3><p class="fine-print help-caption">Actual retained results are shown below. These numbers are descriptive; this interface does not issue an optimization acceptance verdict.</p><div class="comparison-grid">${[[original,a,"Original"],[candidate,b,"Candidate"]].map(([skill,job,label])=>`<div class="comparison-cell"><h4>${label}</h4><span class="badge${eligible(job,skill)?" good":" attention"}">${eligible(job,skill)?"Evaluation retained":job?.result?.kind==="evaluation"?"Older or unconfirmed snapshot":"Not evaluated successfully"}</span><p>${esc(skill?.name || "Original not found")}</p><div class="large-value section-gap">${eligible(job,skill)?esc(scalar(job.result.summary?.with_skill)):"—"}</div><p>${eligible(job,skill)?`${esc(job.result.summary?.scored_attempts ?? "Unknown")} scored attempts · ${time(job.started_at)}`:"No successful comparison evidence yet."}</p>${job?`<button class="text-button" data-action="open-job" data-id="${esc(job.id)}">Inspect run →</button>`:""}</div>`).join("")}</div><div class="note${matching?"":" attention"} section-gap">${matching?"Source and dataset fingerprints, provider, and recorded options match. Review per-case outcomes and regressions before deciding whether to keep the candidate.":"A fair comparison needs current source snapshots, the same saved dataset, provider, and settings. No improvement claim is made here."}</div><div class="button-row section-gap"><button class="button primary" data-stage="evaluate">Evaluate this candidate →</button>${original?`<button class="button secondary" data-action="open-candidate" data-id="${esc(original.id)}">Return to original</button>`:""}</div></section>`;
  }

  function renderHistory() {
    const jobs=[...(ui.state.jobs || [])].sort((a,b)=>String(b.started_at || "").localeCompare(String(a.started_at || "")));
    $("#runs-view").innerHTML=`<div class="page-heading"><div><p class="eyebrow">RETAINED LOCAL WORK</p><h1>Every run has a record.</h1><p class="lede">Drafts, checks, and evaluations stay connected to the skill copy that ran.</p></div><button class="button secondary" data-action="refresh">Refresh</button></div>${jobs.length?`<div class="run-history">${jobs.map((job)=>`<article class="surface run-history-item"><div><h3>${esc(actionNames[job.action] || job.action)} · ${esc(ui.state.skills.find((s)=>s.id===job.skill_id)?.name || "Unknown skill")}</h3><p>${time(job.started_at)} · ${esc(job.id)}</p><div class="pill-row">${badgeStatus(job.status)}${job.result?.optional_checks?badgeStatus(job.result.optional_checks.status):job.result?.kind==="check"?`<span class="badge${job.result.passed?" good":" attention"}">${job.result.passed?"Quality checks passed":"Quality findings reported"}</span>`:""}${job.result?.kind==="validation"?`<span class="badge${job.result.valid?" good":" bad"}">${job.result.valid?"Dataset valid":"Validation issues"}</span>`:""}</div></div><div class="button-row"><button class="button secondary small" data-action="open-job" data-id="${esc(job.id)}">Open run</button>${activeStatus(job.status)?`<button class="button danger small" data-action="cancel-job" data-id="${esc(job.id)}"${job.status==="stopping"?" disabled":""}>Cancel</button>`:""}${job.result?`<button class="button secondary small" data-action="download-result" data-id="${esc(job.id)}">Download report (.json)</button>`:""}</div></article>`).join("")}</div>`:'<div class="surface empty-state"><h2>No runs yet.</h2><p>Load a skill, draft a test set, or start with local quality checks. Your actual runs will appear here.</p><button class="button primary" data-view="workspace">Open my skills</button></div>'}`;
  }

  function renderOptionalChecks(report) {
    const results=report.results || [],findings=[...(report.findings || results.flatMap(result=>(result.findings || []).map(finding=>({...finding,validator:result.validator}))))],severity={critical:0,high:1,medium:2,low:3,info:4};
    findings.sort((a,b)=>(severity[a.severity] ?? 5)-(severity[b.severity] ?? 5));
    const isFinding=value=>findings.some(finding=>finding.message&&String(typeof value==="string"?value:value.message).includes(finding.message));
    const coverage=result=>(result.success_details || []).filter(detail=>detail.check==="context_dedup");
    return `<section class="optional-report section-gap" aria-label="Optional check results"><div class="job-head"><h3>Optional checks</h3>${badgeStatus(report.status)}</div><p class="fine-print">${results.length} validator results · ${results.filter(result=>result.status==="failed").length} failed · ${results.filter(result=>result.status==="incomplete").length} incomplete. Separate from agent scores and lift.</p>${results.map(result=>`<div class="optional-result${result.status==="passed"&&!coverage(result).length&&![...(result.errors || []),...(result.warnings || []),...(result.gate_failures || [])].some(message=>!isFinding(message))?" advanced-only":""}"><div class="check-result-head"><strong>${esc(result.validator || result.description || "Check")}</strong>${badgeStatus(result.status)}</div>${coverage(result).map(detail=>`<p class="note">${esc(detail.message)}</p>`).join("")}${result.incomplete_scans?.length?`<p class="note attention">Not completed: ${esc(result.incomplete_scans.join(", "))}</p>`:""}${[...new Set([...(result.errors || []),...(result.gate_failures || [])])].filter(error=>!isFinding(error)).map(error=>`<p class="banner error">${esc(typeof error==="string"?error:error.message || json(error))}</p>`).join("")}${(result.warnings || []).filter(warning=>!isFinding(warning)).map(warning=>`<p class="note attention">${esc(typeof warning==="string"?warning:warning.message || json(warning))}</p>`).join("")}</div>`).join("")}${findings.length?`<p class="fine-print section-gap">${findings.length} reported ${findings.length===1?"finding":"findings"}</p><ul class="issue-list">${findings.map((finding,i)=>`<li class="${i>4?"advanced-only":""}"><strong>${esc(finding.severity || finding.category || "Finding")}</strong>: ${esc(finding.message || finding.check_name || "No explanation retained.")}${finding.suggestion?`<p class="finding-suggestion">${esc(finding.suggestion)}</p>`:""}${finding.file_path?`<p class="fine-print advanced-only">${esc(finding.file_path)}${finding.line_number?`:${esc(finding.line_number)}`:""}</p>`:""}</li>`).join("")}</ul>${findings.length>5?'<button class="text-button simple-only" data-action="advanced-mode">Show all findings</button>':""}`:results.some(result=>coverage(result).length)?"":`<p class="note${report.status==="passed"?"":" attention"} section-gap">${report.status==="passed"?"No findings reported by the selected checks.":"No findings retained; the selected checks did not all complete successfully."}</p>`}<details class="advanced-only section-gap"><summary>Complete check details</summary><pre class="raw-object">${esc(json(report))}</pre></details></section>`;
  }

  function renderQuality(result) {
    const q=result?.quality;
    if(!q)return "";
    const dimensions=Object.entries(q.dimensions || {}),issues=dimensions.flatMap(([,d])=>d.issues || []);
    return `${!result.issues?.length&&issues.length?`<ul class="issue-list simple-only">${issues.slice(0,3).map(issue=>`<li>${esc(issue.message)}${issue.suggestion?` ${esc(issue.suggestion)}`:""}</li>`).join("")}</ul><button class="text-button simple-only" data-action="advanced-mode">All quality findings</button>`:""}<div class="quality-overview section-gap advanced-only"><div class="quality-score"><span class="config-label">QUALITY SCORE</span><strong>${finite(q.overall_score)?number(q.overall_score,1):"Not available"}<small>${finite(q.overall_score)?" / 100":""}</small></strong><span class="badge">${esc(q.grade || "Grade not retained")}</span></div><div class="quality-dimensions">${dimensions.map(([name,d])=>`<div><span>${esc(name.replaceAll("_"," "))}</span><strong>${finite(d.score)?number(d.score,1):"Not available"}</strong></div>`).join("")}</div></div>${result.scope?`<p class="fine-print section-gap">${esc(result.scope)}</p>`:""}<details class="section-gap advanced-only"><summary>Quality dimensions and specific findings</summary><div>${dimensions.map(([name,d])=>`<div class="quality-dimension"><h4>${esc(name.replaceAll("_"," "))}</h4><p class="fine-print">Score ${finite(d.score)?number(d.score,1):"not available"} · weight ${finite(d.weight)?number(d.weight,2):"not recorded"}</p>${d.issues?.length?`<ul class="issue-list">${d.issues.map((issue)=>`<li><strong>${esc(issue.severity || "Finding")}</strong>: ${esc(issue.message || "No explanation retained.")}${issue.suggestion?`<br>${esc(issue.suggestion)}`:""}</li>`).join("")}</ul>`:'<p class="fine-print">No dimension findings reported.</p>'}</div>`).join("")}</div></details>`;
  }

  function rememberJobLog() {
    const details=$("#job-region .job-progress"),log=details?.querySelector(".job-log");
    if(log)ui.logViews[details.dataset.jobId]={open:details.open,scrollTop:log.scrollTop,nearBottom:log.scrollHeight-log.scrollTop-log.clientHeight<30};
  }

  function renderJobRegion() {
    rememberJobLog();
    const region=$("#job-region");
    if(!region)return;
    const job=selectedJob();
    if(!job||jobStage(job)!==ui.stage||(job.skill_id!==ui.skillId&&!(ui.stage==="improve"&&job.proposal_job_id===ui.improvement.proposalId))){region.innerHTML="";return;}
    const view=ui.logViews[job.id],error=job.result?.error || job.error;
    const detailsOpen=view?.open ?? (activeStatus(job.status)||job.status==="failed");
    const summarized=ui.stage==="improve"&&job.status==="succeeded"&&(["improvement","improvement_comparison"].includes(job.result?.kind)||job.result?.improvement_comparison);
    const quality=job.result?.optional_checks?null:job.result?.kind==="check"?job.result.passed:job.result?.kind==="validation"?job.result.valid:null;
    const fallback=job.result?.kind==="dataset"&&job.result.generation_mode==="template"&&job.options?.no_llm===false;
    region.innerHTML=`<section class="surface job-surface${summarized||(job.action==="draft"&&job.status==="succeeded"&&!fallback&&ui.pendingDraft!==job.id)?" compact-job":""}" aria-label="Current job"><div class="job-head"><h3>${activeStatus(job.status)?'<span class="spinner" aria-hidden="true"></span> ':""}${esc(actionNames[job.action] || job.action)}</h3><div class="button-row">${badgeStatus(job.action==="check"&&job.status==="succeeded"&&job.result?.optional_checks?job.result.optional_checks.status:job.status)}${activeStatus(job.status)?`<button class="button danger small" data-action="cancel-job" data-id="${esc(job.id)}"${job.status==="stopping"?" disabled":""}>${job.status==="stopping"?"Stopping…":"Cancel"}</button>`:""}</div></div><div class="job-meta advanced-only"><span>${time(job.started_at)}</span><span>${esc(job.id)}</span></div>${error?`<div class="banner error section-gap">${esc(typeof error==="string"?error:json(error))}</div>`:""}${job.result?.optional_checks?renderOptionalChecks(job.result.optional_checks):""}${quality!==null?`<div class="note ${quality?"success":"attention"} section-gap"><strong>${job.result.kind==="validation"?(quality?"Dataset validation passed.":"Dataset has validation issues."):(quality?"Skill quality checks passed.":"Skill checks completed with findings.")}</strong> ${quality?"Inspect the retained details before an agent run.":"Process completion does not mean the quality checks passed."}</div>${renderQuality(job.result)}${job.result.issues?.length?`<ul class="issue-list${quality?" advanced-only":""}">${job.result.issues.map((issue,i)=>`<li class="${quality||i>2?"advanced-only":""}">${esc(typeof issue==="string"?issue:issue.message || json(issue))}</li>`).join("")}</ul>${!quality&&job.result.issues.length>3?`<button class="text-button simple-only" data-action="advanced-mode">View all ${job.result.issues.length} findings</button>`:""}`:""}`:""}${job.result?.kind==="dataset"?`<div class="note${fallback?" attention":""} section-gap"><strong>${fallback?"Model drafting failed; showing starter cases.":"Draft created · review required."}</strong> ${fallback?"See the logs below for details.":job.result.generation_mode==="template"?"Starter template cases; review before saving.":"Model-assisted cases; review before saving."}${ui.pendingDraft===job.id?'<div class="button-row section-gap"><button class="button secondary small" data-action="review-ready-draft">Review new draft</button></div>':""}</div>`:""}<details class="job-details job-progress section-gap" data-job-id="${esc(job.id)}"${detailsOpen?" open":""}><summary>Logs and progress</summary><div><pre class="job-log" tabindex="0" aria-label="Job logs">${esc((job.logs || []).join("\n") || "Waiting for worker output…")}</pre><a class="button secondary small" href="/api/jobs/${encodeURIComponent(job.id)}/download?file=output.log">Download log</a>${job.events?.length?`<ul class="events-list">${job.events.slice(-5).map((event)=>`<li><span aria-hidden="true">·</span><span>${esc(event.message || event.detail || event.type || event.event || json(event))}</span></li>`).join("")}</ul>`:""}</div></details>${job.result&&job.result.kind!=="evaluation"?`<details class="job-details advanced-only"><summary>Complete result</summary><div><pre class="raw-object">${esc(json(job.result))}</pre><button class="button secondary small" data-action="download-result" data-id="${esc(job.id)}">Download report (.json)</button></div></details>`:""}</section>`;
    const nextLog=region.querySelector(".job-log");
    if(nextLog)nextLog.scrollTop=!view||view.nearBottom?nextLog.scrollHeight:view.scrollTop;
  }

  async function startJob(action,options={},button,extra={}) {
    const skill=extra.skill_id?ui.state.skills.find(s=>s.id===extra.skill_id):selectedSkill();
    const revisionAtSubmit=ui.datasetDirty?null:ui.caseRevision,selectedAtSubmit=ui.skillId;
    if(!skill)throw new Error("Add a skill first.");
    if(action==="evaluate"){
      if(!skill.cases?.length&&!usesNativeTasks(skill,options)){setStage("cases");throw new Error(skill.native_custom_only?"Choose custom-only grading for the native task package, or save reviewed JSON cases.":"Save at least one reviewed case before evaluating.");}
      if(!ui.state.capabilities?.tier3)throw new Error("Agent evaluation needs the Tier 3 optional dependencies. You can still validate the dataset and run skill checks.");
      if(!capability(ui.state.capabilities?.[options.env_mode]))throw new Error(ui.state.capabilities?.[options.env_mode]?.reason || "The selected execution environment is unavailable.");
      if(!ui.state.provider?.configured){openSettings();throw new Error("Connect a provider before starting an agent evaluation.");}
    }
    if(options.dedup&&ui.state.provider?.name==="anthropic"){openSettings();throw new Error("Repeated-content checks need NVIDIA or OpenAI embeddings. Choose one in model settings.");}
    if(action==="check"&&(options.tier1_llm||options.dedup)&&!ui.state.provider?.configured){openSettings();throw new Error("Connect a model for AI security review or repeated-content checks. The standard skill scan works without a key.");}
    if(action==="draft"&&!options.no_llm&&!ui.state.provider?.configured){openSettings();throw new Error("Connect a provider for a model-assisted draft, or choose Template.");}
    const job=await api("/api/jobs",{method:"POST",body:{skill_id:skill.id,action,guidance:action==="draft"?ui.guidance:undefined,options,...extra}});
    ui.state.jobs.push(job);
    if(ui.skillId!==selectedAtSubmit){renderChrome();schedulePoll();return job;}
    if(extra.validation_of){ui.skillId=skill.id;ui.draft=null;ui.candidateContent=null;}
    ui.jobId=job.id;
    if(action==="draft")ui.draftStartRevision=revisionAtSubmit;
    $("#announcer").textContent=`${actionNames[action] || action} started.`;
    if(action==="draft")ui.stage="cases";
    if(action==="evaluate")ui.stage=extra.validation_of?"improve":"evaluate";
    renderAll();schedulePoll();
    return job;
  }

  function schedulePoll() {
    clearTimeout(ui.pollTimer);
    if((ui.state?.jobs || []).some((j)=>activeStatus(j.status))) ui.pollTimer=setTimeout(pollJobs,1500);
  }

  async function pollJobs() {
    const jobs=(ui.state?.jobs || []).filter((job)=>activeStatus(job.status));
    if(!jobs.length)return;
    const updates=await Promise.allSettled(jobs.map((job)=>api(`/api/jobs/${encodeURIComponent(job.id)}`)));
    let finished=false;
    updates.forEach((result,i)=>{
      if(result.status!=="fulfilled"){connection(false,result.reason.message);return;}
      const job=result.value,index=ui.state.jobs.findIndex((j)=>j.id===job.id);
      ui.state.jobs[index]=job;
      connection(true);
      if(!activeStatus(job.status)){
        finished=true;
        $("#announcer").textContent=`${actionNames[job.action] || job.action} ${job.status}.`;
        if(job.action==="draft"&&job.status==="succeeded"&&job.result?.kind==="dataset"&&job.skill_id===ui.skillId&&job.id===ui.jobId){
          if(ui.draftStartRevision===ui.caseRevision)applyDraft(job);else{ui.pendingDraft=job.id;notify("Your new draft is ready. Your case edits are safe; choose Review new draft when you’re ready.");}
        }
      }
    });
    renderChrome();
    if(ui.view==="runs")renderHistory();
    else if(finished){if(ui.stage==="improve")await loadImprovement();else renderAll();}
    else renderJobRegion();
    schedulePoll();
  }

  async function saveCases() {
    const skill=selectedSkill(),records=draftRecords(),revision=ui.caseRevision;
    const cases=records.filter(record=>record.selected).map(record=>clone(record.value));
    if(records.length&&!cases.length)throw new Error("Include at least one case, or use Remove to delete all cases.");
    if(cases.some(c=>!String(c.prompt ?? c.question ?? "").trim()))throw new Error("Each included case needs a task.");
    const updated=await api(`/api/skills/${encodeURIComponent(skill.id)}/dataset`,{method:"PUT",body:{cases}});
    updateSkill(updated);
    if(ui.skillId===skill.id&&ui.draft===records&&ui.caseRevision===revision){ui.datasetDirty=false;ui.caseRevision++;}
    notify(ui.skillId===skill.id&&ui.datasetDirty?"Saved the earlier snapshot. Your newer edits are still unsaved.":cases.length?`${cases.length} test ${cases.length===1?"case":"cases"} saved.`:"All test cases removed from the saved dataset.");
    return updated;
  }

  function openDialog(dialog) {
    ui.returnFocus=document.activeElement;
    if(!dialog.open)dialog.showModal();
  }
  function closeDialog(dialog) {
    if(dialog?.open)dialog.close();
    const key=$("#provider-key");if(key)key.value="";
    if(ui.returnFocus?.isConnected)ui.returnFocus.focus();
  }
  function openSettings() {
    const p=ui.state?.provider || {};
    $("#provider-name").value=["nv_build","openai","anthropic"].includes(p.name)?p.name:"nv_build";
    $("#provider-model").value=p.model || "";
    $("#provider-key").value="";
    $("#provider-details").open=document.documentElement.dataset.mode==="advanced";
    $("#settings-title").textContent=`Connect ${providerLabel($("#provider-name").value)}.`;
    $("#provider-key-label").textContent=`${$("#provider-name").value==="nv_build"?"NVIDIA":providerLabel($("#provider-name").value)} API key`;
    $("#settings-dialog .provider-link").hidden=$("#provider-name").value!=="nv_build";
    $("#provider-state").textContent=p.configured?"A provider connection is already configured.":"Template drafts and local checks work without a key.";
    $("#settings-error").hidden=true;
    openDialog($("#settings-dialog"));
  }

  async function importSkill(file) {
    if(!file)return;
    if(!/\.(md|zip)$/i.test(file.name))throw new Error("Choose a Markdown skill document or a .zip skill package.");
    if(file.size>12*1024*1024)throw new Error("Choose a skill package smaller than 12 MB.");
    if(!allowSkillSwitch())return;
    const bytes=new Uint8Array(await file.arrayBuffer());
    let binary="";
    for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));
    const skill=await api("/api/skills",{method:"POST",body:{filename:file.name,archive_base64:btoa(binary)}});
    updateSkill(skill);selectSkill(skill.id,false);ui.stage="cases";renderAll();notify(`Added ${skill.name} as a local copy. Review or draft its test cases.`);
  }

  function downloadResult(job) {
    if(!job?.result)throw new Error("This run does not have a result to download yet.");
    const blob=new Blob([json({job_id:job.id,skill_id:job.skill_id,source_hash:job.source_hash,dataset_hash:job.dataset_hash,status:job.status,result:job.result})],{type:"application/json"});
    const url=URL.createObjectURL(blob),link=document.createElement("a");
    link.href=url;link.download=`skill-evaluator-${job.id}.json`;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  document.addEventListener("click",(event)=>{
    const button=event.target.closest("[data-action],[data-stage],[data-view],[data-job]");
    if(!button)return;
    if(button.matches("a")&&!(button.dataset.action||button.dataset.stage||button.dataset.view))return;
    event.preventDefault();
    if(button.disabled)return;
    if(button.dataset.view){ui.view=button.dataset.view;renderAll();return;}
    if(button.dataset.stage){setStage(button.dataset.stage);return;}
    if(button.dataset.job){doAction(`job-${button.dataset.job}`,()=>startJob(button.dataset.job,{},button),button);return;}
    const action=button.dataset.action;
    if(action==="advanced-mode"){applyMode("advanced",true);return;}
    if(action==="settings"){openSettings();return;}
    if(action==="help"){openDialog($("#help-dialog"));return;}
    if(action==="close-dialog"){closeDialog(button.closest("dialog"));return;}
    if(action==="source"){$("#source-content").textContent=selectedSkill()?.content || "";openDialog($("#source-dialog"));return;}
    if(action==="refresh"){doAction("refresh",()=>refreshState(),button);return;}
    if(action==="new-skill"){if(!allowSkillSwitch())return;selectSkill(null,false);ui.addingSkill=true;ui.stage="skill";ui.view="workspace";renderAll();return;}
    if(action==="sample"){if(!allowSkillSwitch())return;doAction("sample",async()=>{const skill=await api("/api/sample",{method:"POST",body:{}});updateSkill(skill);selectSkill(skill.id,false);ui.stage="cases";renderAll();notify("Working sample loaded. Its checks and evaluations run only when you start them.");},button);return;}
    if(action==="add-case"){const records=draftRecords();records.push({selected:true,editing:true,value:{id:`${selectedSkill().name}-${Date.now()}`,prompt:"",expected_skill:selectedSkill().name,expected_script:null,ground_truth:"",expected_behavior:[],authoring_status:"draft_requires_oracle_review"}});ui.datasetDirty=true;ui.caseRevision++;renderAll();$$("[data-case-field=question]").at(-1)?.focus();return;}
    if(action==="remove-case"){
      const records=draftRecords(),index=Number(button.dataset.index);
      if(!Number.isInteger(index)||index<0||index>=records.length)return;
      records.splice(index,1);ui.datasetDirty=true;ui.caseRevision++;renderAll();
      ($$("[data-action=remove-case]")[Math.min(index,records.length-1)] || $("[data-action=save-cases]") || $("[data-action=add-case]"))?.focus();
      return;
    }
    if(action==="run-selected-checks"){const form=$("#evaluate-form");if(!form)return;readEvalOptions(form);const {tier1,tier1_llm,dedup}=ui.evalOptions;doAction("selected-checks",()=>{if(!tier1&&!dedup)throw new Error("Select at least one extra check.");return startJob("check",{tier1,tier1_llm,dedup},button);},button);return;}
    if(action==="save-cases"){doAction("save-cases",async()=>{await saveCases();renderAll();},button);return;}
    if(action==="continue-evaluate"){const skillId=ui.skillId;doAction("save-cases",async()=>{if(ui.datasetDirty)await saveCases();if(ui.skillId!==skillId)return;if(ui.datasetDirty){renderAll();return;}if(selectedSkill().cases?.some(c=>c.authoring_status==="draft_requires_oracle_review"))throw new Error("Verify the inputs and expected result for each draft case, then mark it reviewed.");if(!selectedSkill().cases?.length)throw new Error("Save at least one reviewed case first.");if(ui.stage==="cases"&&ui.view==="workspace")setStage("evaluate");},button);return;}
    if(action==="cancel-job"){doAction(`cancel-${button.dataset.id}`,async()=>{const job=await api(`/api/jobs/${encodeURIComponent(button.dataset.id)}`,{method:"DELETE"});const i=ui.state.jobs.findIndex((j)=>j.id===job.id);ui.state.jobs[i]=job;notify(activeStatus(job.status)?"Cancellation requested. Waiting for the worker to stop.":`Run ${job.status}.`);if(ui.view==="runs")renderHistory();else renderJobRegion();schedulePoll();},button);return;}
    if(action==="open-job"){doAction(`open-${button.dataset.id}`,async()=>{const job=await fetchJob(button.dataset.id);const switching=ui.skillId!==job.skill_id;if(switching&&!allowSkillSwitch())return;if(!switching&&job.action==="draft"&&job.result?.kind==="dataset"&&ui.datasetDirty&&!window.confirm("Replace your unsaved case edits with this completed draft? Saved cases stay unchanged until you Save cases."))return;if(switching)selectSkill(job.skill_id,false);ui.jobId=job.id;ui.view="workspace";ui.stage=jobStage(job);if(job.action==="draft"&&job.result?.kind==="dataset")applyDraft(job);
      if(job.action==="improve")ui.improvement={proposalId:job.id,discoveryId:job.discovery_job_id};
      if(job.action==="compare_improvement")ui.improvement={proposalId:job.proposal_job_id,discoveryId:jobById(job.proposal_job_id)?.discovery_job_id,comparisonId:job.id,beforeId:job.before_job_id,afterId:job.after_job_id,outcomeMetrics:(job.outcome_metrics || []).join(", "),loadedProposalId:job.proposal_job_id};
      if(job.validation_of)ui.improvement={proposalId:job.validation_of,beforeId:job.before_job_id,afterId:job.id,outcomeMetrics:(job.outcome_metrics || []).join(", "),loadedProposalId:job.validation_of};
      renderAll();if(ui.stage==="improve")await loadImprovement();schedulePoll();},button);return;}
    if(action==="download-result"){doAction("download-result",async()=>downloadResult(await fetchJob(button.dataset.id)),button);return;}
    if(action==="select-case"){ui.selectedCase=Number(button.dataset.index);ui.attempt=0;renderEvaluationResults();return;}
    if(action==="select-arm"){ui.arm=button.dataset.arm;ui.attempt=0;renderEvaluationResults();return;}
    if(action==="select-detail"){ui.detail=button.dataset.detail;renderEvaluationResults();return;}
    if(action==="review-ready-draft"){const job=jobById(ui.pendingDraft);if(!job?.result)return;if(ui.datasetDirty&&!window.confirm("Replace your current case edits with the completed draft? Saved cases stay unchanged until you Save cases."))return;applyDraft(job);ui.stage="cases";renderAll();return;}
    if(action==="keep-current-cases"){ui.pendingDraft=null;renderAll();notify("Your current cases are kept. The generated draft is still available in Run history.");return;}
    if(action==="improve-from-run"){const job=jobById(button.dataset.id);ui.improvement=job?.validation_of?{proposalId:job.validation_of,beforeId:job.before_job_id,afterId:job.id,outcomeMetrics:(job.outcome_metrics || []).join(", "),loadedProposalId:job.validation_of}:{discoveryId:button.dataset.id,beforeId:button.dataset.id};setStage("improve");return;}
    if(action==="propose-change"||action==="export-evidence"){doAction("propose",()=>proposeChange(action==="export-evidence"?"export":"llm"),button);return;}
    if(action==="new-proposal"){ui.improvement.proposalId=null;ui.improvement.comparisonId=null;ui.improvement.afterId=null;ui.jobId=ui.improvement.discoveryId;renderAll();return;}
    if(action==="validate-proposal"){doAction("validate-proposal",validateProposal,button);return;}
    if(action==="compare-proposal"){doAction("compare-proposal",compareProposal,button);return;}
    if(action==="original-results"){const {original}=improvementContext(),id=ui.improvement.discoveryId;if(!allowSkillSwitch())return;selectSkill(original.id,false);ui.jobId=id;ui.stage="evaluate";renderAll();return;}
    if(action==="reset-candidate"){ui.candidateContent=null;renderAll();return;}
    if(action==="open-candidate"){if(button.dataset.id===ui.skillId){setStage("evaluate");return;}if(!allowSkillSwitch())return;doAction("open-candidate",async()=>{selectSkill(button.dataset.id,false);if(ui.jobId)await fetchJob(ui.jobId);await loadComparisonJobs();ui.stage="evaluate";ui.view="workspace";renderAll();schedulePoll();},button);return;}
  });

  function renderEvaluationResults() {
    const job=selectedJob()?.result?.kind==="evaluation"?selectedJob():latestJob(ui.skillId,"evaluate");
    const target=$("#evaluation-results");
    if(target&&job?.result?.kind==="evaluation")target.innerHTML=renderResults(job);
  }

  document.addEventListener("submit",(event)=>{
    const form=event.target;if(!form.matches("#paste-form,#draft-form,#evaluate-form,#candidate-form,#settings-form"))return;
    event.preventDefault();
    const submit=event.submitter || $("button[type=submit]",form);
    if(!form.checkValidity()){
      const invalid=form.querySelector(":invalid");
      if(invalid?.closest(".advanced-only"))applyMode("advanced",true);
      for(let parent=invalid?.parentElement;parent;parent=parent.parentElement)if(parent.tagName==="DETAILS")parent.open=true;
      form.reportValidity();return;
    }
    if(form.id==="paste-form"){doAction("import-paste",async()=>{const skill=await api("/api/skills",{method:"POST",body:{content:$("#skill-text").value}});updateSkill(skill);selectSkill(skill.id,false);ui.stage="cases";renderAll();notify("Skill added as a local copy. Your original stays unchanged.");},submit);}
    if(form.id==="draft-form"){const data=new FormData(form);ui.guidance={task:String(data.get("task")||""),success:String(data.get("success")||""),avoid:String(data.get("avoid")||"")};const generation=String(data.get("generation"));doAction("draft",()=>startJob("draft",{no_llm:generation==="template"}),submit);}
    if(form.id==="evaluate-form"){readEvalOptions(form);doAction("evaluate",()=>startJob("evaluate",clone(ui.evalOptions)),submit);}
    if(form.id==="candidate-form"){doAction("candidate",async()=>{const content=$("#candidate-content").value;if(content===selectedSkill().content)throw new Error("Make an intentional edit before creating a candidate copy.");const skill=await api(`/api/skills/${encodeURIComponent(selectedSkill().id)}/candidate`,{method:"POST",body:{content}});updateSkill(skill);selectSkill(skill.id,false);ui.stage="improve";renderAll();notify("Candidate copy created. It has not been evaluated.");},submit);}
    if(form.id==="settings-form"){doAction("settings",async()=>{const data=new FormData(form),key=String(data.get("api_key")||"").trim();try{const provider=await api("/api/config",{method:"POST",body:{name:data.get("name"),model:String(data.get("model")||"").trim(),...(key?{api_key:key}:{})}});ui.state.provider=provider;ui.evalOptions.model=provider.model || "";closeDialog($("#settings-dialog"));renderAll();notify("Provider configuration saved. No model call was made by saving.");}catch(error){$("#settings-error").textContent=error.message;$("#settings-error").hidden=false;throw error;}finally{$("#provider-key").value="";}},submit);}
  });

  function readEvalOptions(form) {
    const data=new FormData(form);
    ui.evalOptions={env_mode:String(data.get("env_mode")||"local"),n_attempts:Number(data.get("n_attempts")||1),n_concurrent:Number(data.get("n_concurrent")||1),skip_baseline:data.has("skip_baseline"),agent:String(data.get("agent")||"opencode"),model:String(data.get("model")||"").trim(),timeout_multiplier:Number(data.get("timeout_multiplier")||3),grading_mode:String(data.get("grading_mode")||"default"),tier1:data.has("tier1"),tier1_llm:data.has("tier1")&&data.has("tier1_llm"),dedup:data.has("dedup")};
    const ai=$("[name=tier1_llm]",form),group=$("#scan-ai-option",form),button=$("#run-selected-checks",form),count=$("#optional-check-count",form);
    if(ai&&!ui.evalOptions.tier1)ai.checked=false;
    if(group)group.hidden=!ui.evalOptions.tier1;
    if(button)button.disabled=!(ui.evalOptions.tier1||ui.evalOptions.dedup);
    if(count)count.textContent=Number(ui.evalOptions.tier1)+Number(ui.evalOptions.dedup) || "";
    const pipeline=$("#pipeline-note",form);if(pipeline)pipeline.hidden=!(ui.evalOptions.tier1||ui.evalOptions.dedup);
    syncEvaluationAction();
  }

  document.addEventListener("input",(event)=>{
    const input=event.target;
    if(input.matches("[data-case-field]")){
      const record=draftRecords()[Number(input.dataset.index)];if(!record)return;
      const field=input.dataset.caseField;
      if(field==="selected"){record.selected=input.checked;input.closest(".case-editor")?.classList.toggle("excluded",!input.checked);}
      else if(field==="reviewed"){
        record.value.authoring_status=input.checked?"qualified":"draft_requires_oracle_review";
        if(input.checked&&Array.isArray(record.value.unresolved_requirements)&&record.value.unresolved_requirements.length){record.value.reviewed_requirements=record.value.unresolved_requirements;record.value.unresolved_requirements=[];}
        if(!input.checked&&record.value.reviewed_requirements?.length){record.value.unresolved_requirements=record.value.reviewed_requirements;delete record.value.reviewed_requirements;}
      }
      else if(field==="inactive"){record.value.expected_skill=input.checked?null:selectedSkill().name;if(input.checked)record.value.expected_script=null;input.closest("label")?.classList.toggle("advanced-only",!input.checked);}
      else if(field==="question"){record.value.prompt=input.value;record.value.question=input.value;}
      else if(field==="ground_truth"){record.value.ground_truth=input.value;record.value.expected_output=input.value;}
      else if(field==="expected_behavior"){const values=input.value.split("\n").map((s)=>s.trim()).filter(Boolean);record.value.expected_behavior=values;record.value.assertions=values;}
      else record.value[field]=input.value;
      if(!["selected","reviewed","split"].includes(field)){
        record.value.authoring_status="draft_requires_oracle_review";
        if(record.value.reviewed_requirements?.length){record.value.unresolved_requirements=record.value.reviewed_requirements;delete record.value.reviewed_requirements;}
      }
      const card=input.closest(".case-editor"),review=card?.querySelector("[data-case-field=reviewed]"),badge=card?.querySelector("[data-review-badge]");
      const qualified=record.value.authoring_status==="qualified";
      if(review)review.checked=qualified;
      if(badge){badge.textContent=qualified?"Reviewed":"Needs review";badge.className=`badge ${qualified?"good":"attention"}`;}
      const preview=card?.querySelector(`[data-case-preview="${field}"]`);if(preview)preview.textContent=input.value;
      ui.datasetDirty=true;ui.caseRevision++;const dirty=$("#dataset-dirty");if(dirty){dirty.textContent="Unsaved changes";dirty.className="unsaved";}const next=$("[data-action=continue-evaluate]");if(next){next.textContent="Save and continue →";}
    }
    if(input.id.startsWith("guidance-"))ui.guidance[input.name]=input.value;
    if(input.id==="improve-outcomes")ui.improvement.outcomeMetrics=input.value;
    if(input.id==="candidate-content")ui.candidateContent=input.value;
    if(input.id==="skill-text")ui.pasteContent=input.value;
    if(input.closest("#evaluate-form")){readEvalOptions($("#evaluate-form"));const count=$("#run-count");if(count)count.innerHTML=renderRunCount(selectedSkill()?.cases?.length||0);const readiness=$("#case-readiness-note");if(readiness)readiness.innerHTML=renderCaseReadiness(selectedSkill());}
  });

  document.addEventListener("toggle",event=>{
    if(event.target.matches("[data-case-editor]")&&event.target.isConnected){const record=draftRecords()[Number(event.target.dataset.caseEditor)];if(record)record.editing=event.target.open;}
    if(event.target.id==="improve-settings"&&event.target.isConnected)ui.improvement.advancedOpen=event.target.open;
  },true);

  document.addEventListener("change",(event)=>{
    const input=event.target;
    if(input.id==="improve-discovery"){ui.improvement.discoveryId=input.value;ui.improvement.beforeId=input.value;ui.improvement.readiness=null;ui.improvement.error=null;loadImprovement();return;}
    if(input.id==="improve-before"){ui.improvement.beforeId=input.value;ui.improvement.comparisonId=null;renderAll();return;}
    if(input.id==="improve-after"){ui.improvement.afterId=input.value;ui.improvement.comparisonId=null;doAction("load-validation",async()=>{if(input.value)await fetchJob(input.value);renderAll();});return;}
    if(input.id==="improve-observations"){ui.improvement.observations=input.checked;renderAll();return;}
    if(input.id==="improve-edits"){doAction("import-edits",async()=>{const file=input.files?.[0];if(!file)return;if(file.size>1024*1024)throw new Error("Choose an edit proposal smaller than 1 MB.");const skillId=ui.skillId,proposal=JSON.parse(await file.text());if(ui.skillId!==skillId)throw new Error("The selected skill changed. Choose the proposal again.");await proposeChange("deterministic",proposal);},input);return;}
    if(input.id==="theme-select"){applyTheme(input.value,true);return;}
    if(input.name==="display_mode"){applyMode(input.value,true);return;}
    if(input.id==="provider-name"){
      const sameProvider=input.value===ui.state?.provider?.name;
      $("#provider-model").value=sameProvider?(ui.state.provider.model || ""):"";
      $("#provider-key").value="";
      $("#settings-dialog .provider-link").hidden=input.value!=="nv_build";
      $("#settings-title").textContent=`Connect ${providerLabel(input.value)}.`;
      $("#provider-key-label").textContent=`${input.value==="nv_build"?"NVIDIA":providerLabel(input.value)} API key`;
      $("#provider-state").textContent=sameProvider&&ui.state.provider.configured?`A provider key is configured (${ui.state.provider.key_source || "server session"}).`:`Choose a key for ${providerLabel(input.value)}, or use an inherited environment key. Leaving Model ID blank uses this provider’s default.`;
      return;
    }
    if(input.id==="skill-picker"){if(!input.value||input.value===ui.skillId)return;if(!allowSkillSwitch()){input.value=ui.skillId || "";return;}doAction("select-skill",async()=>{selectSkill(input.value,false);if(ui.jobId)await fetchJob(ui.jobId);await loadComparisonJobs();renderAll();schedulePoll();});return;}
    if(input.id==="skill-file"){doAction("import-file",()=>importSkill(input.files?.[0]));return;}
    if(input.id==="dataset-file"){doAction("import-dataset",async()=>{const file=input.files?.[0],skillId=ui.skillId;if(!file)return;if(file.size>5*1024*1024)throw new Error("Dataset imports are limited to 5 MB.");const parsed=JSON.parse(await file.text()),cases=Array.isArray(parsed)?parsed:parsed.cases || parsed.evals;if(!Array.isArray(cases)||!cases.length||cases.some((c)=>!c||typeof c!=="object"||Array.isArray(c)))throw new Error("Provide a nonempty array of native test-case objects.");if(ui.skillId!==skillId)throw new Error("Your selected skill changed. Import the dataset again for the skill you want to update.");if(ui.datasetDirty&&!window.confirm("Replace your unsaved case edits with the imported dataset? Saved cases stay unchanged until you Save cases.")){input.value="";return;}ui.draft=cases.map((value)=>({value,selected:true}));ui.draftMode="imported";ui.datasetDirty=true;ui.caseRevision++;renderAll();notify("Dataset imported for review. Save cases before evaluation.");});return;}
    if(input.id==="trial-picker"){ui.attempt=Number(input.value);renderEvaluationResults();return;}
    if(input.name==="generation"){ui.generation=input.value;syncDraftMethod();return;}
    if(input.name==="env_mode"){
      const form=$("#evaluate-form");if(form){readEvalOptions(form);const mode=ui.state.capabilities?.[ui.evalOptions.env_mode],hint=$("#environment-hint"),state=$("#environment-state");hint.textContent=ui.evalOptions.env_mode==="local"?"Runs on this computer.":"Runs in Docker.";state.textContent=mode?.reason || (capability(mode)?`${ui.evalOptions.env_mode==="local"?"Local execution":"Docker"} is available.`:"This environment is unavailable.");state.className=capability(mode)?"advanced-only fine-print":"note attention";}
    }
  });

  document.addEventListener("dragover",(event)=>{const zone=event.target.closest("#dropzone");if(zone){event.preventDefault();zone.classList.add("dragging");}});
  document.addEventListener("dragleave",(event)=>{const zone=event.target.closest("#dropzone");if(zone&&!zone.contains(event.relatedTarget))zone.classList.remove("dragging");});
  document.addEventListener("drop",(event)=>{const zone=event.target.closest("#dropzone");if(zone){event.preventDefault();zone.classList.remove("dragging");doAction("import-file",()=>importSkill(event.dataTransfer?.files?.[0]));}});
  $$("dialog").forEach((dialog)=>{dialog.addEventListener("click",(event)=>{if(event.target===dialog){const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)closeDialog(dialog);}});dialog.addEventListener("close",()=>{if(dialog.id==="settings-dialog")$("#provider-key").value="";});});
  $("#local-address").textContent=location.host;
  refreshState().catch(()=>{});
})();
