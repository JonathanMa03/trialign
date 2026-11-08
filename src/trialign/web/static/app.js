"use strict";
const $ = id => document.getElementById(id);
const labels = {equivalent:"Equivalent",a_more_restrictive:"A more restrictive",b_more_restrictive:"B more restrictive",partial_overlap:"Partial overlap",contradictory_or_disjoint:"Contradictory / disjoint",related_not_comparable:"Related, not comparable",unrelated:"Unrelated",ambiguous:"Ambiguous"};
const coverageLabels = {supported_candidate:"Supported candidate",needs_review:"Needs review",not_examined:"Not examined (run limit)",no_candidate_found:"No counterpart assigned"};
let current = null, selected = null, detailKey = "", polling = null, health = null, uploaded = null, pendingFiles = {combined:[], A:[], B:[]}, catalogTrials = [], viewEpoch = 0;
const running = run => ["queued","running","cancelling"].includes(run.status);
function node(tag, text, cls) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(cls)e.className=cls; return e; }
function notice(text,error=false){$("notice").hidden=!text;$("notice").textContent=text;$("notice").className="notice"+(error?" error":"");}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const result=await response.json();if(!response.ok)throw Error(result.error||"Request failed");return result;}
let selectableTrials = [];
let startingComparison = false;
function trialLabel(t) { return `${t.trial_id.startsWith("uploaded-") ? t.title : t.trial_id + " · " + t.title} (${t.criteria} criteria)`; }
function updateSelectionSummary() {
  const root = $("selectionSummary");
  root.replaceChildren();
  root.hidden=!$("trialA").value&&!$("trialB").value;
  for (const side of ["A", "B"]) {
    const trial = selectableTrials.find(t => t.trial_id === $("trial" + side).value);
    root.append(node("p", `Trial ${side}: ${trial ? trialLabel(trial) : "Choose a matching trial"}`));
  }
  const valid = $("trialA").value && $("trialB").value && $("trialA").value !== $("trialB").value;
  $("compareButton").disabled = !valid || startingComparison || !!(current && running(current));
}
function renderTrialOptions(side, selectedId, {editing = false} = {}) {
  const input = $("trialSearch" + side), list = $("trialOptions" + side);
  if (!editing) input.value = selectedId || "";
  const query = input.value.trim().toLowerCase();
  const matches = selectableTrials.filter(t => `${t.trial_id} ${t.title}`.toLowerCase().includes(query));
  list.replaceChildren();
  for (const t of matches) {
    const option = node("option"); option.value = t.trial_id; option.label = trialLabel(t); list.append(option);
  }
  const exact = query ? selectableTrials.filter(t => t.trial_id.toLowerCase() === query || t.title.toLowerCase() === query) : [];
  const trial = exact.length === 1 ? exact[0] : null;
  $("trial" + side).value = trial?.trial_id || "";
  if (trial) input.value = trial.trial_id;
  input.setAttribute("aria-invalid", String(!!query&&!trial));
  $("trialSearchStatus" + side).textContent = !query ? "" : trial ? trialLabel(trial)
    : matches.length ? `${matches.length} matches · choose a trial from the suggestions`
    : "No matching trial. Enter an NCT number or choose a suggestion.";
  updateSelectionSummary();
}
function resetTrialSearches() {
  for (const side of ["A", "B"]) renderTrialOptions(side, $("trial" + side).value);
}
function trialOptions(trials) {
  selectableTrials = trials;
  for (const side of ["A", "B"]) renderTrialOptions(side, "");
}
for (const side of ["A", "B"]) {
  for (const event of ["input", "change"]) $("trialSearch" + side).addEventListener(event,
    () => renderTrialOptions(side, "", {editing:true}));
}

async function loadSaved(){const {runs}=await api("/api/runs");const select=$("savedRuns"),previous=current?.run_id||select.value;select.replaceChildren(node("option","Choose a saved comparison"));select.firstChild.value="";for(const run of runs){const option=node("option",`${new Date(run.created_at).toLocaleString()} · ${run.title_a||run.trial_a} ↔ ${run.title_b||run.trial_b} · ${run.status}`);option.value=run.run_id;select.append(option);}select.value=previous||"";}
function setActive(run){if(current?.run_id!==run.run_id){resetTrialSearches();$("trialA").value=run.trial_a;$("trialB").value=run.trial_b;resetTrialSearches();$("savedRuns").value=run.run_id;}current=run;render();if(polling)clearTimeout(polling);if(running(run))polling=setTimeout(poll,950);}
async function poll(){
  if(!current)return;
  const id=current.run_id,epoch=viewEpoch;
  try{
    const result=await api("/api/runs/"+id);
    if(epoch!==viewEpoch||current?.run_id!==id)return;
    setActive(result);
    if(!running(result))await loadSaved();
  }catch(error){
    if(epoch!==viewEpoch||current?.run_id!==id)return;
    notice(error.message,true);polling=setTimeout(poll,2500);
  }
}
function render(){const run=current;$("emptyState").hidden=true;$("workspace").hidden=false;$("compareButton").disabled=running(run);$("importButton").disabled=running(run);$("cancelButton").hidden=false;$("cancelButton").disabled=!running(run)||run.status==="cancelling";$("cancelButton").textContent=run.status==="cancelling"?"Stopping…":"Stop comparison";for(const id of ["trialA","trialB","trialSearchA","trialSearchB","swap","useCatalog","prepareUpload","uploadMode","chooseCombined","chooseA","chooseB","savedRuns"]){$(id).disabled=running(run);}$("exportButton").hidden=false;$("exportButton").href=`/api/runs/${run.run_id}/export`;$("runStatus").textContent=run.status.replaceAll("_"," ");$("timeUsage").textContent=`Time budget: ${run.max_seconds/60} min`;$("runMessage").textContent=(run.matching_policy?"":"Historical run used the earlier matcher. Start a new comparison to apply one-to-one clinical matching. ")+run.message;$("callUsage").textContent=run.max_model_calls==null?`${run.model_calls} LLM calls · no call cap`:`${run.model_calls}/${run.max_model_calls} LLM calls`;$("progress").max=Math.max(1,run.summary.candidate_pairs);$("progress").value=run.summary.compared_pairs;$("statPairs").textContent=`${run.summary.compared_pairs} / ${run.summary.candidate_pairs}`;$("statSupported").textContent=run.summary.supported_pairs;$("statReview").textContent=run.summary.needs_review;$("statSaved").textContent=run.summary.reviewed_pairs;
updateSelectionSummary();updateInputSummary(run);if(!selected||!run.results.some(p=>p.pair_id===selected))selected=run.results[0]?.pair_id||null;renderRows();renderDetail();renderCoverage();renderActivity();renderRunMetadata();}
function byId(id){return [...current.criteria_a,...current.criteria_b].find(r=>r.criterion_id===id);}
function renderRows(){const list=$("pairRows");list.replaceChildren();const filter=$("resultFilter").value,search=$("searchResults").value.toLowerCase();for(const pair of current.results){if(filter==="review"&&!pair.should_escalate||filter==="supported"&&pair.should_escalate||filter==="reviewed"&&!current.reviews[pair.pair_id])continue;const a=byId(pair.criterion_a_id),b=byId(pair.criterion_b_id);if(search&&!`${a.entity} ${a.source} ${b.entity} ${b.source}`.toLowerCase().includes(search))continue;const row=node("button",undefined,"pair-row"+(selected===pair.pair_id?" active":""));row.append(node("strong",a.entity==="unspecified"&&b.entity==="unspecified"?"Criterion comparison":`${a.entity} ↔ ${b.entity}`));row.append(node("span",a.source.slice(0,100)+(a.source.length>100?"…":""),"snippet"));row.append(node("span",(labels[pair.label]||"Not resolved"),"tag"+(pair.should_escalate?" review":"")));if(current.reviews[pair.pair_id])row.append(node("span","Reviewed","tag saved"));row.addEventListener("click",()=>{selected=pair.pair_id;renderRows();renderDetail();});list.append(row);}if(!list.children.length)list.append(node("p",running(current)?"Waiting for the first comparison…":"No comparisons match this filter."));}
function sourceCard(row,side){const card=node("section",undefined,"source-card");card.append(node("div",`TRIAL ${side} · ${row.original_trial_id||row.upload_sheet||row.upload_file||row.trial_id}`,"eyebrow"));card.append(node("div",row.source,"source-text"));card.append(node("div",`${row.role_unspecified?"role unspecified":row.criterion_type} · ${row.entity}${row.unit?" · "+row.unit:""}`,"source-meta"));const details=node("details");details.append(node("summary","Source, parent and context"));details.append(node("p",row.parent_statement,"parent-text"));if(row.governing_context)details.append(node("p",row.governing_context,"parent-text"));if(row.retrieved_context){const excerpt=node("details");excerpt.append(node("summary","Retrieved source excerpt"),node("p",row.retrieved_context,"parent-text"));details.append(excerpt);}for(const issue of row.source_quality_issues||[])details.append(node("p",issue,"tag review"));details.append(node("p",`${row.upload_file?row.upload_file+" / "+(row.upload_sheet||"table")+" / row "+row.upload_row+" · ":""}${row.original_criterion_id||row.criterion_id} · ${row.source_version||"imported source"}`,"source-meta"));if(row.source_uri){try{const url=new URL(row.source_uri);if(["https:","http:"].includes(url.protocol)){const a=node("a","Open registry source");a.href=url.href;a.target="_blank";a.rel="noopener noreferrer";details.append(a);}}catch{}}card.append(details);return card;}
function renderDetail(){const pair=current.results.find(r=>r.pair_id===selected);if(!pair){detailKey="";$("pairDetail").replaceChildren(node("div","Pair results will appear here as the workflow runs.","detail-empty"));return;}const key=`${current.run_id}:${selected}:${current.revision}:${running(current)}`;if(key===detailKey)return;detailKey=key;const root=$("pairDetail");root.replaceChildren();const title=node("div",undefined,"detail-title");title.append(node("h2",(labels[pair.label]||"Not resolved")),node("span",pair.should_escalate?"Needs review":"Evidence-supported proposal","tag"+(pair.should_escalate?" review":"")));root.append(title);const sources=node("div",undefined,"source-columns");sources.append(sourceCard(byId(pair.criterion_a_id),"A"),sourceCard(byId(pair.criterion_b_id),"B"));root.append(sources);const explanation=node("section",undefined,"explanation");explanation.append(node("h3","Relationship explanation"),node("p",pair.rationale),node("p","Comparison frame: "+pair.common_frame,"frame"));if(pair.assumptions.length)explanation.append(node("p","Assumptions: "+pair.assumptions.join(" "),"frame"));root.append(explanation);
const critique=node("section",undefined,"critique "+pair.critic.verdict);critique.append(node("strong",`${pair.route==="exact_tools"?"Rule critic":"Critic"}: ${pair.critic.verdict}`),node("div",pair.critic.rationale));if(pair.critic.issues.length){const ul=node("ul");for(const issue of pair.critic.issues)ul.append(node("li",issue));critique.append(ul);}root.append(critique);
const evidence=node("details",undefined,"evidence-list");evidence.append(node("summary","Evidence cited"));for(const item of pair.evidence)evidence.append(node("blockquote",`[${item.side}] ${item.quote}`));root.append(evidence);root.append(reviewForm(pair));const trace=node("details",undefined,"trace");trace.append(node("summary",`How this comparison was made · ${pair.trace.length} executed steps`));trace.append(node("pre",JSON.stringify(pair.trace,null,2)));root.append(trace);}
function reviewForm(pair){const section=node("section",undefined,"review-form");section.append(node("h3","Record your review"));const history=current.reviews[pair.pair_id]||[],latest=history.at(-1);const form=node("form");const fields=node("div",undefined,"review-fields"),nameLabel=node("label","Reviewer"),name=node("input");name.name="reviewer";name.placeholder="Your name";name.required=true;name.value=latest?.reviewer_id||"";nameLabel.append(name);const labelLabel=node("label","Final relationship"),select=node("select");for(const [value,text]of Object.entries(labels)){const o=node("option",text);o.value=value;select.append(o);}select.value=latest?.final_label||pair.label||"ambiguous";labelLabel.append(select);fields.append(nameLabel,labelLabel);form.append(fields);const noteLabel=node("label","Rationale"),note=node("textarea");note.name="rationale";note.placeholder="Explain why you accept or change the proposed relationship.";note.required=true;note.value=latest?.rationale||"";noteLabel.append(note);form.append(noteLabel);const save=node("button","Save review","primary");save.type="submit";save.disabled=running(current);form.append(save);if(running(current))form.append(node("p","Review saving becomes available when the run finishes."));if(history.length)section.append(node("p",`${history.length} saved decision(s) · latest by ${latest.reviewer_id} on ${new Date(latest.created_at).toLocaleString()}`,"review-history"));form.addEventListener("submit",async event=>{event.preventDefault();save.disabled=true;try{const result=await api(`/api/runs/${current.run_id}/review`,{pair_id:pair.pair_id,reviewer_id:name.value,rationale:note.value,final_label:select.value,status:select.value==="ambiguous"?"ambiguous":select.value===pair.label?"accepted":"corrected",expected_revision:current.revision});notice("Review saved. Its previous versions remain in the exported history.");setActive(result);}catch(error){notice(error.message,true);save.disabled=false;}});section.append(form);return section;}
function renderCoverage(){for(const side of ["a","b"]){const root=$("coverage"+side.toUpperCase());root.replaceChildren();for(const row of current["criteria_"+side]){const details=node("details",undefined,"coverage-item"),summary=node("summary");summary.append(node("span",coverageLabels[row.alignment_status],"tag"+(row.alignment_status!=="supported_candidate"?" review":"")),document.createTextNode(row.entity));details.append(summary,node("p",row.source));if(row.alignment_reason)details.append(node("p",row.alignment_reason));root.append(details);}}}
function renderActivity(){const log=$("activityLog");log.replaceChildren();for(const event of current.events.slice(-12)){const li=node("li");li.append(node("time",new Date(event.time).toLocaleTimeString()),document.createTextNode(event.stage.replaceAll("."," → ")));if(event.data.tool)li.append(document.createTextNode(" · "+event.data.tool));if(event.data.label)li.append(document.createTextNode(" · "+(labels[event.data.label]||event.data.label)));if(event.data.verdict)li.append(document.createTextNode(" · "+event.data.verdict));log.append(li);}log.scrollTop=log.scrollHeight;}
$("compareButton").addEventListener("click",async()=>{if(startingComparison)return;startingComparison=true;notice("");$("compareButton").disabled=true;try{resetTrialSearches();if(!$("trialA").value||!$("trialB").value||$("trialA").value===$("trialB").value)throw Error("Choose two different trials from the suggestions before comparing.");const run=await api("/api/runs",{dataset_id:uploaded?.dataset_id||null,trial_a:$("trialA").value,trial_b:$("trialB").value,max_pairs:Number($("pairLimit").value),max_model_calls:configuredCallLimit(),max_seconds:configuredTimeLimit(),judge_model:$("settingsJudgeModel").value.trim(),critic_model:$("settingsCriticModel").value.trim(),retrieval_model:$("retrievalModel").value,pricing:configuredPricing(),use_llm:$("useLlm").checked});selected=null;detailKey="";setActive(run);await loadSaved();}catch(error){notice(error.message,true);}finally{startingComparison=false;updateSelectionSummary();}});
$("cancelButton").addEventListener("click",async()=>{try{setActive(await api(`/api/runs/${current.run_id}/cancel`,{}));}catch(error){notice(error.message,true);}});
$("swap").addEventListener("click",()=>{resetTrialSearches();const a=$("trialA").value;$("trialA").value=$("trialB").value;$("trialB").value=a;resetTrialSearches();});
$("savedRuns").addEventListener("change",async event=>{if(!event.target.value)return;try{selected=null;detailKey="";await openRun(await api("/api/runs/"+event.target.value));}catch(error){notice(error.message,true);}});
$("resultFilter").addEventListener("change",()=>current&&renderRows());$("searchResults").addEventListener("input",()=>current&&renderRows());
function updateInputSummary(run=null){$("useCatalog").hidden=!uploaded;const root=$("inputSummary");root.replaceChildren();if(run){root.append(node("strong",run.input_origin||"Saved criterion catalog"),node("p",`Displayed run: ${run.trial_a} · ${run.title_a||""}: ${run.criteria_a.length} criteria ↔ ${run.trial_b} · ${run.title_b||""}: ${run.criteria_b.length} criteria`));return;}if(uploaded){root.append(node("strong","Comparison source: uploaded files only"),node("p",uploaded.files.join(", ")));for(const t of uploaded.trials){root.append(node("p",`${t.title}: ${t.criteria} criteria`));const preview=node("details");preview.append(node("summary","Preview uploaded source"));for(const text of t.preview||[])preview.append(node("p",text));root.append(preview);}if(uploaded.trials.length<2)root.append(node("p","Upload both trials together, or use the separate Trial A / Trial B upload fields."));}else {const count=catalogTrials.reduce((sum,t)=>sum+t.criteria,0);root.append(node("strong",`Saved corpus active · ${catalogTrials.length} trials · ${count.toLocaleString()} criteria`),node("p","Choose Trial A and Trial B below, then click Compare trials. Import criteria to use your own files instead."));}}
function clearRunView(){viewEpoch++;if(polling)clearTimeout(polling);current=null;selected=null;detailKey="";localStorage.removeItem("trialign-last-run");$("workspace").hidden=true;$("emptyState").hidden=false;$("exportButton").hidden=true;$("cancelButton").disabled=true;$("savedRuns").value="";for(const id of ["trialA","trialB","trialSearchA","trialSearchB","swap","useCatalog","importButton","prepareUpload","uploadMode","chooseCombined","chooseA","chooseB","savedRuns"]){$(id).disabled=false;}$("cancelButton").textContent="Stop comparison";}
function selectUploaded(data,{clear=true}={}){uploaded=data;if(clear)clearRunView();trialOptions(data.trials);if(data.trials.length>=2){$("trialA").value=data.trials[0].trial_id;$("trialB").value=data.trials[1].trial_id;}else{$("trialA").value=data.trials[0]?.trial_id||"";$("trialB").value="";}resetTrialSearches();updateInputSummary();}
async function openRun(run){
  const epoch=++viewEpoch;
  if(run.dataset_id){
    if(uploaded?.dataset_id!==run.dataset_id){
      const data=await api("/api/uploads/"+run.dataset_id);
      if(epoch!==viewEpoch)return;
      selectUploaded(data,{clear:false});
    }
  }else{
    const data=await api("/api/trials");
    if(epoch!==viewEpoch)return;
    catalogTrials=data.trials;uploaded=null;localStorage.removeItem("trialign-upload");trialOptions(catalogTrials);
  }
  setActive(run);
}
$("importButton").addEventListener("click",()=>{$("uploadPanel").hidden=!$("uploadPanel").hidden;});
$("uploadMode").addEventListener("change",()=>{const separate=$("uploadMode").value==="separate";$("combinedUpload").hidden=separate;$("separateUpload").hidden=!separate;});
for(const [button,input,side,label]of [["chooseCombined","fileInput","combined","combinedFiles"],["chooseA","fileInputA","A","filesA"],["chooseB","fileInputB","B","filesB"]]){$(button).addEventListener("click",()=>$(input).click());$(input).addEventListener("change",event=>{pendingFiles[side]=Array.from(event.target.files);$(label).textContent=pendingFiles[side].map(f=>f.name).join(", ")||"No files selected.";event.target.value="";});}
async function filePayload(file,side){if(file.size>15000000)throw Error("Upload limit is 15 MB total.");const content_base64=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(",")[1]);reader.onerror=reject;reader.readAsDataURL(file);});return {filename:file.name,content_base64,side};}
$("prepareUpload").addEventListener("click",async()=>{const button=$("prepareUpload");button.disabled=true;try{const separate=$("uploadMode").value==="separate";if(separate&&(!pendingFiles.A.length||!pendingFiles.B.length))throw Error("Choose files for both Trial A and Trial B.");const chosen=separate?[...pendingFiles.A.map(file=>({file,side:"A"})),...pendingFiles.B.map(file=>({file,side:"B"}))]:pendingFiles.combined.map(file=>({file,side:""}));if(!chosen.length)throw Error("Choose a workbook or criteria files first.");if(chosen.reduce((sum,x)=>sum+x.file.size,0)>15000000)throw Error("Upload limit is 15 MB total.");const files=await Promise.all(chosen.map(x=>filePayload(x.file,x.side)));const result=await api("/api/import",{files,text_column:$("textColumn").value.trim(),trial_column:$("trialColumn").value.trim(),role:$("uploadRole").value});selectUploaded(result);$("uploadPanel").hidden=true;notice(`Ready: ${result.record_count} uploaded criteria across ${result.trials.length} trials. ${result.trials.length===2?"Both trials selected automatically.":"Choose the trials to compare."} ${result.warnings.join(" ")}`);}catch(error){notice(error.message,true);}finally{button.disabled=false;}});
$("useCatalog").addEventListener("click",async()=>{
  if(current&&running(current)){notice("Stop the current comparison before switching inputs.",true);return;}
  const button=$("useCatalog"),epoch=++viewEpoch;
  button.disabled=true;button.textContent="Loading saved corpus…";
  try{
    const data=await api("/api/trials");
    if(epoch!==viewEpoch)return;
    if(!data.trials.length)throw Error("The saved corpus is unavailable in this installation. Your uploaded inputs are still selected.");
    clearRunView();catalogTrials=data.trials;uploaded=null;
    localStorage.removeItem("trialign-upload");
    pendingFiles={combined:[],A:[],B:[]};
    for(const id of ["combinedFiles","filesA","filesB"]){$(id).textContent="No files selected.";}
    $("uploadPanel").hidden=true;
    trialOptions(catalogTrials);updateSelectionSummary();
    updateInputSummary();notice(`Saved corpus loaded: ${catalogTrials.length} trials. Select two trials below to compare.`);
    $("trialSearchA").focus();
  }catch(error){notice(error.message,true);}
  finally{button.disabled=false;button.textContent="Use saved corpus instead";}
});
function configuredCallLimit(){
  if($("unlimitedCalls").checked)return null;
  const value=$("callLimit").valueAsNumber;
  if(!Number.isInteger(value)||value<0)throw Error("Set a nonnegative whole-number call limit or choose Unlimited in Provider settings.");
  return value;
}
function configuredTimeLimit(){
  const minutes=$("timeBudgetMinutes").valueAsNumber;
  if(!Number.isInteger(minutes)||minutes<1||minutes>60)throw Error("Set a whole-number time budget from 1 to 60 minutes.");
  return minutes*60;
}
function configuredPricing(){
  const pricing={};
  for(const agent of ["judge","critic"]){
    pricing[agent]={};
    for(const [suffix,key] of [["Input","input_per_million"],["Cached","cached_input_per_million"],["Output","output_per_million"]]){
      const input=$(agent+"Price"+suffix),value=input.value===""?null:input.valueAsNumber;
      if(value!==null&&(!Number.isFinite(value)||value<0))throw Error("Prices must be nonnegative numbers or blank.");
      pricing[agent][key]=value;
    }
    const rates=pricing[agent];
    if(Object.values(rates).some(v=>v!==null)&&(rates.input_per_million===null||rates.output_per_million===null))throw Error("Provide both input and output prices for each priced model.");
  }
  return pricing;
}
function saveProviderSettings(){
  try{
    configuredCallLimit();configuredTimeLimit();configuredPricing();
    for(const id of ["settingsJudgeModel","settingsCriticModel"]){
      if(!$(id).value.trim())throw Error("Provide both judge and critic model names.");
    }
    $("settingsValidation").hidden=true;
    return true;
  }catch(error){$("settingsValidation").textContent=error.message;$("settingsValidation").hidden=false;return false;}
}
function closeProviderSettings(){if(saveProviderSettings())$("providerSettings").close();}
function initializeCallSettings(){
  $("retrievalModel").value="tfidf";
  $("timeBudgetMinutes").value=15;$("callLimit").value=200;
  $("unlimitedCalls").checked=false;$("callLimit").disabled=false;
  for(const agent of ["judge","critic"])for(const suffix of ["Input","Cached","Output"])$(agent+"Price"+suffix).value="";
}
function renderRunMetadata(){
  const root=$("runMetadata");root.replaceChildren();
  const settings=current.provider_settings,metrics=current.metrics,cost=current.cost_estimate;
  const add=(label,value)=>root.append(node("dt",label),node("dd",String(value??"Not recorded")));
  add("Run ID",current.run_id);
  add("Retrieval",settings?.retrieval_model||current.encoder);
  add("Configured judge model",settings?.judge_model||current.model);
  add("Configured critic model",settings?.critic_model||current.critic_model);
  add("Execution judge model",current.model||(settings?"Not used":"Not recorded"));
  add("Execution critic model",current.critic_model||(settings?"Not used":"Not recorded"));
  add("Resolved API models",current.resolved_models?.length?current.resolved_models.join(", "):"None reported");
  add("Contextual models enabled",settings?String(settings.use_llm):String(current.use_llm));
  add("Call budget",current.max_model_calls??"Unlimited");add("Time budget",`${current.max_seconds/60} minutes`);
  add("Pair limit",current.requested_pair_limit===0?"All assigned counterparts":current.requested_pair_limit);
  for(const [key,label] of [["elapsed_seconds","Total elapsed"],["retrieval_seconds","Retrieval"],["execution_seconds","Comparison execution"],["worker_thread_cpu_seconds","Local worker CPU"],["api_request_seconds","API request time"]]){
    add(label,metrics?.[key]===undefined?"Not recorded":`${metrics[key].toFixed(2)} seconds`);
  }
  add("Input / output tokens",`${current.usage?.input_tokens??0} / ${current.usage?.output_tokens??0}`);
  add("Estimated API cost",cost?.estimated_api_cost_usd==null?"Unavailable":`$${cost.estimated_api_cost_usd.toFixed(6)} USD`);
  if(cost?.estimated_api_cost_usd==null&&cost?.known_api_cost_usd!==undefined)add("Known API subtotal",`$${cost.known_api_cost_usd.toFixed(6)} USD (partial)`);
  if(cost?.reasons?.length)add("Cost unavailable because",cost.reasons.join(" "));
  if(settings?.pricing)for(const agent of ["judge","critic"]){
    const rates=settings.pricing.rates[agent];
    add(`${agent} input / cached / output $ per 1M`,[rates.input_per_million,rates.cached_input_per_million,rates.output_per_million].map(v=>v===null?"Unknown":v).join(" / "));
  }
  add("Cost scope",cost?.scope||"No cost metadata was recorded for this historical run.");
  add("Local compute",metrics?.local_device);add("Platform",metrics?`${metrics.platform} ${metrics.machine}; Python ${metrics.python_version}`:null);
  add("Created / finished",`${current.created_at} / ${current.finished_at||"In progress"}`);
}
$("unlimitedCalls").addEventListener("change",()=>{$("callLimit").disabled=$("unlimitedCalls").checked;});
initializeCallSettings();
$("providerSettingsButton").addEventListener("click",async()=>{
  try{
    health=await api("/api/health");

    $("settingsCredentialStatus").textContent=health.llm_configured?"Configured locally":"Not configured — tools-only comparison is available";
    $("providerSettings").showModal();
  }catch(error){notice("Could not load provider settings: "+error.message,true);}
});
$("closeProviderSettings").addEventListener("click",closeProviderSettings);
$("providerSettings").addEventListener("cancel",event=>{event.preventDefault();closeProviderSettings();});
let settingsBackdropDown=false;
function outsideSettings(event){
  const rect=$("providerSettings").getBoundingClientRect();
  return event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom;
}
$("providerSettings").addEventListener("pointerdown",event=>{settingsBackdropDown=event.target===$("providerSettings")&&outsideSettings(event);});
$("providerSettings").addEventListener("click",event=>{
  if(settingsBackdropDown&&event.target===$("providerSettings")&&outsideSettings(event))closeProviderSettings();
  settingsBackdropDown=false;
});

function resetComparisonForm(){
  current=null;selected=null;detailKey="";uploaded=null;pendingFiles={combined:[],A:[],B:[]};
  if(polling)clearTimeout(polling);
  for(const key of ["trialign-last-run","trialign-upload","trialign-reviewer"])localStorage.removeItem(key);
  for(const input of document.querySelectorAll("main input:not([type=checkbox])"))input.value="";
  $("uploadMode").value="combined";$("uploadRole").value="";$("resultFilter").value="all";
  $("savedRuns").value="";$("uploadPanel").hidden=true;$("workspace").hidden=true;
  $("emptyState").hidden=false;$("exportButton").hidden=true;$("compareButton").disabled=true;
}
async function init(){
  resetComparisonForm();
  try{
    health=await api("/api/health");
    if(["agent-runtime-1","agent-runtime-2","agent-runtime-3","agent-runtime-4","agent-runtime-5"].includes(health.version)){
      notice("The Python server is running an older version. Stop it with Control + C, restart python scripts/trialign.py serve --open-browser, then refresh this page.",true);return;
    }
    $("providerState").textContent=health.llm_configured?`· ${health.model}`:"· not configured; tools-only mode available";
    $("useLlm").checked=health.llm_configured;
    $("settingsJudgeModel").value=health.model||"gpt-4o-mini";
    $("settingsCriticModel").value=health.critic_model||health.model||"gpt-4o-mini";
    $("capabilities").textContent=`Skills: ${health.skills.map(s=>s.title).join("; ")}. Tools: ${health.tools.join(", ")}. Hooks: ${health.hooks.join(", ")}.`;
    catalogTrials=(await api("/api/trials")).trials;
    trialOptions(catalogTrials);await loadSaved();updateInputSummary();
  }catch(error){notice(error.message+". Check that the local server is running.",true);}
}
init();
