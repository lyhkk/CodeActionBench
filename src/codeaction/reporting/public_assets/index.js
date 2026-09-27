'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
const metrics=document.getElementById('metrics');
for(const [value,label] of [[data.episodes,'Attempts'],[data.agents,'Agents'],[data.tasks,'Tasks'],[data.corrected_successes,'Corrected successes']]){const box=el('div',null,'metric');box.append(el('strong',value),el('span',label));metrics.append(box);}
document.getElementById('provenance-note').textContent=`${data.original_successes} original successes · ${data.corrected_successes} corrected successes · ${data.changed_verdicts} verdicts changed. Historical task-pack versions are retained. Turn counts depend on transcript format.`;
const agent=document.getElementById('agent');
for(const name of [...new Set(data.cells.map(r=>r.agent))].sort()){const option=el('option',name);option.value=name;agent.append(option);}
function render(){const query=document.getElementById('search').value.toLowerCase();const outcome=document.getElementById('outcome').value;const field=document.getElementById('scoring').value==='corrected'?'success_corrected':'success';
const rows=data.cells.filter(r=>(!agent.value||r.agent===agent.value)&&(`${r.task} ${r.agent}`.toLowerCase().includes(query))&&(!outcome||(outcome==='changed'?r.changed_by_regrade:outcome==='pass'?r[field]:!r[field])));
const body=document.getElementById('rows');body.replaceChildren();for(const row of rows){const tr=el('tr');const task=el('td');const link=el('a',row.task.replaceAll('_',' '));link.href=row.page;task.append(link);tr.append(task,el('td',row.agent),el('td',row.attempt));const verdict=el('td');verdict.append(el('span',row[field]?'PASS':'FAIL',`badge ${row[field]?'pass':'fail'}`));if(row.changed_by_regrade)verdict.append(el('span',' · revised','note'));tr.append(verdict,el('td',row.turn_count??'—'),el('td',row.tool_calls_used??'—'),el('td',row.wall_s==null?'—':`${(row.wall_s/60).toFixed(1)} min`));body.append(tr);}document.getElementById('count').textContent=`${rows.length} of ${data.episodes} attempts`;}
for(const id of ['search','agent','outcome','scoring'])document.getElementById(id).addEventListener('input',render);render();
