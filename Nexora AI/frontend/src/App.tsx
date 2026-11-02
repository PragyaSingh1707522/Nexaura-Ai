import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, BarChart3, BookOpen, Check, ChevronDown, ChevronLeft, ChevronRight, FileText, LayoutDashboard, LogOut, Menu, MessageSquareText, Settings, Shield, ShieldAlert, ShieldCheck, Users, X } from 'lucide-react';
import { api } from './services/api';
import { Analysis, Assistant, Documents, Evaluation, Login, Monitoring, Overview, Reviews, SettingsPage, Vendors } from './pages/WorkspacePages';

type Doc = {id:number;filename:string;file_type:string;status:string;chunks:number;created_at:string};
type Source = {id:string;document_id:number;filename:string;page:number;snippet:string;score:number};
type Vendor = {id:number;vendor_name:string;industry:string;score:number;level:string;factors:Record<string,unknown>;created_at:string};
type Review = {id:number;kind:string;item_id:number;status:string;reviewer?:string|null;note?:string;created_at:string};
type NavItem = {label:string;icon:typeof LayoutDashboard;group:string};
const nav:NavItem[] = [
  {label:'Overview',icon:LayoutDashboard,group:'Workspace'}, {label:'Document Intelligence',icon:FileText,group:'Workspace'},
  {label:'Regulatory Analysis',icon:BookOpen,group:'Workspace'}, {label:'Vendor Management',icon:Users,group:'Risk'},
  {label:'Risk Assessments',icon:Shield,group:'Risk'}, {label:'AI Assistant',icon:MessageSquareText,group:'Intelligence'},
  {label:'Human Reviews',icon:Check,group:'Intelligence'}, {label:'AI Evaluation',icon:BarChart3,group:'Operations'},
  {label:'Monitoring',icon:Activity,group:'Operations'}, {label:'Settings',icon:Settings,group:'Operations'},
];
const allowed = ['.pdf','.docx','.xlsx','.txt','.md','.csv'];
const bytes = (n:number) => `${(n/1024/1024).toFixed(1)} MB`;

export default function App() {
  const [page,setPage] = useState('Overview'), [collapsed,setCollapsed] = useState(false), [mobileOpen,setMobileOpen] = useState(false);
  const [docs,setDocs] = useState<Doc[]>([]), [vendors,setVendors] = useState<Vendor[]>([]), [reviews,setReviews] = useState<Review[]>([]), [evals,setEvals] = useState<any>(null), [health,setHealth] = useState<any>(null), [models,setModels] = useState<any>(null), [audit,setAudit] = useState<any[]>([]);
  const [error,setError] = useState(''), [notice,setNotice] = useState(''), [busy,setBusy] = useState(false), [search,setSearch] = useState(''), [filter,setFilter] = useState('all');
  const [selected,setSelected] = useState<Doc|null>(null), [clauseResult,setClauseResult] = useState<any>(null), [question,setQuestion] = useState(''), [answer,setAnswer] = useState<any>(null), [reviewNote,setReviewNote] = useState<Record<number,string>>({});
  const [assistantHistory,setAssistantHistory] = useState<any[]>([]);
  const [focusedSource,setFocusedSource] = useState<Source|null>(null);
  const [category,setCategory] = useState('Regulatory Documents');
  const [categoryById,setCategoryById] = useState<Record<number,string>>(()=>{try{return JSON.parse(localStorage.getItem('rl_document_categories')||'{}')}catch{return {}}});
  const [categoryFilter,setCategoryFilter] = useState('all');
  const [vendorForm,setVendorForm] = useState({vendor_name:'',industry:'SaaS',data_sensitivity:3,access_level:3,control_maturity:3,incident_history:0,criticality:3});
  const [assessment,setAssessment] = useState<any>(null);
  const token = localStorage.getItem('rl_token');

  const refresh = useCallback(async (signal?:AbortSignal) => {
    try {
      const h:any = await api.health(signal); setHealth(h);
      if (h.auth_required && !localStorage.getItem('rl_token')) { setDocs([]); setVendors([]); setReviews([]); return; }
      const [d,v,e,r,a] = await Promise.all([api.documents(signal),api.vendors(signal),api.evaluations(signal),api.reviews(signal),api.audit(signal)]);
      setDocs(d as Doc[]); setVendors(v as Vendor[]); setEvals(e); setReviews(r as Review[]); setAudit(a as any[]); setError('');
    } catch (e:any) { if (e.name !== 'AbortError') setError(e.message); }
  },[]);
  useEffect(()=>{const controller=new AbortController();void refresh(controller.signal);return()=>controller.abort()},[refresh]);
  useEffect(()=>{localStorage.setItem('rl_document_categories',JSON.stringify(categoryById))},[categoryById]);
  useEffect(()=>{if(page==='Monitoring'||page==='Settings'){const c=new AbortController();Promise.all([api.audit(c.signal),api.models(c.signal)]).then(([a,m])=>{setAudit(a as any[]);setModels(m)}).catch((e:any)=>{if(e.name!=='AbortError')setError(e.message)});return()=>c.abort()}},[page]);
  const pending = reviews.filter(r=>r.status==='pending');
  const filteredDocs = useMemo(()=>docs.filter(d=>d.filename.toLowerCase().includes(search.toLowerCase())&&(filter==='all'||d.status===filter)&&(categoryFilter==='all'||(categoryById[d.id]||'Uncategorized')===categoryFilter)),[docs,search,filter,categoryFilter,categoryById]);
  const go=(p:string)=>{setPage(p);setMobileOpen(false);setError('');setNotice('')};
  const upload=async(files:FileList|null)=>{
    if(!files?.length)return;setError('');setNotice('');const list=Array.from(files);
    for(const f of list){const ext='.'+f.name.split('.').pop()?.toLowerCase();if(!allowed.includes(ext)){setError(`${f.name}: unsupported format. Choose PDF, DOCX, XLSX, TXT, MD or CSV.`);return}if(f.size>20*1024*1024){setError(`${f.name}: exceeds the 20 MB upload limit.`);return}}
    setBusy(true);let successful=0;
    try{for(const f of list){const result:any=await api.upload(f);setCategoryById(old=>({...old,[result.id]:category}));successful++}await refresh();setNotice(`${successful} document${successful===1?'':'s'} uploaded and processed by the API. Category tags are stored locally in this browser.`)}
    catch(e:any){setError(`${successful} of ${list.length} uploaded. ${e.message}`);await refresh()}finally{setBusy(false)}
  };
  const remove=async(id:number)=>{if(!window.confirm('Delete this document and its indexed chunks?'))return;try{await api.removeDocument(id);if(selected?.id===id){setSelected(null);setClauseResult(null)}await refresh();setNotice('Document deleted.')}catch(e:any){setError(e.message)}};
  const inspect=async(d:Doc)=>{setSelected(d);setClauseResult(null);setError('');try{setClauseResult(await api.clauses(d.id))}catch(e:any){setError(e.message)}};
  const ask=async()=>{if(question.trim().length<3)return;setBusy(true);setError('');try{const ids=selected?[selected.id]:undefined;const result=await api.ask(question.trim(),ids);setAnswer(result);setAssistantHistory(old=>[...old,result]);}catch(e:any){setError(e.message)}finally{setBusy(false)}};
  const assess=async(e:React.FormEvent)=>{e.preventDefault();setBusy(true);setError('');try{setAssessment(await api.assess(vendorForm));await refresh();}catch(err:any){setError(err.message)}finally{setBusy(false)}};
  const updateReview=async(id:number,status:string)=>{try{await api.review(id,{status,reviewer:localStorage.getItem('rl_username')||'reviewer',note:reviewNote[id]||''});await refresh();setNotice(`Review ${status}.`)}catch(e:any){setError(e.message)}};
  const login=async(username:string,password:string)=>{const session:any=await api.login(username,password);localStorage.setItem('rl_token',session.access_token);localStorage.setItem('rl_username',username);await refresh()};
  if(health?.auth_required&&!token)return <Login onLogin={login}/>;

  return <div className="app-shell">
    {mobileOpen&&<button className="mobile-scrim" aria-label="Close navigation" onClick={()=>setMobileOpen(false)}/>}
    <aside className={`sidebar ${collapsed?'is-collapsed':''} ${mobileOpen?'mobile-open':''}`}>
      <div className="brand"><span className="brand-mark"><ShieldCheck size={19}/></span>{!collapsed&&<div><b>RegulaLens <i>AI</i></b><small>COMPLIANCE INTELLIGENCE</small></div>}<button className="sidebar-toggle" title={collapsed?'Expand sidebar':'Collapse sidebar'} onClick={()=>setCollapsed(!collapsed)}><ChevronLeft size={16}/></button></div>
      {!collapsed&&<div className="workspace-switch"><span className="workspace-avatar">RL</span><span><b>Workspace</b><small>{health?.status==='ok'?'Connected to API':'API connection pending'}</small></span><ChevronDown size={15}/></div>}
      <nav className="primary-nav">{['Workspace','Risk','Intelligence','Operations'].map(group=><div className="nav-group" key={group}>{!collapsed&&<div className="nav-heading">{group}</div>}{nav.filter(i=>i.group===group).map(({label,icon:Icon})=><button title={collapsed?label:undefined} key={label} className={`nav-item ${page===label?'active':''}`} onClick={()=>go(label)}><Icon size={18}/>{!collapsed&&<span>{label}</span>}{label==='Human Reviews'&&pending.length>0&&!collapsed&&<em>{pending.length}</em>}</button>)}</div>)}</nav>
      <div className="sidebar-bottom"><div className={`connection ${health?.status==='ok'?'connected':''}`}><i/>{!collapsed&&<span>{health?.status==='ok'?'API connected':'API unavailable'}</span>}</div><div className="user-profile"><span className="profile-avatar">{token?'U':'G'}</span>{!collapsed&&<><span><b>{localStorage.getItem('rl_username')|| (token?'Signed-in user':'Local workspace')}</b><small>{token?'Authenticated session':'No account required'}</small></span>{token&&<button title="Sign out" onClick={()=>{localStorage.removeItem('rl_token');localStorage.removeItem('rl_username');window.location.reload()}}><LogOut size={16}/></button>}</>}</div></div>
    </aside>
    <main className={`main-area ${collapsed?'main-collapsed':''}`}>
      <header className="topbar"><div className="breadcrumb"><button className="mobile-menu" aria-label="Open navigation" onClick={()=>setMobileOpen(true)}><Menu size={20}/></button><span>RegulaLens</span><ChevronRight size={15}/><b>{page}</b></div><div className="topbar-actions"><span className={`system-pill ${health?.status==='ok'?'online':''}`}><i/>{health?.status==='ok'?'All systems operational':'API connection unavailable'}</span><span className="today-label">{new Date().toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'})}</span></div></header>
      <div className="page-content">{error&&<div className="alert error-alert"><ShieldAlert size={17}/><span>{error}</span><button onClick={()=>refresh()} aria-label="Retry"><Activity size={15}/> Retry</button><button className="dismiss" onClick={()=>setError('')} aria-label="Dismiss"><X size={16}/></button></div>}{notice&&<div className="alert success-alert"><Check size={16}/><span>{notice}</span><button className="dismiss" onClick={()=>setNotice('')} aria-label="Dismiss"><X size={16}/></button></div>}
        {page==='Overview'&&<Overview docs={docs} vendors={vendors} reviews={reviews} evals={evals} audit={audit} go={go} onRefreshAudit={async()=>{try{setAudit(await api.audit() as any[])}catch(e:any){setError(e.message)}}}/>}
        {page==='Document Intelligence'&&<Documents docs={filteredDocs} allCount={docs.length} search={search} setSearch={setSearch} filter={filter} setFilter={setFilter} categoryFilter={categoryFilter} setCategoryFilter={setCategoryFilter} categoryById={categoryById} upload={upload} busy={busy} category={category} setCategory={setCategory} inspect={inspect} remove={remove} selected={selected} clauses={clauseResult} ask={(d:Doc)=>{setSelected(d);go('AI Assistant')}} onExport={()=>window.print()}/>}
        {page==='Regulatory Analysis'&&<Analysis docs={docs} selected={selected} clauses={clauseResult} inspect={inspect} setPage={go} onExport={()=>window.print()} focusedSource={focusedSource}/>}
        {(page==='Vendor Management'||page==='Risk Assessments')&&<Vendors vendors={vendors} form={vendorForm} setForm={setVendorForm} assess={assess} assessment={assessment} busy={busy} riskOnly={page==='Risk Assessments'} modelReport={evals?.vendor_model_report}/>}
        {page==='AI Assistant'&&<Assistant docs={docs} selected={selected} setSelected={setSelected} setClauseResult={setClauseResult} setFocusedSource={setFocusedSource} question={question} setQuestion={setQuestion} ask={ask} busy={busy} history={assistantHistory} go={go}/>}
        {page==='Human Reviews'&&<Reviews reviews={reviews} notes={reviewNote} setNotes={setReviewNote} decide={updateReview}/>}
        {page==='AI Evaluation'&&<Evaluation data={evals}/>}
        {page==='Monitoring'&&<Monitoring audit={audit} models={models} refresh={()=>{void refresh();void api.audit().then(setAudit).catch((e:any)=>setError(e.message))}}/>}
        {page==='Settings'&&<SettingsPage health={health} models={models} auditCount={evals?.audit_events} onRefresh={()=>{void refresh();void api.models().then(setModels).catch((e:any)=>setError(e.message))}}/>}
      </div>
    </main>
  </div>;
}

