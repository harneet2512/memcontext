import json, sys, collections
from pathlib import Path
sys.path.insert(0, "/tmp/sub")
from memory_bench.memory.memcontext import MemContextProvider
from memory_bench.models import Document
from memcontext.retrieval import retrieve_memory, EmbeddingClient
from memcontext.claims import get_turn

data=json.load(open("/tmp/lme.json", encoding="utf-8"))
by=collections.defaultdict(list)
for q in data: by[q["question_type"]].append(q)
SAMPLE=int(sys.argv[1]) if len(sys.argv)>1 else 3
prov=MemContextProvider(); prov.initialize(); ec=EmbeddingClient()
def norm(s): return " ".join(s.split()).lower()
def texts(hits):
    out=[]; seen=set()
    for h,_ in hits:
        if h.source_turn_id in seen: continue
        seen.add(h.source_turn_id); tu=get_turn(prov._ensure_conn(), h.source_turn_id)
        out.append(tu.text if tu else h.text)
    return out

VAR=["GLOBAL-50","CONC-S8-K3","CONC-S12-K3","ALL-K3"]
rec={v:collections.defaultdict(lambda:[0,0]) for v in VAR}; nturns={v:[] for v in VAR}
for cat,qs in by.items():
    for q in qs[:SAMPLE]:
        qid=q["question_id"]; gold=set(q["answer_session_ids"])
        prov.prepare(Path("/tmp/sub"), reset=True)
        docs=[]; gturns=[]; sids=[]
        for sid,sess,date in zip(q["haystack_session_ids"],q["haystack_sessions"],q["haystack_dates"]):
            msgs=[{"role":t["role"],"content":t["content"]} for t in sess if isinstance(t,dict) and t.get("content")]
            ss=f"amb_{qid}_{sid}"; sids.append(ss)
            docs.append(Document(id=f"{qid}_{sid}",content=json.dumps(msgs),user_id=qid,timestamp=date))
            if sid in gold: gturns+=[t["content"] for t in sess if isinstance(t,dict) and t.get("has_answer")]
        prov.ingest(docs)
        conn=prov._ensure_conn()
        per={}
        for s in sids:
            hh=retrieve_memory(conn,session_id=s,query=q["question"],top_k=5,embedding_client=ec)
            if hh: per[s]=hh
        flat=[(h,sc) for s in per for (h,sc) in per[s]]
        def conc(S,K):
            rs=sorted(per.items(), key=lambda kv:-max(sc for _,sc in kv[1]))[:S]
            out=[]
            for s,hh in rs: out+=sorted(hh,key=lambda x:-x[1])[:K]
            return out
        cand={"GLOBAL-50":sorted(flat,key=lambda x:-x[1])[:50],
              "CONC-S8-K3":conc(8,3),"CONC-S12-K3":conc(12,3),
              "ALL-K3":[h for s in per for h in sorted(per[s],key=lambda x:-x[1])[:3]]}
        for v in VAR:
            tx=texts(cand[v]); st=norm("\n".join(tx)); nturns[v].append(len(tx))
            hit=any(norm(g)[:90] in st for g in gturns) if gturns else False
            rec[v][cat][0]+=hit; rec[v][cat][1]+=1
print("=== RECALL by retrieval strategy (n=%d/cat) ==="%SAMPLE)
hdr="  %-26s"%"category"+"".join("%-14s"%v for v in VAR); print(hdr)
cats=sorted(rec[VAR[0]])
tot={v:[0,0] for v in VAR}
for c in cats:
    row="  %-26s"%c
    for v in VAR:
        h,n=rec[v][c]; tot[v][0]+=h; tot[v][1]+=n; row+="%-14s"%f"{h}/{n}={100*h/n:.0f}%"
    print(row)
row="  %-26s"%"TOTAL"
for v in VAR: row+="%-14s"%f"{tot[v][0]}/{tot[v][1]}={100*tot[v][0]/tot[v][1]:.0f}%"
print(row)
print("  %-26s"%"avg turns served"+"".join("%-14s"%f"{sum(nturns[v])/len(nturns[v]):.0f}" for v in VAR))
