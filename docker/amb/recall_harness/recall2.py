import json, sys, collections
from pathlib import Path
sys.path.insert(0, "/tmp/sub")
from memory_bench.memory.memcontext import MemContextProvider
from memory_bench.models import Document
from memcontext.retrieval import retrieve_memory_across, retrieve_memory, EmbeddingClient
from memcontext.claims import get_turn

data=json.load(open("/tmp/lme.json", encoding="utf-8"))
by=collections.defaultdict(list)
for q in data: by[q["question_type"]].append(q)
SAMPLE=int(sys.argv[1]) if len(sys.argv)>1 else 3
prov=MemContextProvider(); prov.initialize(); ec=EmbeddingClient()
def norm(s): return " ".join(s.split()).lower()
def served_texts(hits):
    out=[]; seen=set()
    for h,_ in hits:
        t=h.source_turn_id
        if t in seen: continue
        seen.add(t); tu=get_turn(prov._ensure_conn(), t)
        out.append(tu.text if tu else h.text)
    return norm("\n".join(out))

G=collections.defaultdict(lambda:[0,0]); P=collections.defaultdict(lambda:[0,0])
for cat,qs in by.items():
    for q in qs[:SAMPLE]:
        qid=q["question_id"]; gold=set(q["answer_session_ids"])
        prov.prepare(Path("/tmp/sub"), reset=True)
        docs=[]; gturns=[]; sids=[]
        for sid,sess,date in zip(q["haystack_session_ids"],q["haystack_sessions"],q["haystack_dates"]):
            msgs=[{"role":t["role"],"content":t["content"]} for t in sess if isinstance(t,dict) and t.get("content")]
            ssid=f"amb_{qid}_{sid}"; sids.append(ssid)
            docs.append(Document(id=f"{qid}_{sid}", content=json.dumps(msgs), user_id=qid, timestamp=date))
            if sid in gold: gturns+=[t["content"] for t in sess if isinstance(t,dict) and t.get("has_answer")]
        prov.ingest(docs)
        conn=prov._ensure_conn()
        # GLOBAL top-50 (bridge current)
        sg=served_texts(retrieve_memory_across(conn,session_ids=sids,query=q["question"],top_k=50,embedding_client=ec))
        # PER-SESSION keep top-3 (proposed fix)
        ph=[]
        for s in sids: ph+=retrieve_memory(conn,session_id=s,query=q["question"],top_k=3,embedding_client=ec)
        ph.sort(key=lambda x:-x[1]); sp=served_texts(ph[:200])
        gh=any(norm(g)[:90] in sg for g in gturns) if gturns else False
        ph_=any(norm(g)[:90] in sp for g in gturns) if gturns else False
        G[cat][0]+=gh; G[cat][1]+=1; P[cat][0]+=ph_; P[cat][1]+=1
tg=[0,0]; tp=[0,0]
print("=== RECALL: GLOBAL top-50  vs  PER-SESSION keep-3 ===")
for cat in sorted(G):
    g,n=G[cat]; p,_=P[cat]; tg[0]+=g;tg[1]+=n;tp[0]+=p
    print(f"  {cat:28s} global={g}/{n}={100*g/n:.0f}%   per-session={p}/{n}={100*p/n:.0f}%")
print(f"  {'TOTAL':28s} global={tg[0]}/{tg[1]}={100*tg[0]/tg[1]:.0f}%   per-session={tp[0]}/{tg[1]}={100*tp[0]/tg[1]:.0f}%")
