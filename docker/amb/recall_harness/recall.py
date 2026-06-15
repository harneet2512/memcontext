import json, sys, collections
from pathlib import Path
sys.path.insert(0, "/tmp/sub")
from memory_bench.memory.memcontext import MemContextProvider
from memory_bench.models import Document

data = json.load(open("/tmp/lme.json", encoding="utf-8"))
by_cat = collections.defaultdict(list)
for q in data: by_cat[q["question_type"]].append(q)

SAMPLE = int(sys.argv[1]) if len(sys.argv)>1 else 2
prov = MemContextProvider(); prov.initialize()
res = collections.defaultdict(lambda: [0,0])   # cat -> [recall_hits, total]
miss = []

def norm(s): return " ".join(s.split()).lower()

for cat, qs in by_cat.items():
    for q in qs[:SAMPLE]:
        qid=q["question_id"]; gold=set(q["answer_session_ids"])
        prov.prepare(Path("/tmp/sub"), reset=True)
        docs=[]; gold_turns=[]
        for sid,sess,date in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]):
            msgs=[{"role":t["role"],"content":t["content"]} for t in sess if isinstance(t,dict) and t.get("content")]
            docs.append(Document(id=f"{qid}_{sid}", content=json.dumps(msgs), user_id=qid, timestamp=date))
            if sid in gold:
                gold_turns += [t["content"] for t in sess if isinstance(t,dict) and t.get("has_answer")]
        prov.ingest(docs)
        served,_=prov.retrieve(q["question"], k=10, user_id=qid, query_timestamp=q["question_date"])
        st=norm("\n".join(d.content for d in served))
        hit = any(norm(gt)[:90] in st for gt in gold_turns) if gold_turns else False
        res[cat][0]+= 1 if hit else 0; res[cat][1]+=1
        if not hit: miss.append((cat, qid, len(gold_turns)))

tot=[0,0]
print("=== GOLD-TURN RECALL (effective bridge, trial17 config: embedder+semantic) ===")
for cat in sorted(res):
    h,n=res[cat]; tot[0]+=h; tot[1]+=n
    print(f"  {cat:28s} recall={h}/{n} = {100*h/n if n else 0:.0f}%")
print(f"  {'TOTAL':28s} recall={tot[0]}/{tot[1]} = {100*tot[0]/tot[1] if tot[1] else 0:.0f}%")
print("misses:", miss)
