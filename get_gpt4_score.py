import jsonlines
import sys

base_path=sys.argv[1]

data=[]
with open(f"{base_path}/evaluation.jsonl", "r", encoding="utf-8") as f:
    reader=jsonlines.Reader(f)
    for item in reader.iter(type=dict, skip_invalid=True):
        data.append(item)

scores1, scores2=[],[]

for i in data:
    if i["review"]=="None":
        continue
    score1, score2=i["review"].split("\n")[0], i["review"].split("\n")[-1]
    _,_,score1=score1.partition("Assistant 1: ")
    _,_,score2=score2.partition("Assistant 2: ")
    try:
        score1, score2= float(score1), float(score2)
        assert score1<=10 and score2<=10, f"error!"
        scores1.append(score1)
        scores2.append(score2)
    except:
        print(f"skip! As the review is {i['review']}")

print(sum(scores1)/sum(scores2))

with jsonlines.open(f"{base_path}/gpt4_result.jsonl", mode='a') as writer:
    writer.write({
        "score": sum(scores1)/sum(scores2),
        "total": len(scores1)
    })
    
