import json

input_file = "crowd_train.jsonl"
output_file = "crowd_train_my.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)

        if "tensor" in data:
            data["tensor"] = data["tensor"].replace("../../", "")

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")