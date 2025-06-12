from query import Query
from verifier import Verifier
import json

query_util = Query()
verifier = Verifier()


file_path = "dataset/preliminary/questions_preliminary.json"
with open(file_path, 'r', encoding="utf-8") as f:
    raw_data = json.load(f)


data = raw_data['questions']

answer = []
not_found = [17, 38, 52, 57, 85, 92, 114, 122, 152, 164, 187, 189, 198, 202, 210, 211, 222, 237, 249, 251, 252, 268, 269, 277, 310, 447, 483, 484, 532, 534, 538, 565, 578, 598]
new_not_found = []
for i, question in enumerate(data):
    # if question['category'] != "insurance":
    #     continue
    if question['qid'] not in not_found:
        continue
    # print(i)
    # print(question['query'] + question['category'])
    #result = query_util.query(question['query'], question['category'])
    found = False
    result = [621, 175, 561, 184, 115, 553, 152, 363, 396, 94, 616, 169, 619, 211, 508, 158, 265, 571, 399, 109, 626, 391, 260, 198, 446, 26, 129, 248, 183, 443, 617, 232, 104, 160, 534, 203, 212, 339, 387, 505, 511, 200, 437, 67, 110, 300, 557, 559, 194, 13, 466, 406, 133, 201, 609, 281, 629, 389, 172, 618, 81, 270, 499, 284, 153, 27, 374, 468, 601, 635, 522, 246, 234, 447, 65, 173, 142, 70, 269, 311, 45, 450, 326, 150, 413, 145, 398, 590, 454, 268, 217, 358, 64, 86, 426, 431, 295, 79, 58, 80]
    print(question['source'])
    print(len(result))
    for i in result:
        if i in question['source']:
            found = True
            # verifier.answer(question['qid'], question['category'], i)
            answer.append({"qid": question['qid'], "retrieve": i})
            break
    if not found:
        new_not_found.append(question['qid'])
        # verifier.answer(question['qid'], question['category'], -1)
        # answer.append({"qid": question['qid'], "retrieve": -1})

# verifier.print_result()
print(new_not_found)

with open('answer3.json', 'w') as f:
    json.dump({"answer": answer}, f, indent=4)