import json

class Verify_type():
    def __init__(self, category):
        self.category = category
        self.wrong_list = []
        self.correct_list = []
    
    def print_accuracy(self):
        total = len(self.wrong_list) + len(self.correct_list)
        if total == 0:
            return 0, 0
        print(f"Accuracy of {self.category}: {len(self.correct_list)*100/total}")
        return total, len(self.correct_list)
    
    def answer_correct(self, qid):
        self.correct_list.append(qid)

    def answer_wrong(self, qid, answer, your_answer):
        self.wrong_list.append([qid, answer, your_answer])

    def print_wrong(self):
        print("Print Wrong Answers")
        for i in self.wrong_list:
            print(f"Q: {i[0]}, correct answer: {i[1]}, your answer: {i[2]}")
        print("Wrong end.")

    

class Verifier():
    def __init__(self, file_path = 'dataset/preliminary/ground_truths_example.json'):
        with open(file_path, 'r', encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.raw_data = self.raw_data["ground_truths"]
        self.data = {"insurance":{}, "faq":{}, "finance":{}}

        for data in self.raw_data:
            self.data[data['category']][data['qid']] = data['retrieve']


        self.indeces = {"insurance":0, "faq":1, "finance":2}
        self.verify_types = [Verify_type(i) for i in self.indeces.keys()]

    def answer(self, qid, category, answer):
        assert category in ["insurance", "faq", "finance"]
        correct_answer = self.data[category][qid]
        if correct_answer == answer:
            self.verify_types[self.indeces[category]].answer_correct(qid)
        else:
            self.verify_types[self.indeces[category]].answer_wrong(qid, correct_answer, answer)

    def print_result(self):
        total = 0
        correct = 0
        for i in range(3):
            a, b = self.verify_types[i].print_accuracy()  
            total += a     
            correct += b
            self.verify_types[i].print_wrong()
        
        print(f"Overall Accuracy: {correct/total*100}%")

