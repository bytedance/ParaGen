import json

def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []
    ids = []
    q_cnt, a_cnt = 0, 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context'].replace('\n', '')
            for qa in passage['qas']:
                question = qa['question']
                i = qa['id']
                q_cnt += 1
                for answer in qa['answers']:
                    a_cnt += 1
                    ids.append(i)
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    print(f'Total Question: {q_cnt}\tTotal Answer: {a_cnt}')
    return contexts, questions, answers, ids


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        for wsize in range(10):
            for direction in [-1, 1]:
                offset = wsize * direction
                if start_idx + offset < 0 or end_idx + offset > len(context):
                    continue
                if context[start_idx+offset:end_idx+offset] == gold_text:
                    answer['answer_start'] = start_idx + offset
                    answer['answer_end'] = end_idx + offset


for split in ['train', 'dev']:
    path = f'{split}-v1.1.json'
    contexts, questions, answers, ids = read_squad(path)
    assert len(contexts) == len(questions) == len(answers) == len(ids)
    add_end_idx(answers, contexts)
    with open(f'{split}', 'w') as fout, open(f'{split}.id', 'w') as fout_id:
        for c, q, a, i in zip(contexts, questions, answers, ids):
            fout.write(f'{json.dumps({"context": c, "question": q, "answers": a})}\n')
            fout_id.write(f'{i}\n')