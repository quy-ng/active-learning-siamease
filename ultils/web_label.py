import codecs
import json
import os
import pandas as pd
from dataset import augment_address


def web_label(uncertain_pairs, task_id):
    print('working')
    task_status = f'file_{task_id}_status.json'
    task_submit = f'file_{task_id}_submit.json'
    data = []
    for i in uncertain_pairs:
        data.append({
          "a": [i[0][0], i[0][1]],
          "b": [
            [i[1][0], i[1][1]],
            [i[2][0], i[2][1]]
          ]
    })
    json.dump({
        "data": data
    }, codecs.open(task_status, 'w', 'UTF-8'))
    while True:
        if os.path.isfile(task_submit):
            match_list = []
            distinct_list = []

            existing_submit = codecs.open(task_submit, 'r', 'UTF-8').read()
            is_submit_file_empty = len(existing_submit.strip()) == 0
            if is_submit_file_empty is not True:
                with open(task_submit) as json_file:
                    data = json.load(json_file)
                for i in data['data']:
                    anchor = tuple(i['a'])
                    for j in i['b']:
                        if j[2]:
                            match_list.append((anchor, tuple(j[:2])))
                        else:
                            distinct_list.append((anchor, tuple(j[:2])))
                break
    d_plus = pd.DataFrame(match_list, columns=['anchor', 'pos'])
    d_minus = pd.DataFrame(distinct_list, columns=['anchor', 'neg'])
    t1 = pd.merge(left=d_plus, right=d_minus, how='inner', on='anchor')
    t1.dropna(inplace=True)
    t1 = t1[['anchor', 'pos', 'neg']]
    t2 = []
    for i in d_minus.itertuples():
        _aug = (i.anchor, augment_address(i.anchor[1]))
        t2.append((i.anchor, i.neg, _aug))
    t2 = pd.DataFrame(t2, columns=['anchor', 'neg', 'pos'])
    t2 = t2[['anchor', 'pos', 'neg']]

    triplets = pd.concat([t1, t2])

    return triplets.values
