import sys
import pandas as pd
from dataset import augment_address


def console_label(uncertain_pairs):
    finished = False

    match_list = []
    distinct_list = []

    while not finished:
        try:
            record_pair = uncertain_pairs.pop()
        except IndexError:
            break

        n_match = len(match_list)
        n_distinct = len(distinct_list)

        display_pair = (record_pair[0], record_pair[1])

        print("\n", file=sys.stderr)

        line = "(A) %s\n(B) %s" % (display_pair[0], display_pair[1])
        print(line, file=sys.stderr)

        print("{0}/10 positive, {1}/10 negative".format(n_match, n_distinct),
              file=sys.stderr)
        print('Do these records refer to the same thing?', file=sys.stderr)

        valid_response = False
        user_input = ''
        while not valid_response:
            prompt = '(y)es / (n)o / (f)inished'
            valid_responses = {'y', 'n', 'f'}

            print(prompt, file=sys.stderr)
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == 'y':
            match_list.append(display_pair)
        elif user_input == 'n':
            distinct_list.append(display_pair)
        elif user_input == 'f':
            print('Finished labeling', file=sys.stderr)
            finished = True
    d_plus = pd.DataFrame(match_list, columns=['anchor', 'pos'])
    d_minus = pd.DataFrame(distinct_list, columns=['anchor', 'neg'])
    t1 = pd.merge(left=d_plus, right=d_minus, how='inner', on='anchor')
    t1.dropna(inplace=True)
    t1 = t1[['anchor', 'pos', 'neg']]
    t2 = []
    for i in d_minus.itertuples():
        _aug = (i.anchor[0], augment_address(i.anchor[1]))
        t2.append((i.anchor, i.neg, _aug))
    t2 = pd.DataFrame(t2, columns=['anchor', 'neg', 'pos'])
    t2 = t2[['anchor', 'pos', 'neg']]

    triplets = pd.concat([t1, t2])

    return triplets.values


if __name__ == '__main__':
    samples = [
        ('unreal, 1, xa lo ha noi, thao dien, q22, hcm, vietnam',
         'unreal inc. , #1 xlhn, thao dien, d22, hcmc, vietnam'),
        ('k & h modas, s.a., 1a calle 20-11, ',
         'k&h modas, s.a., 1a calle 20-11, , villa nueva, gt 01064'),
        ('song hong garment joint stock company, national  rd.  #10,, loc ha commune, nam dinh, viet nam ',
         'song hong garment co., ltd, 10a1 tan long hamlet, thanh phu commune, ben luc district, nam dinh, vn ')
    ]

    a = console_label(samples)
    print(a)
