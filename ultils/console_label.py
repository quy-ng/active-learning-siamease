import sys
import random
from dataset import augment_address


def console_label(uncertain_pairs):
    finished = False

    match_list = []
    distinct_list = []
    uncertain_list = []

    while not finished:
        try:
            record_pair = uncertain_pairs.pop()
        except IndexError:
            break

        n_match = len(match_list)
        n_distinct = len(distinct_list)

        if random.random() > 0.85:
            display_pair = (record_pair[0], (record_pair[0][0], augment_address(record_pair[0][1])))
            uncertain_pairs.append((record_pair[0], record_pair[1]))
        else:
            display_pair = (record_pair[0], record_pair[1])

        line = "(A) %s\n(B) %s" % (display_pair[0], display_pair[1])
        print(line, file=sys.stderr)

        print("{0}/10 positive, {1}/10 negative".format(n_match, n_distinct),
              file=sys.stderr)
        print('Do these records refer to the same thing?', file=sys.stderr)

        valid_response = False
        user_input = ''
        while not valid_response:
            prompt = '(y)es / (n)o / (u)nsure / (f)inished'
            valid_responses = {'y', 'n', 'u', 'f'}

            print(prompt, file=sys.stderr)
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == 'y':
            # examples_buffer.insert(0, (record_pair, 'match'))
            match_list.append(display_pair)
        elif user_input == 'n':
            # examples_buffer.insert(0, (record_pair, 'distinct'))
            distinct_list.append(display_pair)
        elif user_input == 'u':
            # examples_buffer.insert(0, (record_pair, 'uncertain'))
            uncertain_list.append(display_pair)
        elif user_input == 'f':
            print('Finished labeling', file=sys.stderr)
            finished = True
    return match_list, distinct_list, uncertain_list


if __name__ == '__main__':
    samples = [
        ('unreal, 1, xa lo ha noi, thao dien, q22, hcm, vietnam',
         'unreal inc. , #1 xlhn, thao dien, d22, hcmc, vietnam'),
        ('k & h modas, s.a., 1a calle 20-11, ',
         'k&h modas, s.a., 1a calle 20-11, , villa nueva, gt 01064'),
        ('song hong garment joint stock company, national  rd.  #10,, loc ha commune, nam dinh, viet nam ',
         'song hong garment co., ltd, 10a1 tan long hamlet, thanh phu commune, ben luc district, nam dinh, vn ')
    ]

    a, b, c = console_label(samples)
    print(a, b, c)
