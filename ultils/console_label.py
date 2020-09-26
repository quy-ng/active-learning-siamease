

def console_label(deduper: dedupe.api.ActiveMatching) -> None:  # pragma: no cover
    '''
   Train a matcher instance (Dedupe, RecordLink, or Gazetteer) from the command line.
   Example

   .. code:: python

      > deduper = dedupe.Dedupe(variables)
      > deduper.prepare_training(data)
      > dedupe.console_label(deduper)
    '''

    finished = False
    use_previous = False
    fields = unique(field.field
                    for field
                    in deduper.data_model.primary_fields)

    buffer_len = 1  # Max number of previous operations
    examples_buffer: List[Tuple[TrainingExample, Literal['match', 'distinct', 'uncertain']]] = []
    uncertain_pairs: List[TrainingExample] = []

    while not finished:
        if use_previous:
            record_pair, _ = examples_buffer.pop(0)
            use_previous = False
        else:
            if not uncertain_pairs:
                uncertain_pairs = deduper.uncertain_pairs()

            try:
                record_pair = uncertain_pairs.pop()
            except IndexError:
                break

        n_match = (len(deduper.training_pairs['match']) +
                   sum(label == 'match' for _, label in examples_buffer))
        n_distinct = (len(deduper.training_pairs['distinct']) +
                      sum(label == 'distinct' for _, label in examples_buffer))

        for pair in record_pair:
            for field in fields:
                line = "%s : %s" % (field, pair[field])
                print(line, file=sys.stderr)
            print(file=sys.stderr)

        print("{0}/10 positive, {1}/10 negative".format(n_match, n_distinct),
              file=sys.stderr)
        print('Do these records refer to the same thing?', file=sys.stderr)

        valid_response = False
        user_input = ''
        while not valid_response:
            if examples_buffer:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished / (p)revious'
                valid_responses = {'y', 'n', 'u', 'f', 'p'}
            else:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished'
                valid_responses = {'y', 'n', 'u', 'f'}

            print(prompt, file=sys.stderr)
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == 'y':
            examples_buffer.insert(0, (record_pair, 'match'))
        elif user_input == 'n':
            examples_buffer.insert(0, (record_pair, 'distinct'))
        elif user_input == 'u':
            examples_buffer.insert(0, (record_pair, 'uncertain'))
        elif user_input == 'f':
            print('Finished labeling', file=sys.stderr)
            finished = True
        elif user_input == 'p':
            use_previous = True
            uncertain_pairs.append(record_pair)

        if len(examples_buffer) > buffer_len:
            record_pair, label = examples_buffer.pop()
            if label in {'distinct', 'match'}:

                examples: TrainingData
                examples = {'distinct': [],
                            'match': []}
                examples[label].append(record_pair)  # type: ignore
                deduper.mark_pairs(examples)

    for record_pair, label in examples_buffer:
        if label in ['distinct', 'match']:

            exmples: TrainingData
            examples = {'distinct': [], 'match': []}
            examples[label].append(record_pair)  # type: ignore
            deduper.mark_pairs(examples)