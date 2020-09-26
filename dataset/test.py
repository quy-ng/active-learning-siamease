from dataset.inspectorio import create_data

if __name__ == '__main__':
    f_path = '../data/dac/dedupe-project/new_generated_labeled_data.csv'
    embeddings_dim = 50
    batch_size = 1

    loaders, raw_data = create_data(f_path, embeddings_dim, batch_size)
    anc_loader, pos_loader, neg_loader = loaders
    df, X, X_len, embeddings = raw_data

    for batch, [anc_x, pos_x, neg_x] in enumerate(zip(anc_loader, pos_loader, neg_loader)):
        print(batch)
        print(anc_x)
        print(pos_x)
        print(neg_x)

