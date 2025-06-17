from pfamformer.data_handling import load, clean_train

def test_load_and_clean():
    train = load("data/random_split/train")
    assert train.shape[0] == 1086741
    test = load("data/random_split/test")
    assert test.shape[0] == 126171
    dev = load("data/random_split/dev")
    assert dev.shape[0] == 126171
    cleaned = clean_train(train)
    assert cleaned.shape[0] == 1085997