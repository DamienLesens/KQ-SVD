from datasets import load_dataset
from transformers import AutoTokenizer
import random

def get_c4(nsamples_train, nsamples_test, seed, seqlen, model):
    """
    This function gets samples from the C4 data set, using the train/test split of the data set
    It passes them throught the tokenizer, so it returns tokens
    Parameters:
        nsamples_train: number of samples for training
        nsamples_test: number of samples for testing
        seed: random seed
        seqlen: length of each sample
        model: path to the model, just used to get the tokenizer
    """
    print("get_c4")
    traindata = load_dataset(
        '/network/datasets/c4/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        '/network/datasets/c4/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_train):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    random.seed(0)
    valenc = []
    for _ in range(nsamples_test):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])

    return trainloader, valenc 