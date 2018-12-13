import gensim
import smart_open
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str)
#parser.add_argument('input2', type=str)
parser.add_argument('--save_model', '-s', default='model', type=str)
args = parser.parse_args()

def read_corpus(fname): #省略できそう
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, min_len=1), [i])

def make_model():
    train_corpus = list(read_corpus(args.input))
    print(train_corpus)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=5, epochs=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(args.save_model)

def near(modelf):
    model = gensim.models.Doc2Vec.load(modelf)
    res = model.most_similar("妻")
    print(res)
    
if __name__ == "__main__":
    make_model()
    near(args.save_model)