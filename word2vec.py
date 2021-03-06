import MeCab
import numpy as np
import logging  #途中経過のログ出力
import argparse
from gensim.models import word2vec

def makeModel(wakatiPath,modelPath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(wakatiPath)
    """
    ベクトルの次元数(size)
        大規模(Wikipediaなど・ユニークな語彙数：数万)：数百次元
        中規模(新聞などの数カテゴリな文書・ユニークな語彙数：数千)：50〜200次元
        小規模(あるカテゴリの文書・ユニークな語彙数：数百)：10〜50次元
    window幅：日本語ならデータセットに依存せず5くらい。段落単位くらいの広い解釈をしたいならば15
    学習率(alpha)：0.025(デフォルト)
    微妙な結果が得られた際の対応
        単語間の類似度が低い：次元数増やす
        学習率：減らす
        これでもうまくいかない：エポック数増やす
    """
    # size=次元数、min_count=単語の最小出現数、window=上記に説明、iter=学習繰り返し、cbow_mean=ベクトルの和(0)か平均か(1)
    model = word2vec.Word2Vec(sentences, size=600,alpha=0.01 , min_count=5, window=15,cbow_mean=1,iter=10)
    model.save(modelPath)

def nearWord(modelPath,s_word):
    model = word2vec.Word2Vec.load(modelPath)
    results = model.wv.most_similar(s_word,topn=20)
    print("検索語：",s_word)
    for result in results:
        print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--out', '-o' , type=str)
    parser.add_argument('--search', '-s' , type=str)
    parser.add_argument('--s_word', '-w' , type=str)
    args = parser.parse_args()
    
    if args.search:
        nearWord(args.search,args.s_word)
    else:
        makeModel(args.input,args.out)
        #nearWord(args.out)
        print("にゃーん")
