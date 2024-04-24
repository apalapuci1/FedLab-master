import torch
import codecs
import numpy as np
import hanlp
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import dill
def load_dense(path):
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            vocab["i2w"].append(vec[0])
            # vocab的length不断增长
            matrix[len(vocab["i2w"])-1, :] = np.array([float(x) for x in vec[1:]])
    for i, w in enumerate(vocab["i2w"]):
        vocab["w2i"][w] = i
    return matrix, vocab, size



matrix, vocab, size = load_dense("./datasets/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5")



df=pd.read_csv(filepath_or_buffer="./datasets/new.csv",encoding='gbk')
y=torch.tensor(df['class'].values,dtype=int)
print(df.head())
print(df.columns)
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

list=[]
for index,row in df.iterrows():
    row['name']=tok(row['name'])
    temp=torch.zeros(15,300)
    for i in range(15):
        if i < len(row['name']):
            if row['name'][i] in vocab['w2i']:
                temp[i]=torch.tensor(matrix[vocab['w2i'][row['name'][i]]])
    list.append(temp)

train_df=torch.stack(list)
dataset=TensorDataset(train_df,y)
train_db, test_db = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
torch.save(train_db,"./datasets/new/train_db.pth")
torch.save(test_db,"./datasets/new/test_db.pth")



