import hanlp
import pandas as pd

df=pd.read_csv(filepath_or_buffer="./datasets/1.csv",encoding='gbk')
print(df.head())
print(df.columns)
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

for index,row in df.iterrows():
    row['name']=tok(row['name'])
    print(row['name'])

df.to_csv("./datasets/2.csv")

