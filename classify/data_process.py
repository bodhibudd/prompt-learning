import pandas as pd
from sklearn.utils import shuffle

def get_csv():
    datas = pd.read_csv("data/ChnSentiCorp_htl_all.csv")
    datas = shuffle(datas)
    train_datas = datas[:6000]
    dev_datas = datas[6000:]
    train_datas["review"] = train_datas["review"].str[:500]
    dev_datas["review"] = dev_datas["review"].str[:500]

    train_datas.to_csv("data/train_data.csv", index=False)
    dev_datas.to_csv("data/dev_data.csv", index=False)
get_csv()



