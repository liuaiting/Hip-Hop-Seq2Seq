import pandas as pd

src_df = pd.read_csv("x.txt", header=None, names=["src"])
tgt_df = pd.read_csv("y.txt", header=None, names=["tgt"])
pair_df = pd.concat([src_df, tgt_df], axis=1)
pair_df = pair_df.sample(frac=1).reset_index(drop=True)

train_df = pair_df.iloc[:-200, :]
dev_df = pair_df.iloc[-200: -100, :]
test_df = pair_df.iloc[-100:, :]
print(train_df.head())
print(dev_df.head())
print(test_df.head())
train_df["src"].to_csv("train.src", encoding='utf-8', header=False, index=False)
train_df["tgt"].to_csv("train.tgt", encoding='utf-8', header=False, index=False)
dev_df["src"].to_csv("dev.src", encoding='utf-8', header=False, index=False)
dev_df["tgt"].to_csv("dev.tgt", encoding='utf-8', header=False, index=False)
test_df["src"].to_csv("test.src", encoding='utf-8', header=False, index=False)
test_df["tgt"].to_csv("test.tgt", encoding='utf-8', header=False, index=False)
