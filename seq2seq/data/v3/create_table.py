# -*- coding:utf-8 -*-
import numpy as np

# a = np.load("v3/table.npy")
# print(a)


# b = np.load("seq2seq/data/v3/vocab_dict.npy").item()
# c = np.load("seq2seq/data/v3/large_table.npy")
#
# with open("seq2seq/data/v3/vocab.tgt", "w") as f:
#     f.write("\n".join(b.keys()))
# print(b)
# print(c.shape)
#
# dic = {'i': [0, 2811], 'u': [2812, 4190], 'v': [4191, 4675], 'a': [4676, 5749], 'o': [5750, 6559], 'e': [6560, 8180],
#        'ai': [8181, 9165], 'ei': [9166, 10149], 'ao': [10150, 11533], 'ou': [11534, 12494],
#        'an': [12495, 15206], 'n': [15207, 16002], 'in': [16003, 16485], 'un': [16486, 16485],
#        'vn': [16486, 16515], 'ang': [16516, 17945], 'ing': [17946, 18745], 'eng': [18746, 19283],
#        'ong': [19284, 19914], 'er': [19915, 19999]}
# d = np.asarray(list(dic.values()))
# t = d + 3
# print(d)
# e = [[t[i]] * (t[i][1] - t[i][0] + 1) for i in range(t.shape[0])]
# e = np.concatenate(e)
# first = [[0, 0], [1, 1], [2, 2]]
# res = np.concatenate([first, e])
# print(res.shape)

# table = np.load("seq2seq/data/v3/large_table.npy")
# print(table.shape)
# e = [[20003, 29999]] * 9997
# res = np.concatenate([table, e])
# print(res.shape)
# np.save("seq2seq/data/v3/table.npy", res)
# print(np.load("seq2seq/data/v3/table.npy"))

table = np.load("v3/large_table.npy")
print(table.shape)
t = [[20003, 23441]] * 3439
res = np.concatenate([table, t])
print(res.shape)
np.save("v3/table_23442.npy", res)
