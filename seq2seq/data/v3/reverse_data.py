import pandas as pd

a = "你 是 谁"
b = a.split()
b.reverse()
print(b)


def reverse_str(s):
    s_list = s.split()
    s_list.reverse()
    return " ".join(s_list)


def reverse_data(raw_path, reverse_path):
    df = pd.read_csv(raw_path, header=None, encoding="utf-8", names=["raw"])
    df["reverse"] = df.raw.map(lambda x: reverse_str(x))
    df["reverse"].to_csv(reverse_path, encoding="utf-8", header=False, index=False)


if __name__ == "__main__":
    reverse_data("v2/train.src", "v3/train.src")
    reverse_data("v2/train.tgt", "v3/train.tgt")
    reverse_data("v2/dev.src", "v3/dev.src")
    reverse_data("v2/dev.tgt", "v3/dev.tgt")
    reverse_data("v2/test.src", "v3/test.src")
    reverse_data("v2/test.tgt", "v3/test.tgt")
