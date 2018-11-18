import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    os.chdir("D:/Programs/git_repo/kaggle/titanic")
    df = pd.read_csv("data/train.csv")
    train_list = ["Pclass", "Sex", "Age", "Survived"]
    print_distinct(train_list, df)

    # Fill Null of age with mean
    print("mean of age is:", str(df["Age"].mean()))
    df = df[train_list].fillna({"Age": round(df["Age"].mean(), 0)})

    # Change sex value into boolean 0: male, 1: female
    df = df.replace({"male": 0, "female": 1})

    #
    # features = df[["Pclass", "Sex", "Age"]]
    features = df[["Pclass", "Sex"]]
    labels = df["Survived"]

    # split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=42)

    return features_train, labels_train, features_test, labels_test


def print_distinct(labels, df_):
    for label in labels:
        result = []
        l_data = df_[label].tolist()
        for elm in l_data:
            if elm not in result:
                result.append(elm)
        print("Distinct {} : {}".format(label, result))


# For debug
if __name__ == "__main__":
    os.chdir("D:/Programs/git_repo/kaggle/titanic")
    df = pd.read_csv("data/train.csv")
    train_list = ["Pclass", "Sex", "Age", "Survived"]
    print_distinct(train_list, df)

    # Fill Null of age with mean
    print("mean of age is:", str(df["Age"].mean()))
    df = df[train_list].fillna({"Age": round(df["Age"].mean(), 0)})

    # Change sex value into boolean 0: male, 1: female
    df = df.replace({"male": 0, "female": 1})

