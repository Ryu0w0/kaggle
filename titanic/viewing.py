"""
Class for looking around data relationships
"""
# 結論として、
# Pclass,Sex が大きく生存率を分けている特徴
# Age が少し関係ありそうだが男女でfxが変わる

import pandas as pd
import os
from matplotlib import pyplot as plt

os.chdir("D:/Programs/git_repo/kaggle/titanic")
df = pd.read_csv("data/train.csv")


def rel_suv_pclass(df):
    #   クラスが高いほど生存率が高い
    df = pd.DataFrame(data=df, columns=["PassengerId", "Pclass", "Survived"])
    print(df.groupby(by=["Survived", "Pclass"]).count())


def rel_suv_sex(df):
    # 女性のほうが生存率が高いように見える。
    pt = pd.pivot_table(df,
                        index=["Sex"],
                        columns="Survived",
                        values="PassengerId",
                        aggfunc=lambda x: len(x),
                        fill_value=0
                        )
    print("Pivot table to see tha relation of Survived and Sex")
    print(pt)
    pt_percent = pt.apply(lambda x: round(x / sum(x), 2), axis=1)
    print(pt_percent)


def rel_suv_age_sex(df):
    # 女性の生存率は線形的だが男性の生存率は -x^2 のようなグラフを描いている

    # remove null data
    columns = ["PassengerId", "Survived", "Sex", "Age"]
    df = df[columns].query('Age >= 0')
    f_get_age_list = lambda sex, survived: \
        df.query("Sex == @sex and Survived == @survived")["Age"].tolist()

    # The "number" of survivors
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)  # 2行2列分割の1番目
    ax2 = fig.add_subplot(1, 2, 2)
    males_dd, ind, pacthes = ax1.hist(x=f_get_age_list("male", 0), bins=10, range=(0, 100), color="r", alpha=0.3)
    males_al, ind, pacthes = ax1.hist(x=f_get_age_list("male", 1), bins=10, range=(0, 100), color="b", alpha=0.3)
    ax1.set_title("Hist - male")
    ax1.set_xlabel("Ages")
    females_dd, ind, pacthes = ax2.hist(x=f_get_age_list("female", 0), bins=10, range=(0, 100), color="r", alpha=0.3)
    females_al, ind, pacthes = ax2.hist(x=f_get_age_list("female", 1), bins=10, range=(0, 100), color="b", alpha=0.3)
    ax2.set_title("Hist - female")
    ax2.set_xlabel("Ages")

    # The "rate" of survivors
    def calc_survivor_rate(al, dd):
        result = []
        for a, d in zip(al, dd):
            if a == 0:
                result.append(0)
            else:
                result.append(round(a / (a + d), 2) * 100)
        return result

    ax1.scatter(x=[elm + 5 for elm in ind[0:-1]], y=calc_survivor_rate(males_al, males_dd), color="g",
                label="survive rate")
    ax2.scatter(x=[elm + 5 for elm in ind[0:-1]], y=calc_survivor_rate(females_al, females_dd), color="g",
                label="survive rate")
    ax1.legend()
    ax2.legend()
    plt.show()


def rel_sib(df):
    # (乗船している？）配偶者と兄弟の数と生存率の関連はなさそう

    # Pivot
    pv = pd.pivot_table(data=df
                        , values="PassengerId"
                        , index=["Sex", "SibSp"]
                        , columns="Survived"
                        , aggfunc=lambda x: len(x)
                        , fill_value=0)
    print(pv)
    # Prepare data
    male_x = [b for a, b in pv.query("Sex == 'male'")[0].index.values]
    male_al = pv.query("Sex == 'male'")[1].values
    male_dd = pv.query("Sex == 'male'")[0].values
    male_rate = [(a / (a + d)) * 100 for a, d in zip(male_al, male_dd)]

    female_x = [b for a, b in pv.query("Sex == 'female'")[0].index.values]
    female_al = pv.query("Sex == 'female'")[1].values
    female_dd = pv.query("Sex == 'female'")[0].values
    female_rate = [(a / (a + d)) * 100 for a, d in zip(female_al, female_dd)]

    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x=male_x, height=male_al, color="b", alpha=0.4)
    ax1.bar(x=male_x, height=male_dd, color="r", alpha=0.4)
    ax1.scatter(x=male_x, y=male_rate, color="g")
    ax1.set_title("Male")
    ax1.set_xlabel("Number of SibSp")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x=female_x, height=female_al, color="b", alpha=0.4)
    ax2.bar(x=female_x, height=female_dd, color="r", alpha=0.4)
    ax2.scatter(x=female_x, y=female_rate, color="g")
    ax2.set_title("Female")
    ax2.set_xlabel("Number of SibSp")

    plt.show()


def rel_parch(df):
    # (乗船している？)親子の数と生存率の関連性はなさそう
    # Pivot
    pv = pd.pivot_table(data=df
                        , values="PassengerId"
                        , index=["Sex", "Parch"]
                        , columns="Survived"
                        , aggfunc=lambda x: len(x)
                        , fill_value=0)
    print(pv)
    # Prepare data
    male_x = [b for a, b in pv.query("Sex == 'male'")[0].index.values]
    male_al = pv.query("Sex == 'male'")[1].values
    male_dd = pv.query("Sex == 'male'")[0].values
    male_rate = [(a / (a + d)) * 100 for a, d in zip(male_al, male_dd)]

    female_x = [b for a, b in pv.query("Sex == 'female'")[0].index.values]
    female_al = pv.query("Sex == 'female'")[1].values
    female_dd = pv.query("Sex == 'female'")[0].values
    female_rate = [(a / (a + d)) * 100 for a, d in zip(female_al, female_dd)]

    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x=male_x, height=male_al, color="b", alpha=0.4)
    ax1.bar(x=male_x, height=male_dd, color="r", alpha=0.4)
    ax1.scatter(x=male_x, y=male_rate, color="g")
    ax1.set_title("Male")
    ax1.set_xlabel("Number of Parch")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x=female_x, height=female_al, color="b", alpha=0.4)
    ax2.bar(x=female_x, height=female_dd, color="r", alpha=0.4)
    ax2.scatter(x=female_x, y=female_rate, color="g")
    ax2.set_title("Female")
    ax2.set_xlabel("Number of Parch")

    plt.show()


def rel_fare_pclass(df):
    # 大まかにはFareとPclassは連動しているが、外れ値が多い。
    # Pclassだけ採用すればよいと思う。
    # Check the list of pclass
    print(df.groupby(by=["Pclass"]).count()["PassengerId"].index.values)

    data_ = [df.query("Pclass == 1")["Fare"],
             df.query("Pclass == 2")["Fare"],
             df.query("Pclass == 3")["Fare"]]
    plt.boxplot(x=data_)
    plt.ylim([0, 200])
    plt.title("Pclass 1, 2 and 3")
    plt.show()


rel_suv_pclass(df)
rel_suv_sex(df)
rel_suv_age_sex(df)
rel_sib(df)
rel_parch(df)
rel_fare_pclass(df)
