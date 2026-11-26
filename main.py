
import  pandas  as pd
import  matplotlib.pyplot   as plt

from sklearn.cluster        import  KMeans
from sklearn.preprocessing  import  StandardScaler


df          =   pd.read_csv("data/emp_attrition.csv", encoding='utf-8-sig')

features    =   [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "DistanceFromHome"
]

X           =   StandardScaler().fit_transform(df[features])

inertia     =   {k: KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
    ).fit(X).inertia_ for k in range(1,9)}

df["Cluster"]   =   KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
    ).fit_predict(X)

print(df.groupby("Cluster")[features].mean())
print(df.groupby("Cluster")["Attrition"].apply(lambda s: (s == "Yes").mean()))


summary = df.groupby("Cluster")[features].mean().round(1)
summary["Employees"] = df["Cluster"].value_counts()
summary["Attrition %"] = (df["Attrition"].eq("Yes").groupby(df["Cluster"]).mean() * 100).round(1)
print(summary.to_string())

plt.plot(list(inertia.keys()), list(inertia.values()), marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Within-cluster sum of squares")
plt.title("Elbow method")
plt.show()