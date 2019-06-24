
## EDA
+ 使用给出的特征对房价做出预测
---

##### 查看前5条数据记录
```
train.head()
```
![](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCbS3CUeibENjz7DAGKZaIx8ia3bad78Y2oZz8XQTMF1h2szXjM2lDADx5F7DZGsJESMZSgITawyqyQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

##### 观察训练集中预测值的分布情况
```
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
```
![训练集中预测值的分布情况](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtCbS3CUeibENjz7DAGKZaIx8m01oqgjyzVpcpicR5icNFCZJab7ZoGoGQayK1icujFHnU3eSiaA1AcGljw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

##### 显示正态分布的歪斜(Skewness)和峭度(Kurtosis)
```
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
```
![正态分布的歪斜(Skewness)和峭度(Kurtosis)](./img/skewness_kurtosisi.png)
