
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

##### 深度挖掘特征
首先对数据的特征进行可视化
1. 提取出数据中为数值型的特征
```
# Finding numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)     
```
2. 以散点图对内容进行展示
```
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    # 以散点图对内容进行展示
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()
```
![Features: a deep dive](./img/feature_data_scatter_viewer.jpg)

3. 再通过**train.corr() 特征关系函数**绘制出这些特征之间的关系，以及特征和预测价格之间的关系。
```
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
```
![特征关系](./img/feature_corr.jpg)
