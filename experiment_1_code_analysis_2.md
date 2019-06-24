
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

4. 接下来绘制销售价格与数据集中一些特征是如何相关的
+ 以箱图展示“销售价格”和“总体质量”(OverallQual)间的关系
```
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
```
![箱图展示“销售价格”和“总体质量”(OverallQual)间的关系](./img/box_saleprice_overallqual.jpg)

+ 以箱图展示“销售价格”和“YearBuilt”间的关系
```
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=45);
```
![以箱图展示“销售价格”和“YearBuilt”间的关系](./img/box_saleprice_yearbuilt.jpg)

+ 以散点图展示“销售价格”和“TotalBsmtSF”间的关系
```
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.3, ylim=(0,800000));
```
![以散点图展示“销售价格”和“TotalBsmtSF”间的关系](./img/scatter_saleprice_totalBsmtsf.jpg)

+ 以散点图展示“销售价格”和“LotArea”间的关系
```
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0,800000));
```
![散点图展示“销售价格”和“LotArea”间的关系](./img/scatter_saleprice_lotArea.jpg)

+ 散点图展示“销售价格”和“GrLivArea”间的关系
```
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.3, ylim=(0,800000));
```
![散点图展示“销售价格”和“GrLivArea”间的关系](./img/scatter_saleprice_GrLivArea.jpg)

5. 移除“Id”属性列，因为Id对于每一行数据来说都是独立的，因此对于构建模型没有意思
```
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
train.shape, test.shape
```
![移除“Id”属性列后训练集和测试集的情况](./img/remove_id_train_test_shape.jpg)

##### 特征工程
1. 再次查看 “SalePrice” 的分布情况
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
![“SalePrice” 的分布情况]()

2. 可以看到，“SalePrice”的分布显示向右偏斜。这对于大多数机器学习模型是一个问题，因为大部分的模型对于非正态分布的数据处理的不好。因此，接下来我们使用 **对数函数log(1+x)** 来修正这个偏斜的情况。
```
train["SalePrice"] = np.log1p(train["SalePrice"])
```
现在再将修改了分布的 SalePrice 画图展示。
```
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm, color="b");

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)

plt.show()
```
![正态分布参数]()
![修正后的正态分布图]()

现在，SalePrice已经变成了标准的正态分布，这正是我们想要的。

3. 下面来删除极端值
```
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)
```

4. 分离训练特征和标注值备用
```
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
```


5. 将训练特征和测试特征合并，以便于特征变换阶段的整体管道处理。
```
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape
```
![得到的数据形状]()

6. 填充缺失值
在这里，需要对缺失值的程度设定一个阀值。
```
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]
```
![缺失值情况]()

对缺失值进行可视化展示
```
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
```
![缺失值的可视化展示]()

现在，可以浏览每个属性的缺失情况，并计算相应的缺失值了。

