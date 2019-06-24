
## EDA
+ 使用给出的特征对房价做出预测
---

##### 查看前5条数据记录
```
train.head()
```

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
![训练集中预测值的分布情况](https://www.kaggleusercontent.com/kf/15425540/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..pNKpx8_CeoQisuhcwZ4Yeg.FXDneIjyHdX--MbtjlXjs2ndp02Me5hADRWRp5T2kgeWkpJBJFgGnzj5A-nEohAHXScM6QHWl3GO9Aek69pPAL9xwZHXgr2EAjLmqEl7ApM4Ov8Kc5DipP5QArNRg3H7ryuX1jIo3XL0l6U0kzJ10saOX8TaaGIq6ozK7q_YuXHpRZJTsC5D8646FzBk7WYt.rbzFz2nnm-Y3X_8onDyPDQ/__results___files/__results___10_0.png)
