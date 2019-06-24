## 提交预测结果

1. 将准备预测的内容读取到pd
```
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape
```
![准备预测的内容shape]()

2. 附加混合模型的预测
```
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test)))
```

3. 修复异常值预测
```
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)
```

4. 预测评分
```
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression2.csv", index=False)
```
