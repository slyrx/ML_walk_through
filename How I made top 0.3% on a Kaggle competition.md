### 目标
---
1. 我们的目标是通过给出的特征预测房价。
2. 预测值和实际值首先取log对数，其次再对两者使用均方根差RMSE进行评估。
3. 将RMSE误差转换为log度量可以确保预测结果中，高房价和低房价误差对预测得分的影响是相同的。

### 模型训练中的关键步骤
---
1. 交叉验证: 本例中使用12折交叉验证
2. 模型: 每轮交叉验证都使用了7个模型
3. Stacking融合, 使用xgboost做第二阶段的预测
4. Blending融合

### 使用模型
1. ridge
2. svr
3. gradient boosting
4. random forest
5. xgboost
6. lightgbm regressors

### 文献综述
Blending 混合
