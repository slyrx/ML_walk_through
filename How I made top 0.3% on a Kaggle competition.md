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
4. Blending融合, 

### 模型表现分析
从模型比较图标中可以看出，blended混合模型的表现最好，成绩为0.075。在此作为最终的预测模型使用。
![model_training_advanced_regression](https://www.kaggleusercontent.com/kf/15425540/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..pNKpx8_CeoQisuhcwZ4Yeg.FXDneIjyHdX--MbtjlXjs2ndp02Me5hADRWRp5T2kgeWkpJBJFgGnzj5A-nEohAHXScM6QHWl3GO9Aek69pPAL9xwZHXgr2EAjLmqEl7ApM4Ov8Kc5DipP5QArNRg3H7ryuX1jIo3XL0l6U0kzJ10saOX8TaaGIq6ozK7q_YuXHpRZJTsC5D8646FzBk7WYt.rbzFz2nnm-Y3X_8onDyPDQ/__results___files/__results___2_0.png)

### [实践前提](https://github.com/slyrx/ML_walk_through/edit/master/experiment_1_code_analysis.md)

### 使用模型
1. ridge
2. svr
3. gradient boosting
4. random forest
5. xgboost
6. lightgbm regressors

### 文献综述
Blending 混合
+ 提升预测记过的鲁棒性。
