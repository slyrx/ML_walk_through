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

### 实践前提
#### 引入库
---
```python
import numpy as np
import pandas as pd
import datetime
import random

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
```
#### 模型设置
---
```
pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os
print(os.listdir("../input/kernel-files"))
```

#### 读取数据
---
```
# Read in the dataset as a dataframe
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape, test.shape
```

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
