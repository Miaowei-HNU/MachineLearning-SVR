import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import seaborn as sns
data_file="./data/汇总数据（每日）.xlsx"
data=pd.read_excel(data_file)
new_datafile="./data/new_data.xlsx"
#提取目标的列
targetCol = data.loc[:, ['3412(进水COD)','3413(进水NH3)','3781(1系列A池3好氧)','3430（出水COD）','3431(出水NH3)']]

#解决中文乱码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#产生新的变量 x1='3412(进水COD)'/'3430（出水COD）';x2='3413(进水NH3)'/'3431(出水NH3)';target='3781(1系列A池3好氧)'
new_data=pd.DataFrame()
new_data['x1']=targetCol['3412(进水COD)']/targetCol['3430（出水COD）']
new_data['x2']=targetCol['3413(进水NH3)']/targetCol['3431(出水NH3)']
new_data['target']=targetCol['3781(1系列A池3好氧)']
new_data.to_excel(new_datafile,sheet_name="Sheet1")
new_data.info()

#查看数据的分布情况
plt.figure(figsize=(20,6))
plt.subplot(311)
sns.distplot(new_data['x1'])
plt.subplot(312)
sns.distplot(new_data['x2'])
plt.subplot(313)
sns.distplot(new_data['target'])
plt.show()

#切分数据集
train=pd.DataFrame()
train=new_data.loc[:,['x1','x2']]
target=new_data['target']
train.info()
target.info()

#训练数据的异常值分析
plt.figure(figsize=(12,6))
plt.boxplot(x=train.values,labels=train.columns)
plt.show()

#切分数据 训练数据为80% 验证数据为20%
train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.2,random_state=0)



#开始训练---------------rbf--------------算子
svr_lin = SVR(kernel="linear", C=100, gamma="auto")#构建SVR模型训练器

#记录训练时间
t0 = time.time()

#训练

svr_lin.fit(train_data,train_target)
svr_fit = time.time() - t0
t0 = time.time()

#测试

lin_target_predict  = svr_lin.predict(test_data)

# 计算性能指标以及保存结果
plt.figure(figsize=(20,10))
x=range(73)

#计算绝对误差 MAE

lin_MAE=mean_absolute_error(test_target, lin_target_predict)

#计算均方误差 MSE

lin_MSE=mean_squared_error(test_target, lin_target_predict)

#计算均方根误差 RMSE

lin_RMSE=np.sqrt(mean_squared_error(test_target,lin_target_predict))

#保存结果到./result/predict.xlsx文件
save_result=pd.DataFrame()
save_result['真实值']=test_target

save_result['liner模型预测值']=lin_target_predict
save_result.to_excel('./result/predict.xlsx')

#开始画图

plt.scatter(x,lin_target_predict,alpha=0.8,c='r',label='liner模型预测值')
plt.scatter(x,test_target,alpha=0.8,c='b',label='实际值')
plt.legend()
plt.xlabel("data")
plt.ylabel("target")
plt.title('预测值和真实值的输出结果')
plt.show()


print("linear模型绝对误差MSE={}".format(lin_MSE))

print("linear模型绝对误差MAE={}".format(lin_MAE))

print("linear模型绝对误差RMSE={}".format(lin_RMSE))