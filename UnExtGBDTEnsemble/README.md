### 1 运行环境
 - 操作系统：Linux
 - 编程语言：[Python2.7](https://www.python.org/)
 - 第三方Python库：[Numpy](http://www.numpy.org/) &ensp; [Scikit-learn](http://scikit-learn.org/stable/) &ensp; [XGBoost](https://xgboost.readthedocs.io/en/latest/)

### 2 模型源程序目录及文件说明
|--code：源程序目录  
| &ensp;|-- [fold5_sample.py](code/fold5_sample.py)：采样产生五次五折交叉验证数据的源程序  
| &ensp;|-- [unExtGBDTEnsemble.py](code/unExtGBDTEnsemble.py)：基于Ext-GBDT集成的类别不平衡信用评分模型源程序  
| &ensp;|-- [otherModelForComparison.py](code/otherModelForComparison.py)：论文中提到的所有对比模型的源程序  
|  
|--data：数据目录  
| &ensp; |-- src_data：原始数据目录  
| &ensp; | &ensp; |-- german_data_numeric.csv：UCI德国信用评分数据集  
| &ensp; |  
| &ensp; |-- fold5_data：五次五折交叉验证数据目录  
| &ensp;&ensp;&ensp;&ensp; |-- good：好客户数据目录：  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- good_fold1.csv：第1折好客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- good_fold2.csv：第2折好客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- good_fold3.csv：第3折好客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- good_fold4.csv：第4折好客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- good_fold5.csv：第5折好客户数据  
| &ensp;&ensp;&ensp;&ensp; |  
| &ensp;&ensp;&ensp;&ensp; |-- bad：坏客户数据目录  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- bad_fold1.csv：第1折坏客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- bad_fold2.csv：第2折坏客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- bad_fold3.csv：第3折坏客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- bad_fold4.csv：第4折坏客户数据  
| &ensp;&ensp;&ensp;&ensp; | &ensp; |-- bad_fold5.csv：第5折坏客户数据  
|  
|--model：训练模型时产生的二进制模型文件保存目录  

### 3 模型运行  
- 产生五次五折交叉验证数据  
```shell
>> cd code
>> python fold5_sample.py
```
- 运行基于Ext-GBDT集成的类别不平衡信用评分模型
```shell
>> cd code
>> python unExtGBDTEnsemble.py
```
- 运行其他对比模型
```shell
>> cd code
>> python otherModelForComparison.py
```


