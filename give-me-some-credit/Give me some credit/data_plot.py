import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('data/cs-training.csv')
data_test_a = pd.read_csv('data/cs-test.csv')

# 解决Seaborn中文显示问题并调整字体大小
sns.set(font='SimHei')
data_train['loanAmnt'].value_counts()
#data_train['loanAmnt'].value_counts().plot.hist()
plt.figure(figsize=(16,12))
plt.subplot(221)
sub_plot_1=sns.distplot(data_train['loanAmnt'])
sub_plot_1.set_title("训练集", fontsize=18)
plt.subplot(222)
sub_plot_2=sns.distplot(data_test_a['loanAmnt'])
sub_plot_2.set_title("测试集", fontsize=18)