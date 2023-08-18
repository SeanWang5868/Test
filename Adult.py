import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file = open('/Users/seanwang/Desktop/Python/IntroPy/Data/adult.data', 'r')

def chr_int(a):
    if a.isdigit(): return int(a)
    else: return 0

data = []
for line in file:
    data1 = line.split(', ')
    if len(data1) == 15:
        data.append([chr_int(data1[0]), data1[1],
                 chr_int(data1[2]), data1[3],
                 chr_int(data1[4]), data1[5],
                 data1[6], data1[7], data1[8], data1[9],
                 chr_int(data1[10]), chr_int(data1[11]),
                 chr_int(data1[12]), data1[13], data1[14]
                 ])

df = pd.DataFrame(data)
df.columns = ['age', 'type_employer', 'fnlwgt',
              'education', 'education_num', 'marital',
              'occupation','relationship', 'race', 'sex',
              'capital_gain', 'capital_loss', 'hr_per_week',
              'country', 'income'
              ]

counts = df.groupby('country').size()
print(counts.head())

# focus on high-income professionals separated by sex
ml = df[(df.sex == 'Male')]
ml1 = df[(df.sex == 'Male') & (df.income == '>50K\n')]
fm = df[(df.sex == 'Female')]
fm1 = df[(df.sex == 'Female') & (df.income=='>50K\n')]

# what is the proportion of high- income professionals
df1 = df[(df.income=='>50K\n')]
print('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.')
print('The rate of men with high income is: ', int(len(ml1)/float(len(ml))*100), '%.')
print('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.')
# The rate of people with high income is:  24 %.
# The rate of men with high income is:  30 %.
# # The rate of women with high income is:  10 %.


# what the average age of men and women samples in our dataset would be
print('averange men', ml['age'].mean())
print('averange women', fm['age'].mean())
print('averange high-income men', ml1['age'].mean())
print('averange high-income women', fm1['age'].mean())
# averange men 39.43354749885268
# averange women 36.85823043357163
# averange high-income men 44.62578805163614
# averange high-income women 42.125530110262936


# the mean and the variance of hours per week men and women
ml_mu = ml['age'].mean()
fm_mu = fm['age'].mean()
ml_var = ml['age'].var()
fm_var = fm['age'].var()
ml_std = ml['age'].std()
fm_std = fm['age'].std()
print('Statistics of age for men: mu:',
      ml_mu , 'var:', ml_var , 'std:', ml_std)
print('Statistics of age for women: mu:',
      fm_mu , 'var:', fm_var , 'std:', fm_std)

ml_median = ml['age'].median()
fm_median = fm['age'].median()
print("Median age per men and women: ",
      ml_median , fm_median)
ml_median_age = ml1['age'].median()
fm_median_age = fm1['age'].median()
print("Median age per men and women with high-income: ",
      ml_median_age , fm_median_age)
# Median age per men and women:  38.0 35.0
# Median age per men and women with high-income:  44.0 41.0


# Summarizing data by just looking at their mean, median, and variance can be dangerous:
# very different data can be described by the same statistics
# histogram, which is a graph that shows the frequency of each value.
ml_age = ml['age']
ml_age.hist(histtype='stepfilled',bins=20,)
plt.title('Male')
plt.xlabel('Age')
plt.ylabel('Amount')
plt.show()
fm_age = fm['age']
fm_age.hist(histtype='stepfilled',bins=10)
plt.title('Female')
plt.xlabel('Age')
plt.ylabel('Amount')
plt.show()
# bins参数表示将数据分成多少个区间
# density: 如果为 True，则直方图的面积将归一化为 1。去掉之后不影响绘图
# alpha: 条形的透明度。

import seaborn as sns

ml_age.hist(density=0, histtype='stepfilled', alpha=.5, bins=10,
            color=sns.desaturate('indianred', .75))
fm_age.hist(density=0, histtype='stepfilled', alpha=.5, bins=20,)
plt.show()

ml_age.hist(density=1,histtype='stepfilled', alpha=.5, bins=10,
            color=sns.desaturate('indianred', .75))
fm_age.hist(density=1,histtype='stepfilled', alpha=.5, bins=20)
plt.show()
# density= 1是与上图的唯一区别 去掉之后绘出图形与图三相同；如果加上，则绘出正确图形

ml_age.hist(density=1, histtype='step',
            cumulative=True, linewidth=3.5, bins=20)
fm_age.hist(density=1, histtype='step',
            cumulative=True, linewidth=3.5, bins=20,
            color=sns.desaturate('indianred', .75))
plt.show()
# cumulative: 如果为 True，则绘制累积直方图
# density= 1 去掉之后有影响，绘制图形不相同
# The CDF(Cumulative Distribution Function) of the age of working male (in blue) and female (in red) sample

# Outlier Treatment
df2 = df.drop(df.index[
                  (df.income == '>50K\n') &
                  (df['age'] > df['age']. median() + 35) &
                  (df['age'] > df['age']. median() - 15)
                  ])
ml1_age = ml1['age']
fm1_age = fm1['age']

ml2_age = ml1_age.drop(ml1_age.index[
                           (ml1_age > df['age']. median() + 35) &
                           (ml1_age > df['age']. median() - 15)
                           ])
fm2_age = fm1_age.drop(fm1_age.index[
                           (fm1_age > df['age']. median() + 35) &
                           (fm1_age > df['age']. median() - 15)
                           ])
# we focus on the median age (37, in our case) up to 72 and down to 22 years old,
# and we consider the rest as outliers.

mu2ml = ml2_age . mean()
std2ml = ml2_age . std()
md2ml = ml2_age . median()
mu2fm = fm2_age . mean()
std2fm = fm2_age . std()
md2fm = fm2_age . median()
print(" Men statistics :")
print(" Mean :", mu2ml, " Std :", std2ml)
print(" Median :", md2ml)
print(" Min :", ml2_age . min(), " Max :", ml2_age . max())
print(" Women statistics :")
print(" Mean :", mu2fm, " Std :", std2fm)
print(" Median :", md2fm)
print(" Min :", fm2_age . min(), " Max :", fm2_age . max())

plt.figure()
df.age[(df.income == '>50K\n')].plot(alpha=.25, color='blue')
df2.age[(df2.income == '>50K\n')]. plot(alpha=.45, color='red')
plt.show()
# The red shows the cleaned data without the considered outliers (in blue)

# print("The mean difference with outliers is: ")
# print("%4.2f" %(ml_age.mean() - fm_age.mean()))  2.58
# print("The mean difference without outliers is:")
# print("%4.2f" %(ml2_age.mean() - fm2_age.mean()))  2.44
# "%4.2f" % number formats the floating-point number with 4 total characters
# (including digits and decimal point) and 2 digits after the decimal point.

countx, divisionx = np. histogram(ml2_age, density=1)
county, divisiony = np. histogram(fm2_age, density=1)
val = [(divisionx[i] + divisionx[i +1]) /2
       for i in range
       (len(divisionx) - 1)]
plt.plot(val, countx-county, 'o-')
plt.show()
# density: 如果为 True，则直方图的面积将归一化为 1。

def skewness (x):
    res = 0
    m = x. mean()
    s = x. std()
    for i in x:
        res += (i -m) * (i -m) * (i -m)
    res /= (len(x) * s * s * s)
    return res
print("Skewness of the male population is =")
print(skewness(ml2_age))
print("Skewness of the female population is = ")
print(skewness(fm2_age))

def pearson (x):
    return 3*( x. mean () - x. median () )*x. std ()
print(" Pearson’s coefficient of the male population=")
print(pearson(ml2_age))
print(" Pearson’s coefficient of the female population=")
print(pearson(fm2_age))

# After exploring the data, we obtained some apparent effects that seem to support
# our initial assumptions. For example, the mean age for men in our dataset is 39.4
# years; while for women, is 36.8 years. When analyzing the high-income salaries, the
# mean age for men increased to 44.6 years; while for women, increased to 42.1 years.
# When the data were cleaned from outliers, we obtained mean age for high-income
# men: 44.3, and for women: 41.8. Moreover, histograms and other statistics show the
# skewness of the data and the fact that women used to be promoted a little bit earlier
# than men, in general.
