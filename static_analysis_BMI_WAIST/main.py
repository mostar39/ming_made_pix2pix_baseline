import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols

dia_csv = pd.read_excel('/home/ylab3/dataset_365/diagnose_BMI/waist_train.xlsx')

before_BMI = dia_csv['Before_BMI'].astype(float)
after_BMI = dia_csv['After_BMI'].astype(float)
before_waist = dia_csv['before_waist'].astype(float)
after_waist = dia_csv['after_waist'].astype(float)

#--------------Paried T Test---------------------------------
stats.ttest_rel(before_BMI,after_BMI)
stats.ttest_rel(before_waist,after_waist)


BMI_t_stat,BMI_p_val = stats.ttest_ind(before_BMI, after_BMI, equal_var=True)
waist_t_stat,waist_p_val = stats.ttest_ind(before_waist, after_waist, equal_var=True)

import seaborn as sns
import matplotlib.pyplot as plt


sns.distplot(before_BMI, label = 'before_BMI')
sns.distplot(after_BMI, label = 'after_BMI')
plt.legend()
plt.savefig('BMI_histogram.png')

sns.distplot(before_waist, label = 'before_waist')
sns.distplot(after_waist, label = 'after_waist')
plt.legend()
plt.savefig('waist_histogram.png')

minus_waist = before_waist - after_waist
sns.distplot(minus_waist)
plt.savefig('minus_boonpo_waist.png')

minus_bmi = before_BMI - after_BMI
sns.distplot(minus_bmi)
plt.legend()
plt.savefig('minus_boonpo_BMI.png')

minus_bmi = list(minus_bmi)
np.mean(minus_bmi)
np.std(minus_bmi)

minus_waist = list(minus_waist)
np.mean(minus_waist)
np.std(minus_waist)

bmi_df = pd.DataFrame({'before_bmi' :before_BMI, 'after_bmi' : after_BMI })
bmi_res = ols('after_bmi ~ before_bmi', bmi_df).fit()
bmi_res.summary()

waist_df = pd.DataFrame({'before_waist' :before_waist, 'after_waist' : after_waist })
waist_res = ols('after_waist ~ before_waist', bmi_df).fit()
waist_res.summary()