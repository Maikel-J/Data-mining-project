# importing the required module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
# import statsmodels.api as sm

# https://colab.research.google.com/github/meizmyang/Student-Performance-Classification-Analysis/blob/master/Student%20Performance%20Analysis%20and%20Classification.ipynb#scrollTo=IdD7SgZoHjd9
mat = pd.read_csv("Dataset/student-mat.csv", sep=';') # load dataset of the math classes 

mat.columns = ['school','sex','age','address','familySize','parentsStatus','motherEducation','fatherEducation',
           'motherJob','fatherJob','reason','guardian','commuteTime','studyTime','failures','schoolSupport',
          'familySupport','paidClasses','activities','nursery','desireHigherEdu','internet','romantic','familyQuality',
          'freeTime','goOutFriends','workdayAlc','weekendAlc','health','absences','1stPeriod','2ndPeriod','final']
 
mat['finalGrade'] = 'None'
mat.loc[(mat.final >= 15) & (mat.final <= 20), 'finalGrade'] = 'good' 
mat.loc[(mat.final >= 10) & (mat.final <= 14), 'finalGrade'] = 'fair' 
mat.loc[(mat.final >= 0) & (mat.final <= 9), 'finalGrade'] = 'poor' 
mat.head(5)


df = pd.DataFrame(mat)
df.plot()
plt.show()