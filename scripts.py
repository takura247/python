import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


application = pd.read_csv('../all/application_train.csv')

application['TARGET'].value_counts()


app1 = application.loc[application['TARGET'] == 1]
#app1['AMT_CREDIT'].value_counts()
app1['AMT_CREDIT'].astype(int).plot.hist()



app0 = application.loc[application['TARGET'] == 0]
#app0['AMT_CREDIT'].value_counts()
app0['AMT_CREDIT'].astype(int).plot.hist()


correlations = application.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


with PdfPages('amount_credit.pdf') as pdf:
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['AMT_CREDIT'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'AMT_CREDIT'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'AMT_CREDIT'], label = 'target == 1')
    plt.xlabel('Credit amout of the loan'); plt.ylabel('Desity'); plt.title('Distribution of credit amout of the loan');
    pdf.savefig()
    plt.close()


with PdfPages('others_features_distribution.pdf') as pdf:
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['OWN_CAR_AGE'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'OWN_CAR_AGE'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'OWN_CAR_AGE'], label = 'target == 1')
    plt.xlabel('Own car age'); plt.ylabel('Desity'); plt.title('Distribution of own car age');
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['AMT_INCOME_TOTAL'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'AMT_INCOME_TOTAL'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'AMT_INCOME_TOTAL'], label = 'target == 1')
    plt.xlabel('Amount income total'); plt.ylabel('Desity'); plt.title('Distribution of amount income total');
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['CNT_CHILDREN'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'CNT_CHILDREN'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'CNT_CHILDREN'], label = 'target == 1')
    plt.xlabel('Count children'); plt.ylabel('Desity'); plt.title('Distribution of the number of children in the family');
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['EXT_SOURCE_3'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'EXT_SOURCE_3'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'EXT_SOURCE_3'], label = 'target == 1')
    plt.xlabel('External source 3'); plt.ylabel('Desity'); plt.title('Distribution of external source 3');
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['EXT_SOURCE_2'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'EXT_SOURCE_2'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'EXT_SOURCE_2'], label = 'target == 1')
    plt.xlabel('External source 2'); plt.ylabel('Desity'); plt.title('Distribution of external source 2');
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize = (10,8))
    tmp = application.dropna(subset=['EXT_SOURCE_1'])
    sns.kdeplot(tmp.loc[tmp['TARGET'] ==0, 'EXT_SOURCE_1'], label = 'target == 0')
    sns.kdeplot(tmp.loc[tmp['TARGET'] == 1, 'EXT_SOURCE_1'], label = 'target == 1')
    plt.xlabel('External source 1'); plt.ylabel('Desity'); plt.title('Distribution of external source 1');
    pdf.savefig()
    plt.close()
