""" Final Project – Draft Report
ALY 6110 Data Management and Big Data, Fall 2021
Module 5 Group Assignment 5

Ankit Yadav, Neil Mascarenhas, Sai Anila Sushma Malladhi

College of Professional Studies, Northeastern University
Prof: Daya Rudhramoorthi 
Oct 20th, 2021

"""

#%%
# import modules
from IPython.core.display import display_png
from IPython.display import display
import numpy as np
import pandas as pd
import re, os, glob
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import seaborn as sns
from scipy import interpolate


# scikit-learn machine learning
# from sklearn.preprocessing import Normalizer, StandardScaler, normalize, scale
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# map functions (https://scitools.org.uk/cartopy/docs/v0.15/)
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
osm_tiles = OSM()

# ordinary least squares
from statsmodels.formula.api import ols

plt.style.use('seaborn-colorblind')

# THis is a special Jupyter Notebook command to prepare the notebook for matplotlib and other libraries
#%matplotlib inline 

# Setting up Pandas paramenters so that we see all the results 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

# Setup for Seaborn
sns.set_style("darkgrid")
#sns.set_context("notebook")


## Read the data, and doing basic fixes.

#%%
# read file(s)
path = os.getcwd()                    
all_files = glob.glob(os.path.join(path+'\\Data\\', "allemployees_*.csv"))  


#%%

# look at the first ten thousand bytes to guess the character encoding
import chardet
with open("Data/allemployees_2020.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
display(result)

#%%
col_names = ['name', 'department', 'title', 'regular', 'retro', 'other', 'overtime', 'injured',\
                'detail', 'quinn', 'total', 'zip']

#enocode encoding = "ISO-8859-5",

df_from_each_file = (pd.read_csv(f, delimiter=',', encoding='utf-8', \
                                 header=0, names=col_names, \
                                 index_col=None).assign(year=f) for f in all_files)  # read year from filename 
EmpEarn   = pd.concat(df_from_each_file, ignore_index=True)
display(EmpEarn.info())


#%%
## Glimpese of the data.
display(EmpEarn.head(5))
display(EmpEarn.isna().sum())


#%%
## The basic cleaning steps we will be performing are:


#%%
# 1. We will extract year from path (filename of earnings report) and assign to new column "year"

## Before
display(EmpEarn.year.head(5))

# extract year from filename
EmpEarn['year'] = EmpEarn['year'].replace({'\D':''}, regex=True) 
EmpEarn['year'] = EmpEarn['year'].str[-4:]  # remove any numbers from file path

## After
display(EmpEarn.year.head(5))


#%%
# 2. Columns "department" and "title" are in reverse order for 2013 and 2014

#EmpEarn.where(EmpEarn['year']=='2021').isin(["department","title"])

# Before
print("Read Carefully")
display(EmpEarn.loc[EmpEarn['year'].isin(['2013'])].filter(items=["department","title"]).head(5))
display(EmpEarn.loc[EmpEarn['year'].isin(['2014'])].filter(items=["department","title"]).head(5))


# switch "department" and "title" columns for 2013 and 2014
EmpEarn.loc[EmpEarn.year.isin(['2013', '2014']),['department','title']] = EmpEarn.loc[EmpEarn.year.isin(['2013', '2014']),['title','department']].values

## After

#EmpEarn.where(EmpEarn['year']=='2021').isin(["department","title"])
print("Read Carefully")
display(EmpEarn.loc[EmpEarn['year'].isin(['2013'])].filter(items=["department","title"]).head(5))
display(EmpEarn.loc[EmpEarn['year'].isin(['2014'])].filter(items=["department","title"]).head(5))

#%%
# 3. Make all Zip code, upto 5 digits Trim set of 5+ digit zipcodes

## Before
display(EmpEarn['zip'].sample(n = 5, random_state=1234))

# ignore "+4" zip codes
EmpEarn['zip'] = EmpEarn['zip'].str[:5]

##
display(EmpEarn['zip'].sample(n = 5, random_state=1234))


#%%
# 4. Zipcodes are 5 digits for some years, 4 for other years and only 4 digits in 2017-20 where the leading "0" has been dropped, Adding the leading zero.

## Before
EmpEarn['zip'] = np.where(EmpEarn['zip'].str.len() == 4, '0' + EmpEarn['zip'], EmpEarn['zip'])

display(EmpEarn['zip'].sample(n = 5, random_state=1234))

#%%
# 5. Missing zipcodes can be filled by comparing to previous year's employee entry

## Before
display(EmpEarn.zip.isna().sum())

# Asding missing zip codes from the previous year data.

EmpEarn.loc[EmpEarn['zip'].str.len() < 4, 'zip'] = np.NaN
EmpEarn['zip'] = EmpEarn.sort_values(by='name')['zip'].fillna(method='ffill')

## After
display(EmpEarn.zip.isna().sum())

#%%
# 6. Convert all numbers to numeric dtype

EmpEarn.dtypes

# converting number strings to numeric dtype
num_cols = ['regular', 'retro', 'other', 'overtime', 'injured', 'detail', 'quinn', 'total']
EmpEarn[num_cols] = EmpEarn[num_cols].replace({'\$': '', ',': ''}, regex=True)\
                                .apply(pd.to_numeric, errors='coerce').fillna(0, axis=1)


## After
EmpEarn.dtypes
#%%
# 7. Replacing , with a <space> in the names

## Before
display("Before",EmpEarn.name.head(3))

EmpEarn.name=EmpEarn.name.apply(lambda x: x.replace(',',' '))
EmpEarn.department=EmpEarn.department.apply(lambda x: x.replace(',',' '))
EmpEarn.title=EmpEarn.title.apply(lambda x: x.replace(',',' '))

## After 
display("After",EmpEarn.name.head(3))


#%%
# 8. Consolidating Depaerment names

# unique job titles and departments by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.bar(EmpEarn['year'].unique(), EmpEarn.groupby('year')['title'].nunique())
ax1.set_title('unique job titles')
ax1.set_ylabel('count')
ax1.set_xlabel('year')

ax2.bar(EmpEarn['year'].unique(), EmpEarn.groupby('year')['department'].nunique())
ax2.set_title('unique department names')
ax2.set_ylabel('count')
ax2.set_xlabel('year')

plt.tight_layout()
plt.show()

#%%

# combine all school departments into a single "Boston Public Schools" department

EmpEarn['dept_clean'] = EmpEarn['department'] # all others to stay the same
EmpEarn['dept_clean'] = np.where(EmpEarn.department.astype(str).str[:3] == \
                                  'BPS', 'Boston Public Schools', EmpEarn.dept_clean)
EmpEarn['dept_clean'] = np.where(EmpEarn.department.astype(str).str[:10] == \
                                  'Asst Super', 'Boston Public Schools', EmpEarn.dept_clean)

bps = ['K-8', 'EEC', 'ELC', 'Middle', 'School', 'Academy', 'Elementary', 'Greenwood', 
       'E Leadership Acad', 'UP Academy Dorchester', 'UP "Unlocking Potential" Acad', 
       'Lyon Pilot High 9-12', 'Ellison/Parks EES', 'Chief Academic Officer', 
       'UP Academy Holland', 'Achievement Gap', 'English Language Learn', 'Haley Pilot',
       'Greater Egleston High', 'Early Learning Services', 'Career & Technical Ed', 
       'Teaching & Learning', 'Unified Student Svc', 'Superintendent',
       'Student Support Svc', 'Harbor High', 'Fam & Student Engagemt', 
       'Enrollment Services', 'Food & Nutrition Svc', 'HPEC: Com Acd Science & Health',
       'Institutional Advancemt', 'Legal Advisor', 'Professional Developmnt', 
       'Chief Operating Officer', 'Research Assess & Eval', 'Info & Instr Technology',
       'BTU Pilot', 'Boston Collaborative High Sch', 'Diplomas Plus', 'Chief Financial Officer']
for school in bps:
    EmpEarn['dept_clean'] = np.where(EmpEarn.department.astype(str).str[-len(school):] == \
                                      school, 'Boston Public Schools', EmpEarn.dept_clean)


#%%
print(set(EmpEarn.dept_clean.loc[EmpEarn.year == '2020'].unique()) - \
      set(EmpEarn.dept_clean.loc[EmpEarn.year == '2019'].unique()))

#%%

"""
Rename various departments to 2020 name:

""" 

dept_names = {'Transportation Department': 'Traffic Division',
              'Dept of Voter Mobilization': 'Election Division',
              'State Boston Retirement Syst': 'Boston Retirement System',
              'Youth Fund': 'Youth Engagement & Employment',
              'Administration and Finance': 'Office of Admin & Finance',
              'Office of Finance & Budget': 'Office of Admin & Finance',
              'Office Of Civil Rights': 'Fair Housing & Equity',
              'Small & Local Business': 'Office of Economic Development',
              'Ofc Chf Public Works Transport': 'Office of Streets',
              'Ofc of Strts, Trnsp & Sani': 'Office of Streets',
              'Property Management': 'Public Facilities Department',
              'Mayor\'s Office-Public Info': 'Mayor\'s Office',
              'Arts & Cultural Development': 'Office of Arts & Culture',
              'Accountability': 'Licensing Board',
              'Women\'s Commission': 'Women\'s Advancement'}
for d in dept_names:
    EmpEarn['dept_clean'] = np.where(EmpEarn['department'].str[-len(d):] == \
                                 d, dept_names[d], EmpEarn.dept_clean)


#%%

# pivot to obtain number of employees in each department by year
departments = pd.pivot_table(EmpEarn, values='name', index='dept_clean',  columns='year', aggfunc='count')\
                    .sort_values('2020', ascending=False).fillna(0)

# spread over 3 sub plots for better visibility
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)
ax1.set_title('Department Size by Year')
departments.iloc[:8].T.plot(ax=ax1, cmap='tab10') # large departments with log scale
ax1.set_yscale('log')
departments.iloc[8:16].T.plot(ax=ax2, cmap='tab20') # medium departments
departments.iloc[16:24].T.plot(ax=ax3, cmap='tab20b') # small departments
ax2.set_ylabel('Number of Employees')
ax1.legend(bbox_to_anchor=(1.01, 1.0))
ax2.legend(bbox_to_anchor=(1.01, 1.0))
ax3.legend(bbox_to_anchor=(1.01, 1.0))
ax3.set_xticklabels(['none', '2011', '2012', '2013', '2014', '2015', '2016'])

plt.tight_layout()
plt.show()



#%%
## Consolidate Job Titles

# police titles:
police_titles = EmpEarn.loc[(EmpEarn.department == 'Boston Police Department')\
                            & (EmpEarn.title.str.startswith('Police O'))\
                            & (EmpEarn.year == '2020')]\
                            ['title'].value_counts()
print(police_titles)


#%%


# consolidate police department titles

EmpEarn['title_clean'] = EmpEarn['title'] # all others to stay the same
pol = {'Police Of': 'Police Officer', 
#         'Police De': 'Police Officer',
        'Police Se': 'Police Sergeant',
        'PoliceSer': 'Police Sergeant',
        'Police Ca': 'Police Captain',
        'Police Li': 'Police Lieutenant'}
for p in pol:
    EmpEarn['title_clean'] = np.where(EmpEarn.title.str[:9] == p, pol[p], EmpEarn.title_clean)

#%%

# Display before and after stats

# most common police department titles before adjustment:
before = EmpEarn['title'][(EmpEarn.department == 'Boston Police Department') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# after adjustment
after = EmpEarn['title_clean'][(EmpEarn.department == 'Boston Police Department') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# combine and sort    
joined = pd.merge(before, after, how='left') \
                        .fillna(0) \
                        .sort_values(by=['title'], ascending=False) \
                        .rename(index=str, columns={'index': 'title', 'title': 'Before', 'title_clean': 'After'})

# data by year
titles_by_year = pd.pivot_table(EmpEarn[EmpEarn['dept_clean'] == 'Boston Police Department'],\
                            values=['title', 'title_clean'], columns='year', aggfunc=pd.Series.nunique) \
                        .sort_values(by='2020', ascending=False)[:10] \
                        .transpose() \
                        .reset_index()

# plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle('Police titles before and after consolidation')

joined.plot.barh('title', 'Before', ax=ax1, width=0.4, color='#2f79b4', position=1).invert_yaxis()
joined.plot.barh('title', 'After', ax=ax1, width=0.4, color='#1adf5a', position=0).invert_yaxis()
ax1.set_title('Police titles consolidation')
ax1.set_xlabel('Count')
ax1.set_ylim(10, -0.8)

titles_by_year.plot('year', 'title', kind='barh', ax=ax2, width=0.4, color='#2f79b4', position=1, label='Before', legend=False)
titles_by_year.plot('year', 'title_clean', kind='barh', ax=ax2, width=0.4, color='#1adf5a', position=0, label='After', legend=False)
ax2.set_title('Police titles by year')
ax2.set_xlabel('year')
ax2.set_ylabel('count unique')

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#%%

# consolidate fire department titles

fd = {'FF': 'Fire Fighter', 
      'Fire Fi': 'Fire Fighter',
      'FireFig': 'Fire Fighter',
      'Fire Ca': 'Fire Captain',
      'Fire L': 'Fire Lieutenant',
      'FireLi': 'Fire Lieutenant',
      'Distric': 'District Fire Chief',
      'Dist Fi': 'District Fire Chief',
      'DistFCh': 'District Fire Chief',
      'Dep Fir': 'Dep Fire Chief',
      'DepFire': 'Dep Fire Chief'}

for k in fd:
    EmpEarn['title_clean'] = np.where(EmpEarn.title.astype(str).str[:len(k)] == k, fd[k], EmpEarn.title_clean)

EmpEarn['title_clean'] = np.where((EmpEarn.title.str[:10] == 'Sr Admin A') \
                                   & (EmpEarn.dept_clean == 'Boston Fire Department'),\
                                   'Sr Admin (Fire)', EmpEarn.title_clean)

fd_titles = EmpEarn.title_clean.loc[(EmpEarn.dept_clean == 'Boston Fire Department')]
print('Number of unique job titles in fire department:', len(set(fd_titles)))
print(fd_titles.value_counts().nlargest(10))


#%%

# Display before and after stats

# most common police department titles before adjustment:
before = EmpEarn['title'][(EmpEarn.department == 'Boston Fire Department') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# after adjustment
after = EmpEarn['title_clean'][(EmpEarn.department == 'Boston Fire Department') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# combine and sort    
joined = pd.merge(before, after, how='left') \
                        .fillna(0) \
                        .sort_values(by=['title'], ascending=False) \
                        .rename(index=str, columns={'index': 'title', 'title': 'before', 'title_clean': 'after'})

# data by year
titles_by_year = pd.pivot_table(EmpEarn[EmpEarn['dept_clean'] == 'Boston Fire Department'],\
                            values=['title', 'title_clean'], columns='year', aggfunc=pd.Series.nunique) \
                        .sort_values(by='2020', ascending=False)[:10] \
                        .transpose() \
                        .reset_index()

# plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle('Fire department titles before and after consolidation')

joined.plot.barh('title', 'before', ax=ax1, width=0.4, color='#1f78b4', position=1).invert_yaxis()
joined.plot.barh('title', 'after', ax=ax1, width=0.4, color='#b2df8a', position=0).invert_yaxis()
ax1.set_title('Fire department titles consolidation')
ax1.set_xlabel('Count')
ax1.set_ylim(10, -0.8)

titles_by_year.plot('year', 'title', kind='barh', ax=ax2, width=0.4, color='#1f78b4', position=1, label='before', legend=False)
titles_by_year.plot('year', 'title_clean', kind='barh', ax=ax2, width=0.4, color='#b2df8a', position=0, label='after', legend=False)
ax2.set_title('Fire department titles by year')
ax2.set_xlabel('year')
ax2.set_ylabel('count unique')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



#%%

library = {'Spec Library Asst': 'Spec Library Asst',
       'Sr Library Asst': 'Sr Library Asst',
       'Librarian': 'Librarian',
       'Special Library': 'Spec Library Asst',
       'Spec Collection L': 'Librarian',
       'Collection Libr': 'Librarian'}
for k in library:
    EmpEarn['title_clean'] = np.where(EmpEarn.title.str.contains(k), library[k], EmpEarn.title_clean)

bpl_titles = EmpEarn.title_clean.loc[(EmpEarn.dept_clean == 'Boston Public Library')]
print('Number of unique job titles in Boston Public Library:', len(set(bpl_titles)))
print(bpl_titles.value_counts().nlargest(20))


#%%

# Display before and after stats

# most common police department titles before adjustment:
before = EmpEarn['title'][(EmpEarn.department == 'Boston Public Library') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# after adjustment
after = EmpEarn['title_clean'][(EmpEarn.department == 'Boston Public Library') \
                                        & (EmpEarn.year == '2020')] \
                                        .value_counts() \
                                        .nlargest(10) \
                                        .reset_index() 
# combine and sort    
joined = pd.merge(before, after, how='left') \
                        .fillna(0) \
                        .sort_values(by=['title'], ascending=False) \
                        .rename(index=str, columns={'index': 'title', 'title': 'before', 'title_clean': 'after'})

# data by year
titles_by_year = pd.pivot_table(EmpEarn[EmpEarn['dept_clean'] == 'Boston Public Library'],\
                            values=['title', 'title_clean'], columns='year', aggfunc=pd.Series.nunique) \
                        .sort_values(by='2020', ascending=False)[:10] \
                        .transpose() \
                        .reset_index()

# plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle('Library department titles before and after consolidation')

joined.plot.barh('title', 'before', ax=ax1, width=0.4, color='#1f78b4', position=1).invert_yaxis()
joined.plot.barh('title', 'after', ax=ax1, width=0.4, color='#b2df8a', position=0).invert_yaxis()
ax1.set_title('Library department titles consolidation')
ax1.set_xlabel('Count')
ax1.set_ylim(10, -0.8)

titles_by_year.plot('year', 'title', kind='barh', ax=ax2, width=0.4, color='#1f78b4', position=1, label='before', legend=False)
titles_by_year.plot('year', 'title_clean', kind='barh', ax=ax2, width=0.4, color='#b2df8a', position=0, label='after', legend=False)
ax2.set_title('Library department titles by year')
ax2.set_xlabel('Year')
ax2.set_ylabel('count unique')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%

# unique job titles and departments by year
title_gb = EmpEarn.groupby('year')[['title', 'title_clean']].nunique()
dept_gb = EmpEarn.groupby('year')[['department', 'dept_clean']].nunique()

# plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
title_gb.plot.bar(ax=ax1, cmap='Dark2', rot=0)
ax1.set_title('Unique job titles')
ax1.set_ylabel('count')
ax1.set_xlabel('year')
ax1.legend(loc=4, frameon=True, framealpha=0.8)

dept_gb.plot.bar(ax=ax2, cmap='Dark2', rot=0)
ax2.set_title('Unique department names')
ax2.set_ylabel('count')
ax2.set_xlabel('year')

#plt.tight_layout()
plt.show()

#%%
























#%%
## Exporting CSV
##We are exporting the dataset to CSV so that we can import it to ProsgresSQL using PgAdmin 4

#BosEmpEarn

EmpEarn.to_csv("Data/EmpEarn.csv", sep=",", header=False, index=False, na_rep='', index_label=None, encoding='utf-8')

#%%



# %%
# unique job titles and departments by year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.bar(EmpEarn['year'].unique(), EmpEarn.groupby('year')['title'].nunique())
ax1.set_title('unique job titles')
ax1.set_ylabel('count')
ax1.set_xlabel('year')

ax2.bar(EmpEarn['year'].unique(), EmpEarn.groupby('year')['department'].nunique())
ax2.set_title('unique department names')
ax2.set_ylabel('count')
ax2.set_xlabel('year')

plt.tight_layout()
plt.show()
# %%























