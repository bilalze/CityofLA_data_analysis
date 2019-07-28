import os
import pandas as pd
from IPython.display import display
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from collections import Counter
nltk.download('punkt')
from scipy.stats import linregress
from wordcloud import WordCloud,STOPWORDS


path="C:/Users/twist/CityofLA/Job Bulletins/"
def to_dataframe(path):
    """"function to extract features from job bulletin text files and convert to
    pandas dataframe.
    function take one argument
                        1.path of the files
                                                        """
    bulletins=os.listdir(path)
    df = pd.DataFrame(
        columns=['File Name', 'Position', 'salary_start', 'salary_start_alt', 'salary_end_alt', 'salary_end',
                 'opendate', 'requirements', 'duties', 'deadline'])

    opendate = re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')  # match open date

    salary = re.compile(r'(\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?((\s(and)\s)(\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+)))?')  # match salary

    requirements = re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')  # match requirements

    for no in range(num):
        with open("C:/Users/twist/CityofLA/Job Bulletins/"+ bulletins[no]) as f:  # reading files
            try:
                # headings=get_headings(no)
                file = f.read().replace('\t', '')
                data = file.replace('\n', '')
                headings = [heading for heading in file.split('\n') if
                            heading.isupper()]  ##getting heading from job bulletin

                try:
                   date = datetime.strptime(re.search(opendate, data).group(3), '%m-%d-%y')
                except Exception as e:
                   # date=datetime.now()
                   # date=date.strftime('%m-%d-%y')
                    print(no)
                try:
                   sal = re.search(salary, data)
                except Exception as e:
                   print(bulletins[no])
                try:
                    req = re.search(requirements, data).group(2)
                except Exception as e:
                    # print(e)
                    #print(re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',data))
                    req = re.search('(.*)NOTES?', re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',
                                                             data)[0][1][:1200]).group(1)

                duties = re.search(r'(DUTIES)(.*)(REQ[A-Z])', data).group(2)
                try:
                    enddate = re.search(
                        r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})'
                        , data).group()
                except Exception as e:
                    enddate = np.nan

                selection = [z[0] for z in re.findall('([A-Z][a-z]+)((\s*\.\s\.)+)', data)]  ##match selection criteria
                if  sal.group(9) is None:
                    saler=np.nan
                else:
                    saler=str(sal.group(9))
                if  sal.group(13) is None:
                    saler2=np.nan
                else:
                    saler2=str(sal.group(13))

                df = df.append(
                    {'File Name': bulletins[no], 'Position': headings[0].lower(), 'salary_start': sal.group(1),'salary_start_alt': saler,
                     'salary_end': sal.group(5),'salary_end_alt': saler2, "opendate": date, "requirements": req, 'duties': duties,
                     'deadline': enddate, 'selection': selection}, ignore_index=True)
            except Exception as e:
                #print(no)
                print('umatched sequence'+bulletins[no])
    reg = re.compile(
        r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\s(years?)\s(of\sfull(-|\s)time)')
    df['EXPERIENCE_LENGTH'] = df['requirements'].apply(
        lambda x: re.search(reg, x).group(1) if re.search(reg, x) is not None else np.nan)
    df['FULL_TIME_PART_TIME'] = df['EXPERIENCE_LENGTH'].apply(
        lambda x: 'FULL_TIME' if x is not np.nan else np.nan)

    reg = re.compile(
        r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\s|-)(years?)\s(college)')

    df['EDUCATION_YEARS'] = df['requirements'].apply(
        lambda x: re.search(reg, x).group(1) if re.search(reg, x) is not None else np.nan)
    df['SCHOOL_TYPE'] = df['EDUCATION_YEARS'].apply(
        lambda x: 'College or University' if x is not np.nan else np.nan)
    reg2 = re.compile(
        r"(semester|Semester|quarter units|college|College|university|University|Bachelor's|bachelor's|Bachelor|bachelor|Master's|master's)")
    for index, row in df.iterrows():
        if row['SCHOOL_TYPE'] is np.nan:
            if re.search(reg2, row['requirements']) is not None:
                df.loc[index, 'SCHOOL_TYPE'] = 'College or University'

    return df
df=to_dataframe(path)

plt.figure(figsize=(8,5))
text=''.join(job for job in df['Position'])                                ##joining  data to form text
text=nltk.tokenize.word_tokenize(text)
job_class_final=list() ##counting number of occurences
checker=text.copy()
for b in range(20):
    jobs = Counter(checker)
    jobs_class=[job for job in jobs.most_common() if len(job[0])>3]          ##selecting most common words
    job_class_final.append(jobs_class[0])
    #print(jobs_class)
    black=list()
    for y in checker:
        if y.find(jobs_class[0][0])== -1:
            black.append(y)
    checker=black


a,b=map(list, zip(*job_class_final))
sns.barplot(b,a,palette='RdBu')                                           ##creating barplot
plt.title('Job sectors')
plt.xlabel("count")
plt.ylabel('sector')
plt.show()
df.to_csv('job class output.csv')
df['salary_start']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_start']  ]
df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_start']]
df['salary_end']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_end']  ]
df['salary_end']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_end']]

df['salary_start_alt']=[str(pal).replace('$','')  if pal is not None else 0 for pal in df['salary_start_alt']  ]
df['salary_start_alt']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int and sal!="nan" else 0  for sal in df['salary_start_alt']]
df['salary_end_alt']=[str(sal).replace('$','')  if sal is not None and not float() else 0 for sal in df['salary_end_alt']  ]
df['salary_end_alt']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int  and sal!="nan" else 0 for sal in df['salary_end_alt']]

plt.figure(figsize=(7,5))
sns.distplot(df['salary_start'])
plt.title('salary distribution')
plt.show()
most_paid=df[['Position','salary_start']].sort_values(by='salary_start',ascending=False)[:10]
plt.figure(figsize=(7,5))
sns.barplot(y=most_paid['Position'],x=most_paid['salary_start'],palette='RdBu')
plt.title('Best paid jobs in LA')
plt.show()
test=pd.DataFrame(columns=['salary_diff','salary_diff_alt'])
test['salary_diff_alt']=abs(df['salary_start_alt']-df['salary_end_alt'])
test['salary_diff']=abs(df['salary_start']-df['salary_end'])
df['salary_diff']=[row['salary_diff'] if row['salary_diff']> row['salary_diff_alt']  else row['salary_diff_alt'] for index, row in test.iterrows()  ]
ranges=df[['Position','salary_diff']].sort_values(by='salary_diff',ascending=False)[:10]
plt.figure(figsize=(7,5))
sns.barplot(y=ranges['Position'],x=ranges['salary_diff'],palette='RdBu')
plt.title("Highest deviation")
plt.show()
display(ranges)
df['year_of_open']=[date.year for date in df['opendate']]

count=df['year_of_open'].value_counts(ascending=True)
years=['2020','2019','2018', '2017', '2016', '2015', '2014', '2013', '2012', '2008', '2006',
           '2005', '2002', '1999']
plt.figure(figsize=(7,5))
plt.plot([z for z in reversed(years)],count.values,color='blue')

plt.title('Oppurtunities over years')
plt.xlabel('years')
plt.ylabel('count')
plt.gca().set_xticklabels([z for z in reversed(years)],rotation='45')
plt.show()

years2=df[['Position','year_of_open']].copy()

years2=years2.sort_values(by='year_of_open',ascending=False)
for index,row in years2.iterrows():
    for x,y in job_class_final:
        if row['Position'].find(x)!=-1:
            years2.loc[index, 'sector'] =x
            break
        else:
            years2.loc[index, 'sector'] =np.nan


plt.figure(figsize=(7, 5))
sectory=list()
slopy=list()
for x,y in job_class_final[:][:10] :
    #ploter=[int(row['year_of_open']) if row['sector']==x else np.nan for index,row in years2.iterrows()]
    plotter = years2[years2['sector'] == x]
    plotter=plotter.groupby(['year_of_open']).size().reset_index(name='Size')

    counter=list()
    display(plotter.dtypes)
    for ye in years:
        sizer=len(counter)
        for index,row in plotter.iterrows():

          if int(ye) == int(row['year_of_open']) :
              counter.append(row['Size'])
              break

        if sizer==len(counter):
            counter.append(0)
    sloperr, intercept, r_value, p_value, std_err = linregress(list(map(int, years[1:11][::-1])), counter[1:11][::-1])
    print(sloperr)
    slopy.append(sloperr)
    sectory.append(x)
    #count = plotter['year_of_open'].value_counts(ascending=True)
    plt.plot([z for z in reversed(years)], counter[::-1],label=x)
sloper=pd.DataFrame({'sector':sectory})
sloper['slope']=slopy
print(sloper)
plt.title('Oppurtunities over years')
plt.xlabel('years')
plt.ylabel('count')
plt.gca().set_xticklabels([z for z in reversed(years)],rotation='45')
plt.legend(loc='upper left')
plt.show()
plt.figure(figsize=(7,5))
sloper=sloper.sort_values(by='slope',ascending=False)
sns.barplot(y=sloper['sector'],x=sloper['slope'],palette='RdBu')
plt.title('fastest growing job sector as of 2018')
plt.show()
experience=df['EXPERIENCE_LENGTH'].value_counts().reset_index()
experience['index']=experience['index'].apply(lambda x : x.lower())
experience=experience.groupby('index',as_index=False).agg('sum')
labels=experience['index']
sizes=experience['EXPERIENCE_LENGTH']
plt.figure(figsize=(5,7))
plt.pie(sizes,explode=(0, 0.1, 0, 0,0,0,0),labels=labels)
plt.gca().axis('equal')
plt.title('Experience value count')
plt.show()
x1=df['SCHOOL_TYPE'].value_counts()[0]
x2=df['FULL_TIME_PART_TIME'].value_counts()[0]
plt.figure(figsize=(5,5))
plt.bar(height=[x1,x2],x=['College Degree','Experience'])
plt.show()
plt.figure(figsize=(7,5))
df['open_month']=[z.month for z in df['opendate']]
count=df['open_month'].value_counts(sort=False)
sns.barplot(y=count.values,x=count.index,palette='RdBu')
month_name=['','january','february','march','april','may','june','july','august','september','october','november','december']
plt.gca().set_xticklabels([month_name[x] for x in count.index],rotation='45')
plt.show()
print('%d job applications may close without prior notice' %df['deadline'].isna().sum())
req=' '.join(text for text in df['requirements'])



token=nltk.tokenize.word_tokenize(req)
counter=Counter(token)
count=[x for x in counter.most_common(40) if len(x[0])>3]
print("Most common words in Requirement")
print(count)
plotc=count[:][:10]
a,b=map(list, zip(*plotc))
sns.barplot(b,a,palette='RdBu')                                           ##creating barplot
plt.title('most common requirements')
plt.xlabel("count")
plt.ylabel('requirements')
plt.show()

plt.figure(figsize=(7,7))
count=df['selection'].astype(str).value_counts()[:10]
sns.barplot(y=count.index,x=count,palette='RdBu')
plt.gca().set_yticklabels(count.index,rotation='45')
plt.show()


def pronoun(data):
    '''function to tokenize data and perform pos_tagging.Returns tokens having "PRP" tag'''

    prn = []
    vrb = []
    token = nltk.tokenize.word_tokenize(data)
    pos = nltk.tag.pos_tag(token)

    vrb = Counter([x[0] for x in pos if x[1] == 'PRP'])

    return vrb


req_prn = pronoun(req)
duties= ' '.join(d for d in df['duties'])
duties_prn = pronoun(duties)
print('pronouns used in requirement section are')
print(req_prn.keys())
print('\npronouns used in duties section are')
print(duties_prn.keys())

for name in df['Position']:
    z=re.match(r'\w+?\s?\w+(man|woman|men|women)$',name)
    x = re.match(r'\w+?\s?(man|woman|men|women|male|female)$', name)
    if z is not None:
        print(z)
    if x is not None:
        print(x)


def similar_jobs(job):
    ''' function to find and return jobs with similar job title.take a single argument
            - job title
            returns
                -list of similar jobs '''

    word1 = nltk.tokenize.word_tokenize(job)
    jobs = []
    for i, name in enumerate(df['Position']):
        word2 = nltk.tokenize.word_tokenize(name)
        distance = nltk.jaccard_distance(set(nltk.ngrams(word1, n=1)), set(nltk.ngrams(word2, n=1)))
        if distance < .4:
            jobs.append((name, i))
    return jobs


def cosine_similarity(job):
    word1 = nltk.tokenize.word_tokenize(job)
    jobs = []
    for i, name in enumerate(df['Position']):
        word2 = nltk.tokenize.word_tokenize(name)
        # print(word2)
        # print(set(nltk.ngrams(word2, n=1)))
        from scipy import spatial
        distance = 1 - spatial.distance.cosine(set(nltk.ngrams(word1, n=1)), set(nltk.ngrams(word2, n=1)))
        if distance < .5:
            jobs.append((name, i))
    return jobs

print(similar_jobs(df['Position'][118]))


def similar_req(job):
    ''' function to find and return jobs with similar job title.take a single argument
            - job title
            returns
                -list of similar jobs '''

    word1 = nltk.tokenize.word_tokenize(job)
    jobs = []
    for i, name in enumerate(df['requirements']):
        word2 = nltk.tokenize.word_tokenize(name)
        distance = nltk.jaccard_distance(set(nltk.ngrams(word1, n=1)), set(nltk.ngrams(word2, n=1)))
        if (distance < .5):
            jobs.append((name, df.iloc[i]['Position']))
    return jobs
print(similar_req(df['requirements'][10]))

reading = []
for file in df['File Name']:
    text=open( "C:/Users/twist/CityofLA/Job Bulletins/"+file,'r').read()
    #text = open("../input/cityofla/CityofLA/Job Bulletins/" + file, 'r').read()
    sentence = text.count('.') + text.count('!') + text.count(';') + text.count(':') + text.count('?')
    words = len(text.split())
    syllable = 0
    for word in text.split():
        for vowel in ['a', 'e', 'i', 'o', 'u']:
            syllable += word.count(vowel)
        for ending in ['es', 'ed', 'e']:
            if word.endswith(ending):
                syllable -= 1
        if word.endswith('le'):
            syllable += 1

    G = round((0.39 * words) / sentence + (11.8 * syllable) / words - 15.59)
    reading.append(G)
    #text.close()

plt.hist(reading)
plt.xlabel('Flesch Index')
plt.title('Flesch index distribution')
plt.show()
promotions=list()
for name in df['Position']:
    z=re.match(r'\w*\s*\w*\s*(supervisor|manager|specialist|senior)\s*\w*\s*\w*$',name)
    if z is not None:
        promotions.append(name)
       # print(name)
positions=list()
for y in promotions:
    z=re.sub(r'\s?(supervisor|manager|specialist|senior)\s?$',"",str(y))
    positions.append(z)
lowers=list()
print(positions)
for name in df['Position']:
    for check in positions:
        if name.find(check)!=-1 and name not in promotions and name not in lowers:
            lowers.append(name)
print(lowers)

listprom=pd.DataFrame(columns=['start','promotion','salary'])
for pos in df['Position']:
    promotioner = list()
    salprom = list()
    sim=similar_jobs(pos)
    sal=df.loc[df['Position']==pos,'salary_start'].iloc[0]
    #print('sal=')
   # print(sal)
    if sim is not None:
      for name,index in sim:
          if int(df.loc[index,'salary_start'])>int(sal):
              promotioner.append(name)
              salprom.append(df.loc[index,'salary_start'])
               # print(promotioner)
    df2=pd.DataFrame({'promotion':promotioner})
    df2['salary']=salprom
    df2['start']=pos
    #display(df2)
display(listprom.head())
print(listprom.shape)
listprom=listprom.reset_index(drop=True)
listprom.to_csv('listprom.csv')
print(df['SCHOOL_TYPE'])
print(df[df['SCHOOL_TYPE']=='College or University'].shape)
print(df['SCHOOL_TYPE'].unique())
print(df[df['SCHOOL_TYPE']=='College or University'].shape)
tot=df['SCHOOL_TYPE'].shape[0]
x1=round(df[df['SCHOOL_TYPE']=='College or University'].shape[0]/tot*100)
x2=100-x1
plt.figure(figsize=(5,5))
plt.bar(height=[x1,x2],x=['College Degree','No College Degree'])
plt.show()

