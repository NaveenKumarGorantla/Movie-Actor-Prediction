#python file
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import re
import os
import glob


def get_traindata(input_files):
    if ( len(input_files) == 0):
        raise   Exception('empty values')
    list_filedata = []
    list_filepaths =[]
    for eachfile in input_files:
        for filename in eachfile:
            filename = filename
            filepaths = glob.glob(str(filename))
            list_filepaths.append(filepaths)

    filepaths_list = nltk.flatten(list_filepaths)
    filepaths_list = filepaths_list[:200]
    print('length_train',len(filepaths_list))
    #print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        list_filedata.append(file_data)
        fileopen.close()
    print(list_filedata[0:10])
    return list_filedata

def get_testdata(input_files):
    if ( len(input_files) == 0):
        raise   Exception('empty values')
    list_filedata = []
    list_filepaths =[]
    for eachfile in input_files:
        for filename in eachfile:
            filename = filename
            filepaths = glob.glob(str(filename))
            list_filepaths.append(filepaths)

    filepaths_list = nltk.flatten(list_filepaths)
    filepaths_list = filepaths_list[0:50]
    print('length_test',len(filepaths_list))
    print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        list_filedata.append(file_data)
        fileopen.close()
    return list_filedata


def find_entity(totaldata):
    documentcount=0
    print("length of total data in find entity",len(totaldata))
    nameslist = []
    for doc in totaldata:
        documentwordcount=word_tokenize(doc)
        documentcount+=1
        #print(documentcount)
        chunklist = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(doc)))
        for chunk in chunklist.subtrees(filter=lambda t: t.label() == 'PERSON'):
            name =[];namelength = 0;wordscount = 0; wordlengthlist = [];
            #print(chunk)
            for leave in chunk.leaves():
                #print(leave)
                name.append(leave[0])
                wordscount += 1
                wordlengthlist.append(len(leave[0]))
            if(len(wordlengthlist)<3):
                wordlengthlist.append(0);wordlengthlist.append(0);
            name = ' '.join(name)
            namelength = len(name)
            dict = {'name': name,'wordscount': wordscount, 'Firstwordlength': wordlengthlist[0], 'Secondwordlength': wordlengthlist[1], 'Thirdwordlength': wordlengthlist[2], 'name_length': namelength}
            nameslist.append(dict)
    return nameslist


#Training

inputfiledata = get_traindata([['aclImdb/train/pos/*.txt']])
#print("traindata",inputfiledata[0:10])
trainingdata=find_entity(inputfiledata)
#print('training data',trainingdata)

from sklearn.feature_extraction import DictVectorizer
import numpy as np

#2 training the model

y_train = []
x_train = []
for item in trainingdata:
    y_train.append(item['name'])
    del item['name']
    x_train.append(item)

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train)
#print(x_train)
y_train = np.array(y_train)
#print("Features:", vec.get_feature_names())

from sklearn.naive_bayes import MultinomialNB
#Training the model
model= MultinomialNB()
model.fit(x_train, y_train)



#redaction

def get_redactednameentities(totaldata):
    totalredacteddata=[]

    redactednameentitycount=[]
    for doc in totaldata:
        personslist = []
        chunklist = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(doc)))
        for chunk in chunklist.subtrees(filter=lambda t: t.label() == 'PERSON'):
            name=[]
            #print(chunk)
            for leave in chunk.leaves():
                name.append(leave[0])
            personslist.append(' '.join(name))
        setofnameslist = list(set(personslist))
        #sorting based on length of the name and sorting in descending order
        setofnameslist=sorted(setofnameslist,key=lambda name: len(name),reverse=True)
        #print(setofnameslist)
        for itemname in setofnameslist:
            itemnamelist=itemname.split()
           # print(itemname,len(itemname))
            #doc = doc.replace(itemname,str( '█' * len(itemname)))
            if(len(itemnamelist)==1):
                doc = doc.replace(itemname, '█' * len(itemname)+'#')
            elif(len(itemnamelist) == 2):
                doc = doc.replace(itemname, '█' * len(itemnamelist[0])+'#'+'█' * len(itemnamelist[1]))
            elif (len(itemnamelist) == 3):
                doc = doc.replace(itemname, '█' * len(itemnamelist[0]) +'#'+ '█' * len(itemnamelist[1])+ '#' + '█' * len(itemnamelist[2]))
            elif (len(itemnamelist) == 4):
                doc = doc.replace(itemname, '█' * len(itemnamelist[0]) + '#' + '█' * len(itemnamelist[1])+ '#' + '█' * len(itemnamelist[2])+ '#' + '█' * len(itemnamelist[3]))
        totalredacteddata.append(doc)
    return totalredacteddata




#redaction

def ExtractFeatures_redact_data(totaldata):
    redacted_name_features={}
    redactednames_featureslist=[]
    for redacteddocument in totaldata:
        for item in redacteddocument.split(' '):
            wordlengthlist=[]
            wordlist=[]
            list=item.split('#')
            count=0
            for it in list:
                if(it.startswith('█')):
                    wordlist.append(it)
                    wordlengthlist.append(len(it))
                    count+=1
            #print(item,count)
            if(len(wordlengthlist)<4):
                wordlengthlist.append(0);wordlengthlist.append(0);wordlengthlist.append(0)
            if(count>0):
                redacted_name_features = {'name': ' '.join(wordlist), 'wordscount': count, 'Firstwordlength': wordlengthlist[0],'Secondwordlength': wordlengthlist[1], 'Thirdwordlength': wordlengthlist[2],'name_length': len(' '.join(wordlist))}
                #print(redacted_name_features)
                redactednames_featureslist.append(redacted_name_features)
    return redactednames_featureslist





#Testing

testfiledata=get_testdata([['aclImdb/test/pos/*.txt']])
redacteddocuments=get_redactednameentities(testfiledata)
#print(redacteddocuments)
redactedfeatureslist=ExtractFeatures_redact_data(redacteddocuments)
#print(redactedfeatureslist)

#predicting
x_test=[]
y_test=[]
redacteddocumentcount=0
for item in redactedfeatureslist:
    redacteddocumentcount+=1
    #print(redacteddocumentcount)
    # print(name_dict['name'])
    y_test.append(item['name'])
    del item['name']
    x_test.append(item)
x_test = vec.fit_transform(x_test)
y_pred =  model.predict(x_test)
#print(y_pred)
newfilepath = os.path.join(os.getcwd(),'outputfile')
if not os.path.exists(newfilepath):
    os.makedirs(newfilepath)
    with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
        for item in y_pred:
            outputfile.write(item)
elif os.path.exists(newfilepath):
    with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
        for item in y_pred: 
            outputfile.write(" " + item+ " ")


