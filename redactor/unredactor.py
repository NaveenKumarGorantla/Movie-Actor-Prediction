#python file
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
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
    print('Text files in Train data',len(filepaths_list))
    #print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        #file_data = re.sub(r'[^a-zA-Z0-9]'," ",file_data)
        list_filedata.append(file_data)
        fileopen.close()
    #print(list_filedata[0:10])
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
    filepaths_list = filepaths_list[0:1]
    print('Text files in Test data',len(filepaths_list))
    #print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        list_filedata.append(file_data)
        fileopen.close()
    return list_filedata


def get_named_entities(list_filedata):
    if ( len(list_filedata) == 0):
        raise Exception('Empty text data')
    Dictionary_list = []
    for data in list_filedata:

        words = nltk.word_tokenize(data)
        word_tokens = nltk.pos_tag(words)
        named_entities = nltk.ne_chunk(word_tokens)
    
        for entity in named_entities.subtrees():
            word_name =[]
            total_words = 0
            length_words = [];
            length_words = [ 0 for i in range(100)]
            i = 0
            if ( entity.label() == 'PERSON'):
                for leaf in entity.leaves():
                    word_name.append(leaf[0])
                    #print(leaf[0])
                    total_words += 1
                    length_words[i]= len( leaf[0])
                    i = i + 1
                word_name = ' '.join(word_name)
                #print(word_name)
                total_length = len(word_name)
                dict = {'F1': word_name,'F2': total_words, 'F3': length_words[0], 'F4': length_words[1], 'F5': length_words[2], 'F6': total_length}
                Dictionary_list.append(dict)
    #print(Dictionary_list[0:10])
    return Dictionary_list


def Model_training():

    train_data = get_traindata([['aclImdb/train/pos/*.txt']])
    if ( len(train_data) > 0):

        named_entities=get_named_entities(train_data)
        y_train = []
        x_train = []
        for item in named_entities:
            y_train.append(item['F1'])
            del item['F1']
            x_train.append(item)

        vec = DictVectorizer(sparse=False)
        x_train = vec.fit_transform(x_train)
        #print(x_train)
        y_train = np.array(y_train)
        #print("Features:", vec.get_feature_names())
        #Training the model
        model= MultinomialNB()
        model.fit(x_train, y_train)

    else:
        return None

    return model

model = Model_training()

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
                print(leave[0])
            personslist.append(' '.join(name))
        print('test data person names',personslist)    
        setofnameslist = list(set(personslist))
        #sorting based on length of the name and sorting in descending order
        setofnameslist=sorted(setofnameslist,key=lambda name: len(name),reverse=True)
        print('set of name list :',setofnameslist)
        for itemname in setofnameslist:
            itemnamelist=itemname.split()
            print(itemnamelist)
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
                redacted_name_features = {'F1': ' '.join(wordlist), 'F2': count, 'F3': wordlengthlist[0],'F4': wordlengthlist[1], 'F5': wordlengthlist[2],'F6': len(' '.join(wordlist))}
                #print(redacted_name_features)
                redactednames_featureslist.append(redacted_name_features)
    return redactednames_featureslist





#Testing

testfiledata=get_testdata([['aclImdb/test/pos/*.txt']])
print(testfiledata)
redacteddocuments=get_redactednameentities(testfiledata)
print(redacteddocuments)
redactedfeatureslist=ExtractFeatures_redact_data(redacteddocuments)
print(redactedfeatureslist)

#predicting
vec = DictVectorizer(sparse=False)
x_test=[]
y_test=[]
redacteddocumentcount=0
for item in redactedfeatureslist:
    redacteddocumentcount+=1
    #print(redacteddocumentcount)
    # print(name_dict['name'])
    y_test.append(item['F1'])
    del item['F1']
    x_test.append(item)
x_test = vec.fit_transform(x_test)

#model = Model_training()

y_pred =  model.predict(x_test)
#print('Score', model.score(x_test, y_test))
#print(y_pred)
newfilepath = os.path.join(os.getcwd(),'outputfile')
if not os.path.exists(newfilepath):
    os.makedirs(newfilepath)
    with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
        for item in y_pred:
            outputfile.write(item)
elif os.path.exists(newfilepath):
    with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
        count = 1
        for item in y_pred: 
            outputfile.write(str(count)+"." +" "+ item+ " / ")
            count = count + 1


