import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time

#dataset="No.12_sst2"
dataset="111"

path_0="./data/sst/"
path="tensor"+"/"

#read data
test_vector=np.loadtxt(path+"tensor_test.csv",delimiter=",")
valid_vector=np.loadtxt(path+"tensor_valid.csv",delimiter=",")
train_vector=np.loadtxt(path+"tensor_train.csv",delimiter=",")

#get index,text and label
train_data=pd.read_csv(path_0+"train.csv")
train_label=train_data["label"]
train_text=train_data["text"]
#index_train = np.arange(train.shape[0])
train_signature= np.column_stack((train_vector,train_label,train_text))
#np.savetxt(path+"train_new.csv",train_new,delimiter=",")


valid_data=pd.read_csv(path_0+"valid.csv")
valid_label=valid_data["label"]
valid_text=valid_data["text"]
#index_valid = np.arange(valid.shape[0])
valid_signature= np.column_stack((valid_vector,valid_label,valid_text))
#np.savetxt(path+"valid_new.csv",valid_new,delimiter=",")

#construct signature database
result = np.vstack((train_signature, valid_signature))
index = np.arange(result.shape[0])
database_signature= np.column_stack((index[:, np.newaxis],result))
#np.savetxt(path+"database.csv",result_new,delimiter=",")


test_data=pd.read_csv(path_0+"test.csv")
test_label=test_data["label"]
test_text=test_data["text"]
test_index = np.arange(test_data.shape[0])
test_signature= np.column_stack((test_index[:, np.newaxis],test_vector,test_label,test_text))

tensor_database = torch.from_numpy((database_signature[:,1:3]).astype(float))
tensor_test = torch.from_numpy((test_signature[:,1:3]).astype(float))
label_test=test_signature[:,3]




#Data loading
tensor_database = torch.from_numpy((database_signature[:,1:3]).astype(float))
tensor_test = torch.from_numpy((test_signature[:,1:3]).astype(float))
label_test=test_signature[:,3]

#Similarity calculation
similarity_matrix = F.cosine_similarity(tensor_test.unsqueeze(1), tensor_database.unsqueeze(0), dim=2)
sim=torch.max(similarity_matrix , 1)


# Extract results and save
max=torch.max(similarity_matrix , 1)
sim=max[0].numpy()
idx=max[1].numpy()
#np.savetxt(path+"sim.csv", sim ,delimiter=",")


# Save results to csv
idx_sim = np.column_stack((idx, sim)) 
idx_sim= torch.from_numpy(idx_sim)
np.savetxt(path+'idx_sim.csv', idx_sim, delimiter=',')#,header=["max_index","similarity"])


#Match tags based on index
data_match=np.column_stack((test_index,idx_sim,label_test))

df1 = pd.DataFrame(data_match, columns=['index', 'max_index','similarity','label'])
df2 = pd.DataFrame(database_signature[0:4], columns=['max_index', 'ten','sor','max_label'])
result = pd.merge(df1, df2, on='max_index', how='left')
result=result.drop(["ten","sor"],axis=1)




df=pd.read_csv(path+"to_assess.csv")


# Threshold settings
similarity_list=[0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.999, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999, 0.9999, 0.99991, 0.99992, 0.99993, 0.99994, 0.99995, 0.9999600000000001, 0.99997, 0.9999800000000001, 0.99999, 0.999999, 0.9999991, 0.9999992, 0.9999993, 0.9999994, 0.9999995, 0.9999996, 0.9999997, 0.9999998, 0.9999999, 0.9999999, 0.99999991, 0.99999992, 0.99999993, 0.99999994, 0.99999995, 0.99999996, 0.99999997, 0.99999998, 0.99999999]
def assess(df,similarity):
    sum= (df['index']).count() 
    TP=((df['similarity'] > similarity) & (df['label'] == df['max_label'])).sum()
    FP=((df['similarity'] > similarity) & (df['label'] != df['max_label'])).sum()
    FN=(df['similarity'] <= similarity).sum()
    acc=TP/sum
    pre=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*pre*recall/(pre+recall)
    #print("accuracy:",acc,"\tprecision:",pre,"\nrecall:",recall,"\tF1:",F1,"\n")
    return {  
       'Threshold': similarity,  
       'Accuracy': acc,  
       'Precision': pre,  
       'Recall': recall,  
       'F1': F1  
   }  

results_list = []  
      
for similarity in similarity_list:
    result=assess(df, similarity)
    results_list.append(result) 
 
# convert the result list into a DataFrame  
results_df = pd.DataFrame(results_list)  
