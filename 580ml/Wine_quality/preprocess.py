import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def wine_filter(inputf,t_size, pre_flag=0):
	dtset = pd.read_csv(inputf,delimiter=";",index_col=None)
	#print (dtset)
	#print(dtset["quality"])
	X= dtset.iloc[:,:-1].values
	Y = dtset.iloc[:,-1].values
 	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t_size,random_state= 0)
 	sc_x =StandardScaler()
	X_train = sc_x.fit_transform(X_train)
	X_test = sc_x.transform(X_test)
	return X_train,X_test,Y_train,Y_test


def get_att(inputf,color,col1,col2,sort_flag=0):
	dtset = pd.read_csv(inputf, delimiter=";",index_col=None)
	dtout = dtset[[col1,col2]] 
	if not sort_flag:
		dtout = dtout.sort_values(by=col1)
	else:
		dtout = dtout.sort_values(by=col2)
	outfile_name = "2d_ds/"+color+"_"+col1 + "_"+col2+".csv"
	dtout.to_csv(outfile_name,index=False)

def re_range(inputf,outputf,ori_st,ori_ed):
	df = pd.read_csv(inputf,delimiter=";",index_col=None)
	for i in range(ori_st,ori_ed+1):
		df["quality"] = df["quality"].replace(i,i-2)
	df.to_csv(outputf,sep=';',index=False)	
	


X_train,X_test,Y_train,Y_test = wine_filter('winequality-red.csv',0.2,1)
re_range('winequality-red.csv','rerange_red.csv',3,8)
get_att('rerange_white.csv',"white_rerange","alcohol","quality",1)
#print X_train,X_test,Y_train,Y_test




	
