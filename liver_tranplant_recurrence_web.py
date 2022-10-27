import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
#应用标题
st.set_page_config(page_title='Prediction model for Recurrence of hepatocellular carcinoma')
st.title('Prediction model for recurrence of hepatocellular carcinoma after liver transplantation：Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
tumor_size = st.sidebar.selectbox('tumor_size',('≤3cm','>3cm'),index=0)
tumor_single = st.sidebar.selectbox('tumor_single',('Single tumor','Multiple tumor'),index=0)
vascular_invasion = st.sidebar.selectbox('vascular_invasion',('No','Yes'),index=1)
# extrahepatic_metastasis = st.sidebar.selectbox('extrahepatic_metastasis',('No','Yes'),index=1)
BCLC = st.sidebar.selectbox('BCLC',('Stage 0','Stage A','Stage B','Stage C'),index=1)
Fibrinogen = st.sidebar.slider("Fibrinogen", 0.00, 10.0, value=5.00, step=0.01)
# Age = st.sidebar.slider("Age(year)", 0, 99, value=45, step=1)
# PreAlb = st.sidebar.slider("Albumin", 0.0, 99.0, value=38.7, step=0.1)
# PreGGT = st.sidebar.slider("Gamma-glutamyl transpeptidase", 0, 500, value=284, step=1)
# PreALP = st.sidebar.slider("Alkaline phosphatase", 0, 400, value=81, step=1)
# PreFib = st.sidebar.slider("Fibrinogen", 0.00, 20.00, value=5.05, step=0.01)
WBC = st.sidebar.slider("White blood cell", 0.00, 40.00, value=3.72, step=0.01)
N = st.sidebar.slider("Neutrophil granulocyte", 0.00, 20.00, value=3.72, step=0.01)
M = st.sidebar.slider("Macrophages", 0.00, 1.00, value=0.50, step=0.01)
plt = st.sidebar.slider("Platelet", 0, 500, value=189, step=1)
NLR = st.sidebar.slider("The ratio of neutrophils to lymphocytes", 0.00, 30.00, value=2.93, step=0.01)
PLR = st.sidebar.slider("The ratio of platelet to lymphocytes", 0.00, 500.00, value=56.00, step=0.01)
LMR = st.sidebar.slider("The ratio of lymphocytes to mononuclear macrophages", 0.00, 10.00, value=1.98, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,'Yes':1,'≤3cm':0,'>3cm':1,'Single tumor':1, 'Multiple tumor':2,'Stage 0':0,"Stage A":1,'Stage B':2,'Stage C':3}
tumor_size =map[tumor_size]
tumor_single =map[tumor_single]
vascular_invasion = map[vascular_invasion]
# extrahepatic_metastasis = map[extrahepatic_metastasis]
BCLC = map[BCLC]

# 数据读取，特征标注
hp_train = pd.read_csv('Preoperative_data_of_liver_transplantation_for_liver_cancer_2.csv')
hp_train['Recurrence'] = hp_train['Recurrence'].apply(lambda x : +1 if x==1 else 0)
features =["tumor_size","tumor_single","vascular_invasion","BCLC",'Fibrinogen','WBC','N','plt',"M",'NLR','PLR','LMR']
target = 'Recurrence'
random_state_new = 50
data = hp_train[features]
# for name in ['tumor_size','vascular_invasion','plt','PLR']:
#     X = data.drop(columns=f"{name}")
#     Y = data.loc[:, f"{name}"]
#     X_0 = SimpleImputer(missing_values=np.nan, strategy="constant").fit_transform(X)
#     y_train = Y[Y.notnull()]
#     y_test = Y[Y.isnull()]
#     x_train = X_0[y_train.index, :]
#     x_test = X_0[y_test.index, :]

#     rfc = RandomForestRegressor(n_estimators=100)
#     rfc = rfc.fit(x_train, y_train)
#     y_predict = rfc.predict(x_test)

#     data.loc[Y.isnull(), f"{name}"] = y_predict
    
X_data = data
# X_data.isnull().sum(axis=0)
#转换自变量
X_ros = np.array(X_data)
# X_ros = np.array(hp_train[features])
y_ros = np.array(hp_train[target])
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
# XGB = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = 0)
mlp.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (mlp.predict_proba(np.array([[tumor_size,tumor_single,vascular_invasion,BCLC,Fibrinogen,WBC,N,plt,M,NLR,PLR,LMR]]))[0][1])> sp
prob = (mlp.predict_proba(np.array([[tumor_size,tumor_single,vascular_invasion,BCLC,Fibrinogen,WBC,N,plt,M,NLR,PLR,LMR]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Recurrence'
else:
    result = 'Low Risk Recurrence'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Recurrence':
        st.balloons()
    st.markdown('## Probability of High risk Recurrence group:  '+str(prob)+'%')
