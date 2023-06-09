# %%
from math import sqrt
import regex as re
import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, accuracy_score

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import scipy

import torch
import transformers as ppb
import warnings

from xgboost import XGBClassifier, XGBRFClassifier, XGBRegressor, XGBRFRegressor
from xgboost import plot_importance
from xgboost import plot_importance
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

# %% [markdown]
# ## Interspeech 2020 Challenge
# Deadline: May 8th submission of predictions and paper.
# 
# Main Webpage: http://www.interspeech2020.org/index.php?m=content&c=index&a=lists&catid=66
# 
# Challenge Webpage: http://www.homepages.ed.ac.uk/sluzfil/ADReSS/

# %% [markdown]
# # Initial Plan
# Only using the transcripts:
# - [x] Simple clean and join all sentences, classifiy using DistillBERT, (BERT), (RoBERTa)   ## Done
# 
# ## Further Feature Engineering:
# ### Time dimension
# - [x] Embed time total time taken - parse time blocks, take first and last
# - [x] Embed total time taken per sentence
# - [x] Time before starting speech
# - [x] Time in between each sentence
# - [x] Average / min / max / median time of sentence
# - use of special characters
# - number of sentences spoken
# 
# ### Linguistic Features
# - [x] Embed special character tokens in speech, pauses etc. (not sure if this needed, tokenzier / and
# - [x] classify on a sentence level??
# - [x] Also use the Interviewer INV, questions / speech / time...
# - Use POS Tags: as OHE vector
# 
# ## Demographics
# - [x] Gender
# - [x] Age
# 
# ## Fine-tuning BERT(-esque) models on spontaneous speech datasets
# - fine-tune and re-classify using other spontaneous speech datasets - GPU is down :( 
# 
# ### Further work on
# -  Analysis of what roBERTa has actually learned in the attention heads

# %%
prob_ad_dir = '../train/transcription/cd/*'
controls_dir = '../train/transcription/cc/*'

# %%
# 这段代码定义了一个名为extract_data的函数,该函数接收一个file_name参数作为输入,用于指定要从中提取数据的文件名。
# 函数会返回一个字典,其中包含从文件中提取的数据。
# 具体来说,函数首先创建一个空字典par,然后从文件名中提取出ID,并将其作为字典的id键对应的值。接着,函数遍历文件的每一行,
# 并逐行进行处理。如果当前行以@ID开头,函数会提取出参与者的一些信息（如年龄、性别和MMSE分数）并将其存储到par字典中。如果当前
# 行以*PAR:或*INV开头,函数会将当前行作为正在进行的发言,如果当前行不是以%或*开头且正在进行的发言不为空,则将当前行添加到正在进行的发言中。
# 如果当前行不是以%或*开头且正在进行的发言为空,则跳过当前行。如果正在进行的发言不为空且当前行以%或*开头,则表示当前发言已结束,将当前发言添
# 加到speech列表中,并将当前正在进行的发言重置为空字符串。
# 函数返回一个字典,其中包含从文件中提取的参与者信息（如ID、年龄、性别和MMSE分数）以及发言列表（即speech变量）。
def extract_data(file_name):#从指定文件中提取数据
    par = {}
    par['id'] = file_name.split('/')[-1].split('.cha')[0]
    f = iter(open(file_name))
    l = next(f)
    speech = []
    try:
        curr_speech = ''
        while (True):
            if l.startswith('@ID'):
                participant = [i.strip() for i in l.split('|')]
                if participant[2] == 'PAR':
                    try:
                        par['mmse'] = '' if len(participant[8]) == 0 else float(participant[8])
                    except:
                        par['mmse'] = ''
                    par['sex'] = participant[4][0]
                    par['age'] = int(participant[3].replace(';', ''))
            if l.startswith('*PAR:') or l.startswith('*INV'):
                curr_speech = l
            elif len(curr_speech) != 0 and not(l.startswith('%') or l.startswith('*')):
                curr_speech += l
            elif len(curr_speech) > 0:
                speech.append(curr_speech)
                curr_speech = ''
            l = next(f)
    except StopIteration:
        pass

    clean_par_speech = []
    clean_all_speech = []
    par_speech_time_segments = []
    all_speech_time_segments = []
    is_par = False
    for s in speech:
        def _parse_time(s):
            return [*map(int, re.search('\x15(\d*_\d*)\x15', s).groups()[0].split('_'))]
        
        def _clean(s):
            s = re.sub('\x15\d*_\d*\x15', '', s) # remove time block 
            s = re.sub('\[.*\]', '', s) # remove other speech artifacts [.*]
            s = s.strip()
            s = re.sub('\t|\n|<|>', '', s) # remove tab, new lines, inferred speech??, ampersand, &
            return s
        
        if s.startswith('*PAR:'):
            is_par = True
        elif s.startswith('*INV:'):
            is_par = False
            s = re.sub('\*INV:\t', '', s) # remove prefix
        if is_par:
            s = re.sub('\*PAR:\t', '', s) # remove prefix    
            par_speech_time_segments.append(_parse_time(s))
            clean_par_speech.append(_clean(s))
        all_speech_time_segments.append(_parse_time(s))
        clean_all_speech.append(_clean(s))
        
    par['speech'] = speech
    par['clean_speech'] = clean_all_speech
    par['clean_par_speech'] = clean_par_speech
    par['joined_all_speech'] = ' '.join(clean_all_speech)
    par['joined_all_par_speech'] = ' '.join(clean_par_speech)
    
    # sentence times
    par['per_sent_times'] = [par_speech_time_segments[i][1] - par_speech_time_segments[i][0] for i in range(len(par_speech_time_segments))]
    par['total_time'] =  par_speech_time_segments[-1][1] - par_speech_time_segments[0][0]
    par['time_before_par_speech'] = par_speech_time_segments[0][0]
    par['time_between_sents'] = [0 if i == 0 else max(0, par_speech_time_segments[i][0] - par_speech_time_segments[i-1][1]) 
                                 for i in range(len(par_speech_time_segments))]
    return par

# %%
# 封装一个函数可以使代码更加模块化、可复用、易于维护和测试。对于上面的例子,封装一个函数parse_train_data()可以将数据解析的过程独立出来,方便在其他部分调用,
# 同时也方便对该函数进行单元测试,保证函数的正确性和稳定性。在后续可能的需求变化中,只需要修改该函数的实现,而不需要改变调用该函数的其他部分。
def parse_train_data():
    return _parse_data('../data/train')
    

def parse_test_data():
    """
    这段代码会读取指定文件夹中所有的测试集音频转写文件,然后通过调用 extract_data 函数,将每个音频转写文件中的数据提取出来,
    并将提取的结果转换为一个 Pandas DataFrame 格式的数据,最后返回这个 DataFrame。
    """
    return pd.DataFrame([extract_data(fn) for fn in glob('../data/test/transcription/*')])

def _parse_data(data_dir):
    prob_ad_dir = f'{data_dir}/transcription/cd/*'
    controls_dir = f'{data_dir}/transcription/cc/*'
    
    prob_ad = [extract_data(fn) for fn in glob(prob_ad_dir)]
    controls = [extract_data(fn) for fn in glob(controls_dir)]
    controls_df = pd.DataFrame(controls)
    prob_ad_df = pd.DataFrame(prob_ad)
    controls_df['ad'] = 0
    prob_ad_df['ad'] = 1
    df = pd.concat([controls_df, prob_ad_df]).sample(frac=1).reset_index(drop=True)
    return df

# %%
train_df = parse_train_data() # 加载训练数据

# %% [markdown]
# ### Base-line TF-IDF -> GBDT / SVM Models

# %%
random_state = 42

# %%
def cv10_avg(score, model, features, labels):
    """
    这是一个用于进行10折交叉验证的函数,函数参数包括评分方式 score,模型 model,特征数据 features 和标签数据 labels。
    函数中先调用 cross_val_score 函数计算得分,打印输出并返回10次得分的平均值（保留两位小数）。
    """
    print(cross_val_score(model, features, labels, cv=10, scoring=score))
    return round(cross_val_score(model, features, labels, cv=10, scoring=score).sum() / 10, 2)


# %%
# 使用基线模型对文本数据进行分类
def baseline_models(text: pd.Series, labels: list, mmse: list, xgb=True, shuffle=True, print_stats=True):
    ## AD Classification Pred
    # sklearn pipeline
    """
    这是一个函数 baseline_models(),接受三个参数 text、labels、mmse,其中 text 是包含文本数据的 Pandas Series 对象,labels 是标签列表,mmse 
    是一个包含 MMSE 分数的列表。函数的主要目的是使用基线模型来对这些数据进行分类和回归。
函数首先定义了一个包含分类和回归模型的 pipeline,使用 GridSearchCV 通过交叉验证来确定模型的最佳参数,然后在训练集上拟合模型并在测试集上进行预测。对于分类任务,
函数使用准确率、精确度、召回率和 F1 分数来评估模型的性能,对于回归任务,函数使用均方根误差来评估模型的性能。最后,函数返回分类模型和回归模型。如果 print_stats 参数为 True,则会打印出评估指标的值。
    """
    param_space = {
        'vec__max_features': [100, 500, 1000, 2000, 10000],
        'vec__stop_words': ['english', None],
        'vec__analyzer': ['word', 'char'],
        'vec__sublinear_tf': [True, False]       
    }    
    if xgb:
        param_space['clf__n_estimators'] = [100, 200, 500]  
        param_space['clf__max_depth'] = [3, 5, 10]
    else:
        param_space['clf__C'] = [0.1, 0.5, 1.]              
        param_space['clf__kernel'] = ['rbf', 'sigmoid']    

    clf_pipe = Pipeline([
        ('vec', TfidfVectorizer()),
        ('clf', XGBClassifier(probability=True)) if xgb else ('clf', SVC(probability=True))
    ])
    train_features, test_features, train_labels, test_labels = train_test_split(text, labels, random_state=random_state, test_size=0.2, shuffle=shuffle)
    search = GridSearchCV(clf_pipe, param_space, cv=5, n_jobs=6)
    search.fit(train_features, train_labels)

    clf_pipe.set_params(**search.best_params_)
    print(search.best_params_)
    clf_pipe.fit(train_features, train_labels)
    preds = clf_pipe.predict(test_features)
    if print_stats:
        print('prec, rec, f1 test', precision_recall_fscore_support(test_labels, preds))
        print(f'accu:{cv10_avg("accuracy", clf_pipe, text, labels)}')
        print(f'prec:{cv10_avg("precision", clf_pipe, text, labels)}')
        print(f'rec:{cv10_avg("recall", clf_pipe, text, labels)}')
        print(f'f1:{cv10_avg("f1", clf_pipe, text, labels)}')

    ## MMSE Regression Pred
    reg_features, reg_scores = text.drop([i for i, s in enumerate(mmse) if s == '']).reset_index(drop=True), [s for s in mmse if s != '']
    train_features, test_features, train_scores, test_scores = train_test_split(reg_features, reg_scores, random_state=random_state, test_size=0.2, shuffle=shuffle)

    # sklearn pipeline
    param_space = {
        'vec__max_features': [100, 500, 1000, 2000, 10000],
        'vec__stop_words': ['english', None],
        'vec__analyzer': ['word', 'char'],
        'vec__sublinear_tf': [True, False]
    }    
    if xgb:
        param_space['clf__n_estimators'] = [100, 200, 500]  
        param_space['clf__max_depth'] = [3, 5, 10]  
    else:
        param_space['clf__C'] = [0.1, 0.5, 1.]              
        param_space['clf__kernel'] = ['rbf', 'sigmoid']   

    rgs_pipe = Pipeline([
        ('vec', TfidfVectorizer()),
        ('clf', XGBRegressor()) if xgb else ('clf', SVR())
    ])

    search = GridSearchCV(rgs_pipe, param_space, cv=5, n_jobs=6)
    search.fit(train_features, train_scores)

    rgs_pipe.set_params(**search.best_params_)
    print(search.best_params_)
    rgs_pipe.fit(train_features, train_scores)
    preds = rgs_pipe.predict(test_features)
    if print_stats:
        print('rmse test:', sqrt(mean_squared_error(test_scores, preds)))
        print('rmse cv:', cross_val_score(rgs_pipe, reg_features, reg_scores, cv=10, scoring='neg_root_mean_squared_error').sum() / 10)

    return clf_pipe, rgs_pipe

# %%
# SVM - par speech only
clf_svm, rgs_svm = baseline_models(train_df.joined_all_par_speech, train_df.ad,  train_df.mmse, xgb=False)

# %%
# SVM - par + inv speech
clf_svm, rgs_svm = baseline_models(train_df.joined_all_speech, train_df.ad, train_df.mmse, xgb=False)

# %%
# XGBoost - par speech only
clf_xgb_par, rgs_xgb_par = baseline_models(train_df.joined_all_par_speech, train_df.ad, train_df.mmse, xgb=True)

# %%
# XGBoost - par + inv speech
clf_xgb_all, rgs_xgb_all = baseline_models(train_df.joined_all_par_speech, train_df.ad, train_df.mmse, xgb=True)

# %% [markdown]
# ## Segment Based Models
# - Treating each participant utterance as seperate data points

# %%
# explode out each segment into AD / control segments
# Do not shuffle, as parent level segments have already been shuffled
"""
这段代码是将原始数据集(train_df)中的每一个对话(segment)分割成多个段落(part),
然后将每个段落与对应的属性(ad, mmse, age, sex)组成一个新的数据集(train_df_segments)。不同于对话级别的shuffle,这里不会对每个段落进行shuffle,
因为已经在对话级别shuffle过了。最后将所有分割后的段落合并成一个新的数据集。
"""
segmented_speech = train_df.apply(lambda r: pd.DataFrame({'t_id': r.id, 'part_id': [str(i) for i, _ in enumerate(r.clean_par_speech)], 'speech_sent': r.clean_par_speech, 'ad': r.ad, 'mmse': r.mmse, 'age': r.age, 'sex': 1 if r.sex == 'm' else 0}), axis=1).tolist()
train_df_segments = pd.concat(segmented_speech).reset_index(drop=True)

# %%
def add_time_features(df: pd.DataFrame):
    # time features
    # - Embed time total time taken 
    # - parse time blocks, take first and last
    # - Embed total time taken per sentence
    # - Time before starting speech
    # - Time in between each sentence
    # - Average / min / max / median time of sentence
    """
    这段代码定义了一个函数 add_time_features,用于给输入的 DataFrame 添加时间相关的特征,并返回一个新的 DataFrame。
    """
    time_dims = df.loc[:, ['id', 'total_time', 'time_between_sents', 'per_sent_times']] # 对话的ID,对话的总时长,句子之间的间隔时间以及每个句子的时长列表
    time_dims['avg_sent_time'] = time_dims.per_sent_times.apply(lambda t: round(sum(t) / len(t))) # 平均句子时长
    time_dims['max_sent_time'] = time_dims.per_sent_times.apply(max) # 最大句子时长
    time_dims['min_sent_time'] = time_dims.per_sent_times.apply(min) # 最小句子时长
    time_dims_eng = time_dims.drop('time_between_sents', axis=1)

    def time_seg_df(r):
        return pd.DataFrame({'t_id': r.id,
                             'part_id': [str(i) for i, _ in enumerate(r.per_sent_times)],
                             'total_time': r.total_time, 
                             'segment_time': [t for t in r.per_sent_times],
                             'avg_sent_time': r.avg_sent_time,
                             'max_sent_time': r.max_sent_time,
                             'min_sent_time': r.min_sent_time,
                             })
    time_dim_segments = pd.concat(time_dims_eng.apply(time_seg_df, axis=1).tolist())

    d_cols = [c for c in time_dim_segments.columns if c not in ['t_id', 'part_id']]
    scaled = StandardScaler().fit_transform(time_dim_segments.loc[:, d_cols])
    for i, col in enumerate(d_cols):
        time_dim_segments[col] = scaled[:, i]
    return time_dim_segments

# %%
train_df_segments.age = StandardScaler().fit_transform(pd.DataFrame(train_df_segments.age.to_numpy()).to_numpy())#对DataFrame中的"age"列进行标准化处理。

# %%

# 将时间特征与原始数据合并成一个新的dataframe
time_dim_segments = add_time_features(train_df) #添加一些关于时间的特征
train_df_segments = train_df_segments.merge(time_dim_segments, on=['t_id', 'part_id']) #t_id表示题目的ID，part_id表示该题目所属的部分，两者都是共同的列，用于将两个DataFrame进行对应合并。


# %%
# SVM - segmented
clf_svm_segment, rgs_svm_segments = baseline_models(train_df_segments.speech_sent, train_df_segments.ad, train_df_segments.mmse, xgb=False, shuffle=False, print_stats=False) 

# %%
# XGBoost - segmented
"""
这段代码调用了一个名为 baseline_models() 的函数，并传递了一些参数。该函数的作用是训练一个基准模型，并返回两个模型对象 clf_xgb_segments 和 rgs_xgb_segments。

具体来说，该函数接受四个参数：

train_df_segments.speech_sent：表示训练集中的语音情感特征。
train_df_segments.ad：表示训练集中的阿尔茨海默症诊断结果。
train_df_segments.mmse：表示训练集中的MMSE评分结果。
xgb=True：表示使用XGBoost算法进行模型训练
"""
clf_xgb_segments, rgs_xgb_segments = baseline_models(train_df_segments.speech_sent, train_df_segments.ad, train_df_segments.mmse, xgb=True, shuffle=False, print_stats=False) 

"""
这两个数值是模型性能的评估指标，用于衡量模型在测试集上的表现和在交叉验证中的平均表现。

rmse test（Root Mean Squared Error）是测试集上的均方根误差，表示模型在测试集上预测值与真实值之间的平均差距。
rmse cv 是交叉验证中的均方根误差，表示模型在交叉验证中不同数据集上预测值与真实值之间的平均差距。
这两个指标的值越小，模型的表现越好。
"""

# %%
def features(df, base_clf_ft_probs, r, i, include_time=True, include_demo=False):
    # base clf 
    seq = df[base_clf_ft_probs].tolist()
    seq_item = seq[i]
    features = {
        'bias': 1.,
        'prob_ad': seq_item[1],
        'prob_cc': seq_item[0],
    }
    if include_time:
        features.update({
            # time based features
            'total_time': r.total_time,
            'segment_time': r.segment_time,
            'avg_sent_time': r.avg_sent_time,
            'max_sent_time': r.max_sent_time,
            'min_sent_time': r.min_sent_time
        })
    if include_demo:
        features.update({
            'age': r.age,
            'sex': r.sex,
        })

    # context features
    if i > 0:
        features.update({
            '-1:prob_ad': seq[i-1][1],
            '-1:prob_ad': seq[i-1][0]
        })
        if include_time:
            features['-1:segment_time'] = df.iloc[i-1].segment_time
    else:
        features['BOS'] = True
    
    if i < len(seq) - 1:
        features.update({
            '+1:prob_ad': seq[i+1][1],
            '+1:prob_cc': seq[i+1][0],
            '+1:segment_time': df.iloc[i+1].segment_time
        })
        if include_time:
            features['-1:segment_time'] = df.iloc[i-1].segment_time
    else:
        features['EOS'] = True
    return features

# %%
# Collapse sequences back into DataFrame with CRF features.
def crf_featurize(X, y, base_clf_feature_name):
    collapsed_df = {'t_id': [], 'crf_features': [], 'ad': []}
    for t_id, pt_sq in X.groupby('t_id', sort=False):
        collapsed_df['t_id'].append(t_id)
        collapsed_df['crf_features'].append([features(pt_sq, base_clf_feature_name, r, i) for i, r in enumerate(pt_sq.itertuples())])
        collapsed_df['ad'].append([str(i) for i in pt_sq.ad])
    collapsed_df = pd.DataFrame(collapsed_df)
    return collapsed_df

def cv_linear_CRF(base_clf_feature_name, segment_model):
    accus, precs, recall, f1s = [], [], [], []
    kf = KFold(n_splits=10, random_state=random_state, shuffle=False)
    accus, precs, recall, f1s = [], [], [], []
    for train_idxs, test_idxs in kf.split(train_df):
        train_x_y =  train_df_segments[train_df_segments.t_id.isin(train_df.iloc[train_idxs].id)]
        test_x_y = train_df_segments[train_df_segments.t_id.isin(train_df.iloc[test_idxs].id)]
        train_x, train_y = train_x_y, train_x_y.ad.tolist()
        test_x, test_y = test_x_y, test_x_y.ad.tolist()

        # refit the base sentence 'embedding layer'. We've already found optimal hyper params - so no need to CV this model
        segment_model.fit(train_x.speech_sent, train_y)
        train_x[base_clf_feature_name] = segment_model.predict_proba(train_x.speech_sent).tolist()
        test_x[base_clf_feature_name] = segment_model.predict_proba(test_x.speech_sent).tolist()

        # Collapse sequences back into DataFrame with CRF features.
        seq_data_train = crf_featurize(train_x, train_y, base_clf_feature_name)
        seq_data_test = crf_featurize(test_x, test_y, base_clf_feature_name)

        params_space = {
            'c1': scipy.stats.expon(scale=0.5).rvs(size=15),
            'c2': scipy.stats.expon(scale=0.05).rvs(size=15),
        }

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            all_possible_transitions=True
        )

        gs = GridSearchCV(crf, params_space, cv=5, n_jobs=6)
        gs.fit(seq_data_train.crf_features.tolist(), seq_data_train.ad)

        # one last fit on all train_data
        crf.set_params(**gs.best_params_)
        crf.fit(seq_data_train.crf_features.tolist(), seq_data_train.ad)
        last_test_labels = [p[-1] for p in seq_data_test.ad]
        preds = [p[-1] for p in crf.predict(seq_data_test.crf_features)]
        print(precision_recall_fscore_support(last_test_labels, preds, average='binary', pos_label='1'))
        accus.append(accuracy_score(last_test_labels, preds))
        p, r, f1, _ = precision_recall_fscore_support(last_test_labels, preds, average='binary', pos_label='1')
        precs.append(p)
        recall.append(r)
        f1s.append(f1)
    
    return accus, precs, recall, f1s, crf

# %%
accus, precs, recall, f1s, svm_crf_model = cv_linear_CRF('clf_svm_preds', clf_svm_segment)

# %%
print('avg accu:', round(sum(accus) / len(accus), 2))
print('avg precs:', round(sum(precs) / len(precs),2))
print('avg recall:', round(sum(recall) / len(recall),2))
print('avg f1s:', round(sum(f1s) / len(f1s),2))

# %%
# With Time - set include_time in features() to True.
accus, precs, recall, f1s, svm_crf_time_model = cv_linear_CRF('clf_svm_preds', clf_svm_segment)
print('avg accu:', round(sum(accus) / len(accus), 2))
print('avg precs:', round(sum(precs) / len(precs),2))
print('avg recall:', round(sum(recall) / len(recall),2))
print('avg f1s:', round(sum(f1s) / len(f1s),2))

# %%
# %time
accus, precs, recall, f1s, xgb_crf_model = cv_linear_CRF('clf_xgb_preds', clf_xgb_segments)
print('avg accu:', round(sum(accus) / len(accus), 2))
print('avg precs:', round(sum(precs) / len(precs),2))
print('avg recall:', round(sum(recall) / len(recall),2))
print('avg f1s:', round(sum(f1s) / len(f1s),2))

# %%
xgb_crf_model.get_params

# %%
print('avg accu:', round(sum(accus) / len(accus), 2))
print('avg precs:', round(sum(precs) / len(precs),2))
print('avg recall:', round(sum(recall) / len(recall),2))
print('avg f1s:', round(sum(f1s) / len(f1s),2))

# %%
print('avg accu:', round(sum(accus) / len(accus), 2))
print('avg precs:', round(sum(precs) / len(precs),2))
print('avg recall:', round(sum(recall) / len(recall),2))
print('avg f1s:', round(sum(f1s) / len(f1s),2))

# %%
def feat_impor(pipe):
    feat_impor = [(pipe.steps[0][1].get_feature_names()[i], pipe.steps[1][1].feature_importances_[i]) for i in np.argsort(pipe.steps[1][1].feature_importances_)[::-1][0:20]]
    labs = [i[0] for i in feat_impor]
    scores = [i[1] for i in feat_impor]
    return labs, scores

# %%
def plt_feats(clf_labels, clf_impor, rgs_labels, rgs_impor):
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(8, 5))
    ax1.barh(range(len(clf_impor)), clf_impor, label='AD Classification')
    ax1.set_yticks(range(len(clf_labels)))
    ax1.set_ylabel('TF-IDF Feature')
    ax1.set_yticklabels(clf_labels)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_xlabel('Feature Importance')

    ax2.barh(range(len(rgs_labels)), rgs_impor, label='MMSE Regression')
    ax2.set_yticks(range(len(rgs_labels)))
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.set_yticklabels(rgs_labels)
    ax2.set_xlabel('Feature Importance')

    # plt.legend([bar1, bar2], [bar1.get_label(), bar2.get_label()], fontsize=12)
    plt.tight_layout()
    plt.savefig('./feat_impor_xgboost.png')
    plt.show()

# %%
# clf_labels, clf_impor = 
# rgs_labels, rgs_impor =
plt_feats(*feat_impor(clf_xgb_par), * feat_impor(rgs_xgb_par))

# %% [markdown]
# ## BERT (type) model Experimentaton

# %%
# Embedding function
def bert_embed(text: pd.Series, tokenizer, model):
    tokenized = text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)))

    # pad so can be treated as one batch
    max_len = max([len(i) for i in tokenized.values])
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # attention mask - zero out attention scores where there is no input to be processed (i.e. is padding)
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # check if multiple GPUs are available
    multi_gpu = torch.cuda.device_count() > 1

    if torch.cuda.is_available():
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    last_hidden_states = last_hidden_states[0]
    if device.type == 'cuda':
        last_hidden_states = last_hidden_states.cpu()
    features = last_hidden_states[:,0,:].numpy()
    return features, attention_mask

# %%
# linear classifier fit/transform
def fit_transform(features, labels: list, mmse: list):
    def cv10_avg_nn(score):
        return cv10_avg(score, lr_clf, features, labels)

    # AD classification task
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=random_state)
    parameters = {'C': np.linspace(0.0001, 100, 20)}
    grid_search = GridSearchCV(LogisticRegression(), parameters)
    grid_search.fit(train_features, train_labels)
    print('best parameters:', grid_search.best_params_)
    print('best scores: ', grid_search.best_score_)
    lr_clf = LogisticRegression(**grid_search.best_params_)
    lr_clf.fit(train_features, train_labels)
    preds = lr_clf.predict(test_features)
    print('prec, rec, f1 test', precision_recall_fscore_support(test_labels, preds))
    print(f'accu:{cv10_avg_nn("accuracy")} prec:{cv10_avg_nn("precision")}, rec:{cv10_avg_nn("recall")}, f1:{cv10_avg_nn("f1")}')
          
    # MMSE regression task
    # remove missing row
    reg_features, reg_scores = pd.DataFrame(features).drop([i for i, s in enumerate(mmse) if s == '']).to_numpy(), [s for s in mmse if s != '']
    train_features, test_features, train_scores, test_scores = train_test_split(reg_features, reg_scores, random_state=random_state)
    parameters = {'alpha': np.linspace(0.001, 100, 20)}
    grid_search = GridSearchCV(Ridge(), parameters)
    grid_search.fit(train_features, train_scores)
    print('best parameters:', grid_search.best_params_)
    print('best scores: ', grid_search.best_score_)
    reg_model = Ridge(**grid_search.best_params_)
    reg_model.fit(train_features, train_scores)
    preds = reg_model.predict(test_features)
    print('rmse test:', sqrt(mean_squared_error(test_scores, preds)))
    print('rmse cv:', cross_val_score(reg_model, reg_features, reg_scores, cv=10, scoring='neg_root_mean_squared_error').sum() / 10)
    return lr_clf, reg_model

# %%
# BERT Large
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-large-uncased')

# roBERTa base
# model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')

# %%
def load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights):
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return tokenizer, model

# %%
def run_model(model_class, tokenizer_class, pretrained_weights, text):
    tokenizer, model = load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights)
    features, _ = bert_embed(text, tokenizer, model)
    clf_model, reg_model = fit_transform(features, train_df.ad, train_df.mmse)
    return tokenizer, model, features, clf_model, reg_model

# %%
# Distil BERT - par speech
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_par_speech)

# %%
# Distil roBERTa
model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'distilroberta-base')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_par_speech)

# %%
# BERT Base
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_par_speech)

# %%
# Distil BERT - par + inv speech
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_speech)

# %%
# Distil roBERT - par + inv speech
model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'distilroberta-base')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_speech)

# %%
# BERT Base -  par + inv speech
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_speech)

# %%
# roBERTa Base - par + inv speech
model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_speech)

# %%
# roBERTa Large - par + inv speech
model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-large')
tokenizer, model, features, clf_model, reg_model = run_model(model_class, tokenizer_class, pretrained_weights, train_df.joined_all_speech)

# %% [markdown]
# ### Test Set submission 
# output in the challenge format.
# 
# We'll use the SVM, DistilBERT and SVM/CRF models

# %%
test_df = parse_test_data()

# %%
def assign_model_preds(model, features):
    out_file = pd.read_csv('../data/test/test_results.txt', delimiter=';')
    out_file.columns = [c.strip() for c in out_file.columns]
    out_file['ID'] = out_file.ID.str.strip()
    model_preds = {t_id: model.predict([feature_row])[0] for feature_row, t_id in zip(features, test_df.id)}
    for t_id in out_file.ID:
        out_file.loc[out_file.ID == t_id, 'Prediction'] = model_preds[t_id]
    return out_file

# %%
# Distil BERT / PAR
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer, model = load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights)
bert_features, _ = bert_embed(test_df.joined_all_par_speech, tokenizer, model)

# %%
out_file_dstl_bert_clf = assign_model_preds(clf_model, bert_features)
out_file_dstl_bert_clf.to_csv('../data/test/results/test_results-classif-1.txt', sep=';', index=False)

# %%
out_file_dstl_bert_rgs = assign_model_preds(reg_model, bert_features)
out_file_dstl_bert_rgs.to_csv('../data/test/results/test_results-regression-1.txt', sep=';', index=False)

# %%
# Distill BERT / PAR + INV
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer, model = load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights)
bert_features, _ = bert_embed(test_df.joined_all_speech, tokenizer, model)

# %%
out_file_dstl_bert_clf = assign_model_preds(clf_model, bert_features)
out_file_dstl_bert_clf.to_csv('../data/test/results/test_results-classif-2.txt', sep=';', index=False)

# %%
out_file_dstl_bert_rgs = assign_model_preds(reg_model, bert_features)
out_file_dstl_bert_rgs.to_csv('../data/test/results/test_results-regression-2.txt', sep=';', index=False)

# %%
# SVM/PAR preds
out_file_clf_svm = assign_model_preds(clf_svm, test_df.joined_all_par_speech)
out_file_clf_svm.to_csv('../data/test/results/test_results-classif-3.txt', sep=';', index=False)

# %%
out_file_rgs_svm = assign_model_preds(rgs_svm, test_df.joined_all_par_speech)
out_file_rgs_svm.to_csv('../data/test/results/test_results-regression-3.txt', sep=';', index=False)

# %%
# SVM/CRF / PAR_SPLT

# %%
segmented_speech_test = test_df.apply(lambda r: pd.DataFrame({'t_id': r.id, 'part_id': [str(i) for i, _ in enumerate(r.clean_par_speech)], 'speech_sent': r.clean_par_speech}), axis=1).tolist()
test_df_segments = pd.concat(segmented_speech_test).reset_index(drop=True)
in_features_probas = clf_svm_segment.predict_proba(test_df_segments.speech_sent)

# %%
test_time_dim_segments = add_time_features(test_df)

# %%
test_df_segments = test_df_segments.merge(test_time_dim_segments, on=['t_id', 'part_id'])

# %%
test_df_segments['base_preds'] = in_features_probas.tolist()

# %%
# Collapse sequences back into DataFrame with CRF features.
def crf_featurize_no_label(X, base_clf_feature_name, include_time=False):
    collapsed_df = {'t_id': [], 'crf_features': []}
    for t_id, pt_sq in X.groupby('t_id', sort=False):
        collapsed_df['t_id'].append(t_id)
        collapsed_df['crf_features'].append([features(pt_sq, base_clf_feature_name, r, i, include_time=include_time) for i, r in enumerate(pt_sq.itertuples())])
    collapsed_df = pd.DataFrame(collapsed_df)
    return collapsed_df

# %%
def assign_to_file(preds_dict):
    out_file = pd.read_csv('../data/test/test_results.txt', delimiter=';')
    out_file.columns = [c.strip() for c in out_file.columns]
    out_file['ID'] = out_file.ID.str.strip()
    for t_id in out_file.ID:
        out_file.loc[out_file.ID == t_id, 'Prediction'] = preds_dict[t_id]
    return out_file

# %%
test_crf_features = crf_featurize_no_label(test_df_segments, 'base_preds', include_time=False)

# %%
test_svm_crf_preds = {t_id : p[-1] for t_id, p in zip(test_crf_features.t_id, svm_crf_model.predict(test_crf_features.crf_features))}

# %%
out_file_svm_crf = assign_to_file(test_svm_crf_preds)
out_file_svm_crf.to_csv('../data/test/results/test_results-classif-4.txt', sep=';', index=False)

# %%
# SVM/CRF / PAR_SPLT+T
# re-run the CRF model training with the extra time based features before executing.

# %%
test_crf_features_inc_time = crf_featurize_no_label(test_df_segments, 'base_preds', include_time=True)

# %%
test_svm_crf_preds = {t_id : p[-1] for t_id, p in zip(test_crf_features_inc_time.t_id, svm_crf_time_model.predict(test_crf_features_inc_time.crf_features))}

# %%
out_file_svm_crf = assign_to_file(test_svm_crf_preds)
out_file_svm_crf.to_csv('../data/test/results/test_results-classif-5.txt', sep=';', index=False)


