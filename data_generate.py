import numpy as np
from scipy.sparse import vstack, csr_matrix
import gym, recogym
import pandas as pd
import math
from copy import deepcopy
from tqdm import tqdm, trange
from recogym import env_1_args, Configuration
from recogym.agents import RandomAgent, random_args

def train_data(data, num_products, va_ratio=0.2, te_ratio=0.2):
        data = pd.DataFrame().from_dict(data)

        features = []
        actions = []
        pss = []
        deltas = []
        set_flags = []

        for user_id in tqdm(data['u'].unique()):
            tmp_feature = []
            tmp_action = []
            tmp_ps = []
            tmp_delta = []
            tmp_set_flag = []

            views = np.zeros((0, num_products))
            history = np.zeros((0, 1))
            for _, user_datum in data[data['u'] == user_id].iterrows():
                assert (not math.isnan(user_datum['t']))
                if user_datum['z'] == 'organic':
                    assert (math.isnan(user_datum['a']))
                    assert (math.isnan(user_datum['c']))
                    assert (not math.isnan(user_datum['v']))

                    view = int(user_datum['v'])

                    tmp_view = np.zeros(num_products)
                    tmp_view[view] = 1

                    # Append the latest view at the beginning of all views.
                    views = np.append(tmp_view[np.newaxis, :], views, axis = 0)
                    history = np.append(np.array([user_datum['t']])[np.newaxis, :], history, axis = 0)
                else:
                    assert (user_datum['z'] == 'bandit')
                    assert (not math.isnan(user_datum['a']))
                    assert (not math.isnan(user_datum['c']))
                    assert (math.isnan(user_datum['v']))

                    action = int(user_datum['a'])
                    delta = int(user_datum['c'])
                    ps = user_datum['ps']
                    time = user_datum['t']

                    train_views = views

                    feature = np.sum(train_views, axis = 0)
                    feature = feature/np.linalg.norm(feature)

                    tmp_feature.append(feature)
                    tmp_action.append(action)
                    tmp_delta.append(delta)
                    tmp_ps.append(ps)
                    tmp_set_flag.append(0)

            tmp_set_flag = np.array(tmp_set_flag)
            va_num = math.ceil(va_ratio*tmp_set_flag.shape[0]) 
            te_num = math.ceil(te_ratio*tmp_set_flag.shape[0]) 
            if va_num + te_num < tmp_set_flag.shape[0]:
                tmp_set_flag[-1*(va_num+te_num):] = 1
                tmp_set_flag[-1*te_num:] = 2

            features.append(csr_matrix(np.array(tmp_feature)))
            actions.append(tmp_action)
            deltas.append(tmp_delta)
            pss.append(tmp_ps)
            set_flags.append(tmp_set_flag)

        return features, actions, deltas, pss, set_flags

def dump_svm(f, X, y_idx, y_propensity, y_value):
    if not hasattr(f, "write"):
        f = open(f, "w")
    X_is_sp = int(hasattr(X, "tocsr"))
    #y_is_sp = int(hasattr(y, "tocsr"))

    value_pattern = "%d:%.6g"
    label_pattern = "%d:%d:%.16g"
    line_pattern = "%s %s\n"
    
    for i, d in enumerate(zip(y_idx, y_value, y_propensity)):
        if X_is_sp:
            span = slice(X.indptr[i], X.indptr[i + 1])
            row = zip(X.indices[span], X.data[span])
        else:
            nz = X[i] != 0
            row = zip(np.where(nz)[0], X[i, nz])

        s = " ".join(value_pattern % (j, x) for j, x in row)
        labels_str = label_pattern % d 
        feat = (labels_str, s)
        f.write(line_pattern % feat)
    
    return

def merge_list(ls):
    res = []
    for l in ls:
        res.extend(l)
    return res

P = 100
U = 40000

env_1_args['random_seed'] = 8964
env_1_args['num_products'] = P
env_1_args['K'] = 5
env_1_args['number_of_flips'] = P//2
env_1_args['prob_bandit_to_organic'] = 0.99
env_1_args['prob_organic_to_bandit'] = 0.1


env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

data = env.generate_logs(U)
data.to_csv('data_%d_%d.csv'%(P, U), index=False)
#data = pd.read_csv('./50w/data.csv')

features, actions, deltas, pss, set_flags = train_data(data, P)
features, actions, deltas, pss, set_flags = vstack(features), \
        np.array(merge_list(actions)), \
        np.array(merge_list(deltas)), \
        np.array(merge_list(pss)), \
        np.hstack(set_flags)
with open('tr.nonmerge.uniform.svm', 'w') as tr, \
        open('va.nonmerge.uniform.svm', 'w') as va, \
        open('te.nonmerge.uniform.svm', 'w') as te:
            dump_svm(tr, vstack(features[set_flags==0]), \
                    actions[set_flags==0], \
                    pss[set_flags==0], \
                    deltas[set_flags==0])
            dump_svm(va, vstack(features[set_flags==1]), \
                    actions[set_flags==1], \
                    pss[set_flags==1], \
                    deltas[set_flags==1])
            dump_svm(te, vstack(features[set_flags==2]), \
                    actions[set_flags==2], \
                    pss[set_flags==2], \
                    deltas[set_flags==2])

with open('label.svm', 'w') as label:
    for i in range(P):
        label.write('%d:1\n'%i)
