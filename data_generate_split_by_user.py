import numpy as np
from scipy.sparse import vstack, csr_matrix
import gym, recogym
import pandas as pd
import math
from functools import partial
from copy import deepcopy
from tqdm import tqdm, trange
from recogym import env_1_args, Configuration
from recogym.agents import RandomAgent, random_args
import os, sys
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
try:
        import cPickle as pickle
except:
        import pickle

def process_helper(num_products, num_users, root, va_ratio, te_ratio, det_reward, det_reward_thres, accum, use_userid, user_id):
    tmp_feature = []
    tmp_action = []
    tmp_ps = []
    tmp_delta = []
    tmp_set_flag = []
    
    if use_userid:
        feature_dim = num_products+num_users
    else:
        feature_dim = num_products
    views = np.zeros((0, feature_dim))
    history = np.zeros((0, 1))
    #for _, user_datum in data[data['u'] == user_id].iterrows():
    #for _, user_datum in data.get_group(user_id).iterrows():
    f = open('%s/%08d.pkl'%(root, user_id), 'rb')
    user_data = pickle.load(f)
    f.close()
    for _, user_datum in user_data.iterrows():
        assert (not math.isnan(user_datum['t']))
        if user_datum['z'] == 'organic':
            assert (math.isnan(user_datum['a']))
            assert (math.isnan(user_datum['c']))
            assert (not math.isnan(user_datum['v']))

            view = int(user_datum['v'])

            tmp_view = np.zeros(feature_dim)
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
            if det_reward:
                delta = 1 if user_datum['ctr'] > det_reward_thres else 0
            else:
                delta = int(user_datum['c'])
            ps = user_datum['ps']
            time = user_datum['t']
            
            train_views = views
            feature = np.sum(train_views, axis = 0)
            feature = feature/np.linalg.norm(feature)
            if use_userid:
                feature[num_products+user_id] = 1

            if not accum:
                views = np.zeros((0, feature_dim))

            tmp_feature.append(feature)
            tmp_action.append(action)
            tmp_delta.append(delta)
            tmp_ps.append(ps)
            tmp_set_flag.append(-1) # user without enough bandits will be removed
    tmp_set_flag = np.array(tmp_set_flag)
    va_num = math.ceil(va_ratio*tmp_set_flag.shape[0]) 
    te_num = math.ceil(te_ratio*tmp_set_flag.shape[0]) 
    if va_num + te_num < tmp_set_flag.shape[0]:
        tmp_set_flag[:-1*(va_num+te_num)] = 0
        tmp_set_flag[-1*(va_num+te_num):] = 1
        tmp_set_flag[-1*te_num:] = 2

    return csr_matrix(np.array(tmp_feature)), \
            np.array(tmp_action), \
            np.array(tmp_delta), \
            np.array(tmp_ps), \
            np.array(tmp_set_flag)

def gen_data(data, num_products, num_users, root, start_user_id=0, va_ratio=0.2, te_ratio=0.2, det_reward=False, det_reward_thres=0.02, accum=False, use_userid=False):
    #data = pd.DataFrame().from_dict(data)
    data = data.groupby('u')
    root = os.path.join(root, 'pkls')
    os.makedirs(root, exist_ok=True)
    for i in tqdm(range(start_user_id, start_user_id + num_users)):
        f = open('%s/%08d.pkl'%(root, i), 'wb')
        pickle.dump(data.get_group(i), f)
        f.close()

    #with Pool(8) as p:
    #    #output = list(tqdm(p.imap(process_helper, data['u'].unique(), chunksize=10000), total=num_users))
    #    output = list(tqdm(p.map(process_helper, np.arange(num_users)), total=num_users))
    output = process_map(partial(process_helper, num_products, num_users, root, \
            va_ratio, te_ratio, det_reward, det_reward_thres, accum, use_userid), \
            np.arange(start_user_id, start_user_id + num_users), max_workers=8, chunksize=1)
    features, actions, deltas, pss, set_flags = zip(*output)

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

def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    P = 3000
    U = 500000
    use_userid = False 
    start_user_id = int(sys.argv[4])*U
    
    env_1_args['random_seed'] = 8964
    env_1_args['random_seed_for_user'] = int(sys.argv[3]) 
    env_1_args['num_products'] = P
    env_1_args['K'] = 32
    env_1_args['sigma_omega'] = 0.0  # default 0.1, the varaince of user embedding changes with time.
    env_1_args['number_of_flips'] = 0 #P//2
    env_1_args['prob_leave_bandit'] = float(sys.argv[2])
    env_1_args['prob_leave_organic'] = 0.0
    env_1_args['prob_bandit_to_organic'] = 1 - env_1_args['prob_leave_bandit']
    env_1_args['prob_organic_to_bandit'] = 0.05
    env_1_args['deterministic_reward'] = True
    #with open('%s/env_setting.info'%root, 'w') as info:
    #    print(env_1_args)
    #    print('num_of_users: %d, num_of_expected_context: %d'%(U, U/env_1_args['prob_leave_bandit']))
    #    print(env_1_args, file=info)
    #    print('num_of_users: %d, num_of_expected_context: %d'%(U, U/env_1_args['prob_leave_bandit']), file=info)
    
    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)
    
    env.reset_random_seed(epoch = env_1_args['random_seed_for_user'])
    #data = env.generate_logs(U, unique_user_id=start_user_id)
    #data.to_csv('%s/data_%d_%d.csv'%(root, P, U), index=False)
    #sys.exit(0)
    data = pd.read_csv('%s/data_%d_%d.csv'%(root, P, U))
    print('Data loaded!')
    
    features, actions, deltas, pss, set_flags = gen_data(data, P, U, root, use_userid=use_userid, start_user_id=start_user_id)
    tr_num = int(U*0.6)
    va_num = int(U*0.2)
    with open('%s/tr.nonmerge.uniform.svm'%root, 'w') as tr, \
            open('%s/va.nonmerge.uniform.svm'%root, 'w') as va, \
            open('%s/te.nonmerge.uniform.svm'%root, 'w') as te:
                dump_svm(tr, vstack(features[:tr_num]), \
                        np.hstack(actions[:tr_num]), \
                        np.hstack(pss[:tr_num]), \
                        np.hstack(deltas[:tr_num]))
                dump_svm(va, vstack(features[tr_num:(tr_num+va_num)]), \
                        np.hstack(actions[tr_num:(tr_num+va_num)]), \
                        np.hstack(pss[tr_num:(tr_num+va_num)]), \
                        np.hstack(deltas[tr_num:(tr_num+va_num)]))
                dump_svm(te, vstack(features[(tr_num+va_num):]), \
                        np.hstack(actions[(tr_num+va_num):]), \
                        np.hstack(pss[(tr_num+va_num):]), \
                        np.hstack(deltas[(tr_num+va_num):]))
    
    with open('%s/label.svm'%root, 'w') as label:
        for i in range(P):
            label.write('%d:1\n'%i)

if __name__ == '__main__':
    main()
