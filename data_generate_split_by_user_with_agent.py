import numpy as np
from scipy.sparse import vstack, csr_matrix
import gym, recogym
import pandas as pd
import math
import sys
from copy import deepcopy
from tqdm import tqdm, trange
from recogym import env_1_args, Configuration
from recogym.agents import RandomAgent, random_args, FeatureProvider, Agent
from multiprocessing import Pool
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from numpy.random.mtrand import RandomState
import util


class CrossFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defined interface of setting/getting features"""

    def __init__(self, config):
        super(CrossFeatureProvider, self).__init__(config)
        self.feature_data = None

    def observe(self, observation):
        """Consider an Organic Event for a particular user"""
        for session in observation.sessions():
            self.feature_data[session['v']] += 1

    def features(self, observation):
        """Provide feature values adjusted to a particular feature set"""
        return self.feature_data

    def reset(self):
        self.feature_data = np.zeros((self.config.num_products))

class ModelBasedAgent(Agent):
    def __init__(self, env, feature_provider, model_path, V, mode):
        # Set environment as an attribute of Agent.
        self.env = env
        self.feature_provider = feature_provider
        self.model = torch.load(model_path)
        self.model.eval()
        self.rng = RandomState(self.env.random_seed)
        self.V = V
        self.mode = mode
        self.reset()

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past history"""
        self.feature_provider.observe(observation)
        if self.mode == 'uniform':
            probs = np.ones(self.env.num_products) / self.env.num_products
            action = self.rng.choice(self.env.num_products, p=probs)
        else:
            feature = self.feature_provider.features(observation)
            feature = feature/np.linalg.norm(feature)
            U = Variable(util.csr2tensor(feature))
            with torch.no_grad():
                P, Q = self.model(U, self.V)
                logits = torch.mm(P, Q.T)
                if self.mode == 'greedy':
                    action = np.argmax(logits.detach().cpu().numpy().flatten())
                    probs = np.zeros_like(logits.detach().cpu().numpy().flatten())
                    probs[action] = 1.0
                elif self.mode == 'is':
                    probs = F.softmax(logits, dim=1).detach().cpu().numpy().flatten()
                    action = self.rng.choice(self.env.num_products, p=probs)
                else:
                    raise ValueError('No such mode:', self.mode)

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': probs[action],
                'ps-a': probs if self.env.with_ps_all else (),
            }
        }

    def reset(self):
        self.feature_provider.reset()

def gen_data(data, num_products, va_ratio=0.2, te_ratio=0.2):
    data = pd.DataFrame().from_dict(data)

    global process_helper
    def process_helper(user_id):
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

    with Pool(8) as p:
        output = p.map(process_helper, data['u'].unique())
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
    P = 100
    U = 80000

    with open('%s/label.svm'%root, 'w') as label:
        for i in range(P):
            label.write('%d:1\n'%i)
    
    env_1_args['random_seed'] = 8964
    env_1_args['random_seed_for_user'] = 1
    env_1_args['num_products'] = P
    env_1_args['K'] = 5
    env_1_args['sigma_omega'] = 0.0  # default 0.1, the varaince of user embedding changes with time.
    env_1_args['number_of_flips'] = P//2
    env_1_args['prob_leave_bandit'] = float(sys.argv[2])
    env_1_args['prob_leave_organic'] = 0.0
    env_1_args['prob_bandit_to_organic'] = 1 - env_1_args['prob_leave_bandit']
    env_1_args['prob_organic_to_bandit'] = 0.1
    
    
    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)
    config = Configuration(env_1_args)

    V = util.read_data('%s/label.svm'%root, False)
    V = Variable(util.csr2tensor(V))
    agent = ModelBasedAgent(config, CrossFeatureProvider(config), sys.argv[3], V, sys.argv[4])

    data = env.generate_logs(U, agent)
    data.to_csv('%s/data_%d_%d_agent.csv'%(root, P, U), index=False)
    #data = pd.read_csv('%s/data_%d_%d_agent.csv'%(root, P, U))
    
    features, actions, deltas, pss, set_flags = gen_data(data, P)
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
    

if __name__ == '__main__':
    main()
