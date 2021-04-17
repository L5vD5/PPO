import datetime,gym,os,pybullet_envs,psutil,time,os
import scipy.signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete

print("Pytorch version:[%s]."%(torch.__version__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:[%s]."%(device))

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x
    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    if with_min_and_max:
        global_min = (np.min(x) if len(x) > 0 else np.inf)
        global_max = (np.max(x) if len(x) > 0 else -np.inf)
        return mean, std, global_min, global_max
    return mean, std

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input:
        vector x, [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, odim, adim, size=5000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, adim), dtype=np.float32)
        self.act_old_buf = np.zeros(self._combined_shape(size, adim), dtype=np.float32) # added
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, act_old):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_old_buf[self.ptr] = act_old    # added
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf, self.act_old_buf]  # modified

##### Model construction #####
class MLP(nn.Module):
    def __init__(self, o_dim=24, hdims=[64,64], actv=nn.ReLU(),
                 output_actv=None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()

        self.o_dim = o_dim
        self.hdims = hdims
        self.actv = actv
        self.ouput_actv = output_actv
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        self.layers = []
        prev_hdim = self.o_dim
        for hdim in self.hdims[:-1]:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(actv)
            prev_hdim = hdim
        self.layers.append(nn.Linear(prev_hdim, hdims[-1]))

        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

    def forward(self, inputs):
        x = inputs
        if self.ouput_actv is None:
            x = self.net(x)*self.output_scale
        else:
            x = self.net(x)
            x = self.actv(x)*self.output_scale
        return x.sueeze() if self.output_squeeze else x

class CategoricalPolicy(nn.Module):
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(), output_actv=None):
        super(CategoricalPolicy, self).__init__()

        self.output_actv = output_actv
        self.net = MLP(odim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.logits = nn.Linear(in_features=hdims[-1], out_features=adim)

    def forward(self, x, a):
        output = self.net(x)
        logits = self.logits(output)
        if self.output_actv:
            logits = self.output_actv(logits)
        prob = F.softmax(logits, dim=-1)
        dist = Categorical(probs=prob)
        pi = dist.sample()
        logp_pi = dist.log_prob(pi)
        logp = dist.log_prob(a)
        return pi, logp_pi, logp, pi

class GaussianPolicy(nn.Module):
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(), output_actv=None):
        super(GaussianPolicy, self).__init__()

        self.output_actv = output_actv
        self.net = MLP(odim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.mu = nn.Linear(in_features=hdims[-1], out_features=adim)
        self.log_std = nn.Parameter(-0.5*torch.ones(adim))

    def forward(self, x, a=None):
        output = self.net(x)
        mu = self.mu(output)

        if self.output_actv:
            mu = self.output_actv(mu)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        # gaussian likelihood
        logp_pi = policy.log_prob(pi).sum(dim=1)
        logp = policy.log_prob(a).sum(dim=1)
        return pi, logp, logp_pi, mu

    def select_action(self,o,_eval=False):
        pi,_, _, mu = self.forward(torch.Tensor(o.reshape(1, -1)))
        return mu.cpu().detach().numpy()[0] if _eval else pi.cpu().detach().numpy()[0]

class ActorCritic(nn.Module):
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(),
                 output_actv=None, policy=None, action_space=None):
        super(ActorCritic,self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(odim, adim, hdims, actv, output_actv)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(odim, adim, hdims, actv, output_actv)
        self.vf_mlp = MLP(odim, hdims=hdims+[1],
                          actv=actv, output_actv=output_actv)

    def forward(self, x, a):
        pi, logp, logp_pi, mu = self.policy(x, a)
        v = self.vf_mlp(x)
        return pi, logp, logp_pi, mu, v

# Configuration, set model parameter
class Config:
    def __init__(self):
        # Model
        self.hdims = [256,256]
        #Graph
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        #Buffer
        self.steps_per_epoch = 5000
        self.gamma = 0.99
        self.lam = 0.95
        #Update
        self.train_pi_iters = 100
        self.train_v_iters = 100
        self.target_kl = 0.01
        self.epochs = 1000
        self.max_ep_len = 1000
        self.print_every = 10
        self.evaluate_every = 10

def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

class PPOAgent(Config):
    def __init__(self):
        self.env, self.eval_env = get_envs()
        odim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = ActorCritic(odim,adim,self.hdims,**ac_kwargs)
        self.buf = PPOBuffer(odim=odim,adim=adim,size=self.steps_per_epoch,
                             gamma=self.gamma,lam=self.lam)

        # Optimizers
        self.train_pi = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=self.pi_lr)
        self.train_v = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=self.vf_lr)

    def update(self):
        self.actor_critic.train()
        self.actor_critic.cuda()

        obs, act, adv, ret, logp_act, act_old = [torch.Tensor(x) for x in self.buf.get()]

        obs = torch.FloatTensor(obs).to(device)
        act = torch.FloatTensor(act).to(device)
        act_old = torch.FloatTensor(act_old).to(device)
        adv = torch.FloatTensor(adv).to(device)
        ret = torch.FloatTensor(ret).to(device)
        logp_a_old = torch.FloatTensor(logp_act).to(device)

        _, _, _, logp_a, _ = self.actor_critic.policy(obs, act_old)

        ratio = (logp_a - logp_a_old).exp()
        min_adv = torch.where(adv > 0, (1 + self.clip_ratio) * adv,
                              (1 - self.clip_ratio) * adv)
        pi_l_old = -(torch.min(ratio * adv, min_adv)).mean()
        ent = (-logp_a).mean()

        for i in range(self.train_pi_iters):
            _, _, _, logp_a,_ = self.actor_critic.policy(obs, act_old)

            ratio = (logp_a - logp_a_old).exp()
            min_adv = torch.where(adv > 0, (1 + self.clip_ratio) * adv,
                                  (1 - self.clip_ratio) * adv)
            pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
            # Policy gradient step
            self.train_pi.zero_grad()
            pi_loss.backward()
            self.train_pi.step()

            _, _, _, logp_a, _ = self.actor_critic.policy(obs, act_old)
            kl = (logp_a_old - logp_a).mean()
            if kl > 1.5 * self.target_kl:
                break

        v = self.actor_critic.vf_mlp(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(self.train_v_iters):
            v = self.actor_critic.vf_mlp(obs)
            v_loss = F.mse_loss(v, ret)
            # Value gradient step
            self.train_v.zero_grad()
            v_loss.backward()
            self.train_v.step()

    def train(self):
        start_time = time.time()
        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            if (epoch == 0) or (((epoch + 1) % self.print_every) == 0):
                print("[%d/%d]" % (epoch + 1, self.epochs))
            # self.actor_critic.eval()
            # self.actor_critic.cpu()
            for t in range(self.steps_per_epoch):
                a_scaled, logp_pi, a, _, v_t = self.actor_critic(
                    torch.Tensor(o.reshape(1, -1)))  # pi, logp_pi, v, logp

                o2, r, d, _ = self.env.step(a_scaled.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                n_env_step += 1

                # save and log
                self.buf.store(o, a_scaled.detach().numpy(), r, v_t.item(), logp_pi.detach().numpy(),
                               a.detach().numpy())

                # Update obs (critical!)
                o = o2

                terminal = d or (ep_len == self.max_ep_len)
                if terminal or (t == (self.steps_per_epoch - 1)):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else self.actor_critic.vf_mlp(torch.Tensor(o.reshape(1, -1))).item()
                    self.buf.finish_path(last_val)
                    o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

            # Perform PPO update!
            self.update()

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.epochs, epoch / self.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.max_ep_len)):
                    a = self.actor_critic.policy.select_action(np.array(o))
                    o, r, d, _ = self.eval_env.step(a[0])
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))
        print("Done.")

        self.env.close()
        self.eval_env.close()


agent = PPOAgent()
agent.train()
