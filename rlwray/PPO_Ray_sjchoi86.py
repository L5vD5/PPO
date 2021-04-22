import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import numpy as np
import tensorflow as tf
from util import gpu_sess,suppress_tf_warning
from ppo import PPOBuffer,create_ppo_model,create_ppo_graph,update_ppo,\
    save_ppo_model,restore_ppo_model
np.set_printoptions(precision=2)
suppress_tf_warning() # suppress warning
gym.logger.set_level(40) # gym logger
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

# Rollout Worker
def get_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    eval_env = gym.make('AntBulletEnv-v0')
    _ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env

# Model
hdims = [256,256]

# Graph
clip_ratio = 0.2
pi_lr = 3e-4
vf_lr = 1e-3
epsilon = 1e-2

# Buffer
gamma = 0.99
lam = 0.95

# Update
train_pi_iters = 100
train_v_iters = 100
target_kl = 0.01
epochs = 1000
max_ep_len = 1000

# Worker
n_cpu = n_workers = 15
total_steps,evaluate_every,print_every = 1000,50,10
ep_len_rollout = 500
batch_size = 4096


class RolloutWorkerClass(object):
    """
    Worker without RAY (for update purposes)
    """

    def __init__(self, seed=1):
        self.seed = seed
        # Each worker should maintain its own environment
        import pybullet_envs, gym
        from util import suppress_tf_warning
        suppress_tf_warning()  # suppress TF warnings
        gym.logger.set_level(40)  # gym logger
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # Initialize PPO
        self.model, self.sess = create_ppo_model(env=self.env, hdims=hdims, output_actv=tf.nn.tanh)
        self.graph = create_ppo_graph(self.model,
                                      clip_ratio=clip_ratio, pi_lr=pi_lr, vf_lr=vf_lr, epsilon=epsilon)
        # Initialize model
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.sess.run(tf.global_variables_initializer())

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

    def get_action(self, o, deterministic=False):
        act_op = self.model['mu'] if deterministic else self.model['pi']
        return self.sess.run(act_op, feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]

    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.sess.run(self.model['pi_vars'] + self.model['v_vars'])
        return weight_vals

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            for w_idx, weight_tf_var in enumerate(self.model['pi_vars'] + self.model['v_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
        for w_idx, weight_tf_var in enumerate(self.model['pi_vars'] + self.model['v_vars']):
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})


@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """

    def __init__(self, worker_id=0, ep_len_rollout=1000):
        # Parse
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout
        # Each worker should maintain its own environment
        import pybullet_envs, gym
        from util import suppress_tf_warning
        suppress_tf_warning()  # suppress TF warnings
        gym.logger.set_level(40)  # gym logger
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # Replay buffers to pass
        self.o_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.a_buffer = np.zeros((self.ep_len_rollout, self.adim))
        self.r_buffer = np.zeros((self.ep_len_rollout))
        self.v_t_buffer = np.zeros((self.ep_len_rollout))
        self.logp_t_buffer = np.zeros((self.ep_len_rollout))
        # Create PPO model
        self.model, self.sess = create_ppo_model(env=self.env, hdims=hdims, output_actv=tf.nn.tanh)
        # Initialize model
        self.sess.run(tf.global_variables_initializer())
        # Buffer
        self.buf = PPOBuffer(odim=self.odim, adim=self.adim,
                             size=ep_len_rollout, gamma=gamma, lam=lam)

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def get_action(self, o, deterministic=False):
        act_op = self.model['mu'] if deterministic else self.model['pi']
        return self.sess.run(act_op, feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            for w_idx, weight_tf_var in enumerate(self.model['pi_vars'] + self.model['v_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
        for w_idx, weight_tf_var in enumerate(self.model['pi_vars'] + self.model['v_vars']):
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})

    def rollout(self):
        """
        Rollout
        """
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment
        # Loop
        for t in range(ep_len_rollout):
            a, v_t, logp_t = self.sess.run(
                self.model['get_action_ops'], feed_dict={self.model['o_ph']: self.o.reshape(1, -1)})
            o2, r, d, _ = self.env.step(a[0])
            # save and log
            self.buf.store(self.o, a, r, v_t, logp_t)
            # Update obs (critical!)
            self.o = o2
            if d:
                self.buf.finish_path(last_val=0.0)
                self.o = self.env.reset()  # reset when done

        last_val = self.sess.run(self.model['v'],
                                 feed_dict={self.model['o_ph']: self.o.reshape(1, -1)})
        self.buf.finish_path(last_val)
        return self.buf.get()


# Initialize PyBullet Ant Environment
eval_env = get_eval_env()
adim,odim = eval_env.action_space.shape[0],eval_env.observation_space.shape[0]
print ("Environment Ready. odim:[%d] adim:[%d]."%(odim,adim))

# Initialize Workers
ray.init(num_cpus=n_cpu,
         _memory = 5*1024*1024*1024,
         object_store_memory = 10*1024*1024*1024,
         _driver_object_store_memory = 1*1024*1024*1024)
tf.reset_default_graph()
R = RolloutWorkerClass(seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i,ep_len_rollout=ep_len_rollout)
           for i in range(n_workers)]
print ("RAY initialized with [%d] cpus and [%d] workers."%
       (n_cpu,n_workers))

time.sleep(1)

# Loop
start_time = time.time()
n_env_step = 0  # number of environment steps
for t in range(int(total_steps)):
    esec = time.time() - start_time

    # 1. Synchronize worker weights
    weights = R.get_weights()
    set_weights_list = [worker.set_weights.remote(weights) for worker in workers]

    # 2. Make rollout and accumulate to Buffers
    t_start = time.time()
    ops = [worker.rollout.remote() for worker in workers]
    rollout_vals = ray.get(ops)
    sec_rollout = time.time() - t_start

    # 3. Update
    t_start = time.time()  # tic
    """ 
    # Old update routine with batch learning 
    # Get stats before update
    feeds_list = []
    for rollout_val in rollout_vals:
        feeds = {k:v for k,v in zip(R.model['all_phs'],rollout_val)}
        feeds_list.append(feeds)
        pi_l_old, v_l_old, ent = R.sess.run(
            [R.graph['pi_loss'],R.graph['v_loss'],R.graph['approx_ent']],feed_dict=feeds)
    # Update the central agent 
    for _ in range(train_pi_iters):
        for r_idx,rollout_val in enumerate(rollout_vals):
            feeds = feeds_list[r_idx]
            _, kl = R.sess.run([R.graph['train_pi'],R.graph['approx_kl']],feed_dict=feeds)
            if kl > 1.5 * target_kl:
                print ("kl(%.3f) is higher than 1.5x(%.3f)"%(kl,target_kl))
                break
    for _ in range(train_v_iters):
        for r_idx,rollout_val in enumerate(rollout_vals):
            feeds = feeds_list[r_idx]
            R.sess.run(R.graph['train_v'],feed_dict=feeds)
    # Get stats after update
    for r_idx,rollout_val in enumerate(rollout_vals):
        feeds = feeds_list[r_idx]
        pi_l_new,v_l_new,kl,cf = R.sess.run(
            [R.graph['pi_loss'],R.graph['v_loss'],R.graph['approx_kl'],R.graph['clipfrac']],
            feed_dict=feeds)
    """
    # Mini-batch type of update
    for r_idx, rval in enumerate(rollout_vals):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = \
            rval[0], rval[1], rval[2], rval[3], rval[4]
        if r_idx == 0:
            obs_bufs, act_bufs, adv_bufs, ret_bufs, logp_bufs = \
                obs_buf, act_buf, adv_buf, ret_buf, logp_buf
        else:
            obs_bufs = np.concatenate((obs_bufs, obs_buf), axis=0)
            act_bufs = np.concatenate((act_bufs, act_buf), axis=0)
            adv_bufs = np.concatenate((adv_bufs, adv_buf), axis=0)
            ret_bufs = np.concatenate((ret_bufs, ret_buf), axis=0)
            logp_bufs = np.concatenate((logp_bufs, logp_buf), axis=0)
    n_val_total = obs_bufs.shape[0]
    for pi_iter in range(train_pi_iters):
        rand_idx = np.random.permutation(n_val_total)[:batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]
        feeds = {k: v for k, v in zip(R.model['all_phs'], buf_batches)}
        _, kl, pi_loss, ent = R.sess.run([R.graph['train_pi'], R.graph['approx_kl'],
                                          R.graph['pi_loss'], R.graph['approx_ent']],
                                         feed_dict=feeds)
        if kl > 1.5 * target_kl:
            # print ("  pi_iter:[%d] kl(%.3f) is higher than 1.5x(%.3f)"%(pi_iter,kl,target_kl))
            break
    for _ in range(train_v_iters):
        rand_idx = np.random.permutation(n_val_total)[:batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]
        feeds = {k: v for k, v in zip(R.model['all_phs'], buf_batches)}
        R.sess.run(R.graph['train_v'], feed_dict=feeds)
    sec_update = time.time() - t_start  # toc

    # Print
    if (t == 0) or (((t + 1) % print_every) == 0):
        print("[%d/%d] rollout:[%.1f]s pi_iter:[%d/%d] update:[%.1f]s kl:[%.4f] target_kl:[%.4f]." %
              (t + 1, total_steps, sec_rollout, pi_iter, train_pi_iters, sec_update, kl, target_kl))
        print("   pi_loss:[%.4f], entropy:[%.4f]" %
              (pi_loss, ent))

    # Evaluate
    if (t == 0) or (((t + 1) % evaluate_every) == 0):
        ram_percent = psutil.virtual_memory().percent  # memory usage
        print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
              (t + 1, total_steps, t / total_steps * 100,
               n_env_step,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
               ram_percent)
              )
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        _ = eval_env.render(mode='human')
        while not (d or (ep_len == max_ep_len)):
            a = R.sess.run(R.model['mu'], feed_dict={R.model['o_ph']: o.reshape(1, -1)})
            o, r, d, _ = eval_env.step(a[0])
            _ = eval_env.render(mode='human')
            ep_ret += r  # compute return
            ep_len += 1
        print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

        # Save
        npz_path = '../data/net/ppo_ant/model.npz'
        save_ppo_model(npz_path, R, VERBOSE=False)

print("Done.")

# Close
eval_env.close()
ray.shutdown()

# Re-init
R.sess.run(tf.global_variables_initializer())

# Restore
npz_path = '../data/net/ppo_ant/model.npz'
restore_ppo_model(npz_path,R,VERBOSE=False)

# Evaluate
eval_env = get_eval_env()
o,d,ep_ret,ep_len = eval_env.reset(),False,0,0
_ = eval_env.render(mode='human')
while not(d or (ep_len == max_ep_len)):
    a = R.sess.run(R.model['mu'],feed_dict={R.model['o_ph']:o.reshape(1,-1)})
    o,r,d,_ = eval_env.step(a[0])
    _ = eval_env.render(mode='human')
    ep_ret += r # compute return
    ep_len += 1
print ("[Evaluate] ep_ret:[%.4f] ep_len:[%d]"%(ep_ret,ep_len))
eval_env.close()