# Notes

seeding 
```
torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
```

make_vec_envs func uses multi-process already ?
vector envs ? just wrap input-output with tensor ?

policy: input shape, output shape, other info in kwargs 

agent is trainer, take in policy, training hyperparams
has .update function 

RolloutStorage, take in obs shape, action shape 

episode_rewards as a deque 


what is `num_steps` ? length of each episode ? yes !
`num_env_steps` is total steps to run, also being reported in figure in x axis 
```
num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
```


reward seems to be recorded only when episode ends 
reward queue has only final rewards from multi-process 
```
for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
```


masks to mask rewards based on done flag in episodes
needed for multi-process for envs ? 
mask is a input to action sample, why ? 
bad_masks? what is bad transotion ?


what is `next_value` ? used for Q value updates ? needed becoz last step (done True) also need updates ?
```
with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
```

save and log and eval intervals on number of updates 
why save this `getattr(utils.get_vec_normalize(envs), 'ob_rms', None)`
what is `ob_rms` ?


total_num_steps = (j + 1) * args.num_processes * args.num_steps
j number of updates so far 
so num_steps is steps interacting with env 

### main framework 

- get args 
- set seed, cuda, thread, device 
- create, clean directories 
- make vec envs 
- make policy 
- make agent/trainer 
- make buffer/rollout storage, reward queue
```
num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
```
- env reset
- for in range num_updates 
    - step scheduler 
    - for in range num_steps/episode_length 
        - sample actions 
        - step envs 
        - append reward 
        - make masks, save tuple to buffer 
    - get next final step Q if AC 
    - compute returns 
    - agent update, get losses and step optimizer 
    - update buffer 
    - save if needed 
    - log if needed 
    - eval if needed 


### eval framework 

- make vec envs and set initialization 
- reset env 
- while not enough episodes 
    - sample action
    - step env 
    - record reward 
- env close 
- return mean reward


### key arguments

- experiment params: env-name, seed, cuda-deterministic, num-mini-batch, log-interval, eval-interval, save-interval, log-dir, save-dir 
- model params: algo (a2c, ppo), recurrent-policy
- training params: lr, gamma, gae-lambda, entropy-coef, value-loss-coef, max-grad-norm, num-steps (forward steps in A2C), num-env-steps (env steps to train, e.g. 10e6), ppo-epoch, use-proper-time-limits

```python
args.cuda = not args.no_cuda and torch.cuda.is_available()
```

### buffer 

RolloutStorage
args: num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size

fields  
- obs, (#steps+1, #ps, *obs_shape)
- recurrent_hidden_states, (#steps+1, #ps, h_dim) 

- actions, (#steps, #ps, action_shape)
- action_log_probs, (#steps, #ps, 1)
- value_preds, (#steps+1, #ps, 1), since require Q trained on last step too 

- rewards, (#steps, #ps, 1)
- masks, (#steps+1, #ps, 1)
- bad_masks, (#steps+1, #ps, 1)

- returns, (#steps+1, #ps, 1)

- step, 0

methods 
- to: fields to device 
- insert:
    - obs, recurrent_hidden_states, masks, bad_masks add at step+1 (since there's init reset)
    - actions, action_log_probs, value_preds, rewards add at step 
    - step = (step + 1) % num_steps
- after_update 
    - copy last obs, recurrent_hidden_state, masks and bad_masks to 0 step
    - why ???
- compute_returns 
    - args: next_value, gamma, use_gae, gae_lambda, use_proper_time_limits
    - next_value needed to do return discount from backward, could be 0 for last step since no further reward
    ```python 
    returns[-1] = next_value
    ```
    - recursively calculate returns
    ```python
    returns[step] = returns[step + 1] * gamma * masks[step + 1] + rewards[step]
    ```
    - for gae, has another innner recursive counter to accumulate delta (reference: https://arxiv.org/pdf/1506.02438.pdf eqn 11-16)
    - for proper_time_limit, truncate rewards with bad_masks, use value_preds to replace truncated returns 
    ```python
    returns[step] = (returns[step + 1] * gamma * masks[step + 1] 
    + rewards[step]) * bad_masks[step + 1]
    + (1 - bad_masks[step + 1]) * value_preds[step]
    ```
- feed_forward_generator
    - used to turn rollouts into batch of inputs for markov policy
    - batch_size = num_processes * num_steps 
    - return batch of (obs, recurrent_hidden_states, actions, value_preds, returns, masks, action_log_probs)
- recurrent_generator
    - for recurrent policy



### distributions 

Categorical 
DiagGaussian 
Bernoulli 

each has an an interface that overwrites 
- sample
- log_probs 
- mode 

each is now a trainable nn.Module, with Linear layer to output logits, mean



### model 

Policy 
- args: obs_shape, action_space, base, base_kwargs 
- dim(obs_space), 3 --> CNNBase, 1 --> MLPBase 
- there's also a `self.dist`, Discrete --> Categorical, Box --> DiagGaussian, MultiBinary --> Bernoulli 
- forward, NotImplemented ??? 
- act
    - args: inputs, rnn_hxs, masks, deterministic
    - if deterministic --> `dist.mode`, else --> `dist.sample`
    - return: value, action, action_log_probs, rnn_hxs 
- get_value 
- evaluate_actions
    - return: value, action_log_probs, dist_entropy, rnn_hxs 


Base 
- NNBase 
    - is_recurrent --> property 
- CNNBase 
    - `self.main` is convolution base, `self.critic_linear` is Linear 
    - forward
        - inputs --> `main` --> `gru` (optional) --> `critic_linear`
        - return: value, x, rnn_hxs
- MLPBase 
    - `self.actor` is Sequential Linear, `self.critic` is Sequential Linear, `self.critic_linear` is Linear 
    - forward
        - inputs --> `gru` (optional) --> `critic` & `actor`
        - returns: value, hidden_actor, rnn_hxs 


### envs 

reference to openai baseline (wrapper, vec_env, atari modules), https://github.com/openai/baselines
- `vec_env`, base classes for vectorized environments and wrappers
    - `VecEnv`, sub-class to *ABC*, abstract asynchronous vectorized environment
        - args: num_envs, observation_space, action_space
        - abstract methods: reset, step_async, step_wait (returns obs, rews, dones, infos; each is arrays)
        - methods: step, close, render
    - `VecEnvWrapper`, sub-class to *VecEnv*, env wrapper that applies to entire batch of envs at once 
- `DummyVecEnv`, uns multiple environments sequentially
- `ShmemVecEnv`, optimized version of SubprocVecEnv that uses shared variables to communication observations
- `atari_wrappers`
- `VecNormalize`, wrapper that normalizes observations and returns 


make_vec_envs
- args: env_name, seed, num_processes, gamma, log_dir, device, allow_early_resets, num_frame_stack 
- `num_processes` is like batch size; 1 ps --> DummyVecEnv, >1 ps --> ShmemVecEnv
- [env] * num_processes --> ShmemVecEnv --> VecNormalize --> VecPyTorch --> VecPyTorchFrameStack  


make_env
- args: env_id, seed, rank, log_dir, allow_early_resets 
- return a func that returns env (becoz multiprocessing)


### algo / agent/ trainer 

#### ppo 

PPO 
- args: policy, **loss params/training params
- contain optimizer 
- update
    - args: rollouts 
    - calculate and normalize advantages (returns - value_preds in rollouts)
    ```python 
    # last step is useless, since reward is 0 
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    ```
    - initialize epoch value loss, action_loss, entropy 
    - for in range ppo_epochs
        - get data_generator from rollouts, input (advantages, num_minin_batch)
        - for in data_generator / iterate mini-batches 
            - run policy network, get log_probs, critic values, entropy 
            ```python
            values, action_log_probs, dist_entropy, _ = policy.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,actions_batch)
            ```
            - calculate action loss (ppo), value loss 
            ```python
            # action loss 
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            # value loss 
            value_loss = 0.5 * (return_batch - values).pow(2).mean()
            ```
            - step optimizer
            - append losses to epoch losses
    - get average epoch losses 
    ```python
    num_updates = ppo_epoch * num_mini_batch
    epoch_loss /= num_updates
    ```

#### a2c_acktr 

A2C_ACKTR
- args: policy, **loss params/training params, acktr flag 
- acktr flag, true --> KFACOptimizer; false --> RMSprop
- update
    - args: rollouts 
    - get values, action_log_probs, dist_entropy with `policy.evaluate_actions`
    - get advantags, losses 
    ```python 
    advantages = rollouts.returns[:-1] - values
    value_loss = advantages.pow(2).mean()
    action_loss = -(advantages.deteach() * action_log_probs).mean
    ```
    - (optinal), use acktr with natural gradient (fisher_loss)
    ```python
    fisher_loss = pg_fisher_loss + vf_fisher_loss
    optimizer.acc_stats = True
    fisher_loss.backward(retain_graph=True)
    optimizer.acc_stats = False
    ```
    - step optimizer 
    - return losses

KFACOptimizer

#### gail 

Discriminator
- MLP discriminator with tanh activations, input is `[state, action]`
- contain optimizer 
- compute_grad_pen
    - args: expert_state, expert action, policy state, policy_action, lambda
    - reference: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
    - gradient penalty for discriminator (*WGAN* technique)
    - construct mixed data by concatenating expert data and policy data 
    - use `autograd.grad` to get gradient penalty 
    ```python
    # forward 
    disc = discriminator(mixup_data)
    ones = torch.ones(disc.size()).to(disc.device)
    # grad operation in computation graph, need "retain_graph"
    grad = autograd.grad(
        outputs=disc,
        inputs=mixup_data,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # distance of grad norms to 1 
    grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
    ```
- update
    - args: expert_loader, rollouts, obsfilt
    - get export_loader, data_generator
    - for in export_loader & data_generator
        - run discriminator forward for export and generated data 
        - get gail loss and grad penalty 
        ```python
        # real --> 1
        expert_loss = F.binary_cross_entropy_with_logits(expert_d,
            torch.ones(expert_d.size()).to(self.device))
        # fake --> 0, can consider using soft labels for both 
        policy_loss = F.binary_cross_entropy_with_logits(policy_d,
            torch.zeros(policy_d.size()).to(self.device))
        gail_loss = expert_loss + policy_loss
        # grad penalty 
        grad_pen = compute_grad_pen(expert_state, expert_action, policy_state, policy_action)
        loss += (gail_loss + grad_pen).item()
        ```
        - step optimizer 
- predict_reward
    - args: state, action, gamma, masks, update_rms 
    ```python
    d = discriminator(torch.cat([state, action], dim=1))
    s = torch.sigmoid(d)
    reward = s.log() - (1 - s).log()
    ```

ExpertDataset
- args: file_name, num_trajectories, subsample_frequency 
- `__getitem__` returns `[state, action]` at particular traj and time step