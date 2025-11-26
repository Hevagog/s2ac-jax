from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG
import jax
import gymnax
from skrl.envs.wrappers.jax import wrap_env
import brax.envs
from skrl.memories.jax import RandomMemory
from skrl.trainers.jax import SequentialTrainer

from agent.sac import Target_Critic_MLP, Policy_MLP, Critic_MLP

device = jax.devices("gpu")[0]
env = brax.envs.create("reacher", batch_size=4092, backend="mjx")
env = wrap_env(env)

memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)

# # instantiate the agent's models
models = {}
models["policy"] = Policy_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["policy"].init_state_dict("policy")
models["critic_1"] = Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["critic_1"].init_state_dict("critic_1")
models["critic_2"] = Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["critic_2"].init_state_dict("critic_2")
models["target_critic_1"] = Target_Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["target_critic_1"].init_state_dict("target_critic_1")
models["target_critic_2"] = Target_Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["target_critic_2"].init_state_dict("target_critic_2")

# adjust some configuration if necessary
cfg_agent = SAC_DEFAULT_CONFIG.copy()

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SAC(
    models=models,
    memory=memory,  # only required during training
    cfg=cfg_agent,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
)

cfg = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)
trainer.train()

trainer.eval()
