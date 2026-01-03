# fmt: off
# [start-config-dict-jax]
S2AC_DEFAULT_CONFIG = {
    "particles": 16,
    "svgd_steps": 5,
    "svgd_step_size": 0.05,
    "kernel_sigma": None,
    "kernel_sigma_adaptive": True,
    "stop_grad_svgd_score": True,

    "use_soft_q_backup": True,
    
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "critic_target_update_interval": 1,
    "actor_update_frequency": 2,

    "entropy_floor": 0.0,
    "entropy_floor_coef": 0.0,

    "auto_entropy_tuning": True,
    "alpha": 0.2,
    "target_entropy": None,

    "reward_scale": 1.0,
    "random_timesteps": 1000,
    "learning_starts": 1000,
    "grad_norm_clip": 10.0,
    "weight_decay": 1e-4,

    "twin_critics": True,

    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": 250,
        "checkpoint_interval": 'auto',
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }
}
# [end-config-dict-jax]
# fmt: on
