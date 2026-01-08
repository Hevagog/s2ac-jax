# fmt: off
# [start-config-dict-jax]
S2AC_DEFAULT_CONFIG = {
    "particles": 32,            
    "svgd_steps": 10,            
    "svgd_step_size": 0.1,       
    "kernel_sigma": None,       
    "kernel_sigma_adaptive": True,
    "kernel_sigma_max": 5.0,    
    "stop_grad_svgd_score": True,  
    "max_phi_norm": 10.0,  
    "u_clip_bound": 6.0,        

    "use_soft_q_backup": True,  

    "gradient_steps": 2,        

    "batch_size": 256,          
    "discount": 0.99,           
    "tau": 0.005,               
    "actor_lr": 3e-4,           
    "critic_lr": 3e-4,         
    "critic_target_update_interval": 1,
    "actor_update_frequency": 1, 

    "entropy_floor": 0.0,
    "entropy_floor_coef": 0.0,

    "auto_entropy_tuning": True,
    "alpha": 0.2,               
    "target_entropy": None,    
    "log_alpha_bounds": (-5.0, 2.0),  

    "reward_scale": 1.0,
    "random_timesteps": 1,   
    "learning_starts": 1,
    "grad_norm_clip": 1.0,    
    "weight_decay": 1e-4,

    "twin_critics": True,

    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": 100,
        "checkpoint_interval": 'auto',
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }
}


S2AC_MULTIGOAL_CONFIG = {
    "particles": 16,             
    "svgd_steps": 5,             
    "svgd_step_size": 0.1,       
    "kernel_sigma": None,       
    "kernel_sigma_adaptive": True,
    "kernel_sigma_max": 5.0,    
    "stop_grad_svgd_score": True,  
    "max_phi_norm": 10.0,  
    "u_clip_bound": 6.0,        

    "use_soft_q_backup": True,  

    "gradient_steps": 1,        

    "batch_size": 256,          
    "discount": 0.99,           
    "tau": 0.005,               
    "actor_lr": 3e-4,           
    "critic_lr": 3e-4,          
    "critic_target_update_interval": 1,
    "actor_update_frequency": 1, 

    "entropy_floor": 0.0,
    "entropy_floor_coef": 0.0,

    "auto_entropy_tuning": True,
    "alpha": 0.2,               
    "target_entropy": -2.0,    
    "log_alpha_bounds": (-5.0, 2.0),

    "reward_scale": 1.0,
    "random_timesteps": 1,   
    "learning_starts": 1,
    "grad_norm_clip": 1.0,    
    "weight_decay": 1e-4,

    "twin_critics": True,

    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": 100,
        "checkpoint_interval": 'auto',
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }
}
# [end-config-dict-jax]
# fmt: on
