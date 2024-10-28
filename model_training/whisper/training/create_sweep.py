import wandb


sweep_config = {
    'method': 'random',
    'program': 'training.py',
    'name': 'sweep-whisper-finetune',
    'project': 'language-x-change',

}

metric = {
    'name': 'wer',
    'goal': 'minimize'
}

parameters_dict = {
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.001
        },
    'num_train_epochs': {
        'value': 1
    },
    'eval_steps': {
      'min': 10,
      'max':1000
    },
    'max_steps': {
      'min': 10,
      'max':5000
    },
    'gradient_accumulation_steps':  {
      'min': 1,
      'max':10
    },
    'per_device_train_batch_size':{
      'min': 1,
      'max': 16
    },
    'per_device_eval_batch_size':{
      'min': 1,
      'max': 16
    },
    "generation_max_length":{
      'min': 20,
      'max': 100
    }
}

parameters_dict.update({
    
})

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="language-x-change")