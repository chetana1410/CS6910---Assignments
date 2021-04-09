sweep_config = {
   #'program': train(),
    'method': 'bayes',         
    'metric': {
        'name': 'val_accuracy',     
        'goal': 'maximize'      
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3,1e-4]
        },
        'activ': {
            'values': ['relu']
        },
        'bn': {
            'values': [0,1]
        },
        'num_filters': {
            'values': [32, 64, 128]
        },
        'filter_org': {
            'values': ['double', 'same']
        },
        'epochs': {
            'min': 5,
	    'max': 30
        },
        'dropout': {
            'values': [0, 0.2, 0.3]
        },
        'data_aug': {
            'values': [0,1]
        },
        'optimizer' : {
            'values': ['Adam','rmsprop','sgd']
        },

        'batch_size': {
            'values': [32, 64]
        },
        'ker_size_input':{
            'values':[3, 5, 7]
        },
        'bn_vs_dp':{
            'values':['bn', 'dp']
        }
        'max_vs_avg': {
            'values':['max', 'avg']
        }

        }
    }


sweep_id = wandb.sweep(sweep_config,project='asgn2 q1 1')

wandb.agent(sweep_id, function=train)