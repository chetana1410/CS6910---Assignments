sweep_config = {
   #'program': train(),
    'method': 'bayes',         
    'metric': {
        'name': 'val_accuracy',     
        'goal': 'maximize'      
    },
    'parameters': {
        'learning_rate': {
            'values': [0, 1e-3, 5*1e-4, 1e-4]   # 0 implies default learning rate set by model
        },
        'freeze': {
            'values': [-1, 5, 10]
        },
        'epochs': {
            'min': 5,
            'max': 20
        },
        'data_aug': {
            'values': [0,1]
        },
        'optimizer' : {
            'values': ['Adam', 'rmsprop', 'SGD']
        },
        'tr_model' : {
            'values': ['VGG16', 'InceptionV3', 'ResNet50', 'Xception', 'InceptionResNetV2']
        },
        'batch_size': {
            'values': [32, 64]
        }
        }
    }
