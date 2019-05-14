def GetConfig(option):
    CONFIG = {}
    if option=='16000oneliners':
        CONFIG['ROOT_PATH'] = './data/16000 oneliners/'
        CONFIG['RAW_DATA'] = './data/16000 oneliners/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/16000 oneliners/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/16000 oneliners/tensor_embedding.pkl'
        CONFIG['WV_PATH'] = './data/16000 oneliners'
        CONFIG['CNN_FEATURES'] = './data/16000 oneliners/cnn_features.pkl'

    if option=='Pun':
        CONFIG['ROOT_PATH'] = './data/Pun/'
        CONFIG['RAW_DATA'] = './data/Pun/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/Pun/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/Pun/tensor_embedding.pkl'
        CONFIG['WV_PATH'] = './data/Pun'
        CONFIG['CNN_FEATURES'] = './data/Pun/cnn_features.pkl'
    return CONFIG