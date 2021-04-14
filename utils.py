import pickle
from RNN import model as rnn


loaded_model_weights = None

def dump_model(model_weights=None):
    '''
    save the trained model (morse, simplenet, rnn) to a binary file
    '''
    if model_weights is None:
        model_weights = wrap_model()
    model_dir = gv.model_save_path
    with open("model_weights.pkl", 'wb') as handle:
        pickle.dump(model_weights, handle, pickle.HIGHEST_PROTOCOL)

def load_model():
    '''
    load a trained model from a binary file
    '''
    with open("model_weights.pkl", 'rb') as handle:
        loaded_model_weights = pickle.load(handle)

def wrap_model():
    '''
    grap different sub models (morse, simplenet, rnn) together 
    '''
    from target import Target as tg
    from globals import GlobalVariable as gv  
    from RNN import model as rnn  
    model_weights = {
        "morse": {},
        "simplenet": {},
        "rnn": rnn
    }
    model_weights["morse"]["a_b"] = tg.a_b
    model_weights["morse"]["a_e"] = tg.a_e
    model_weights["simplenet"]["1"] = tg.benign_thresh_model
    model_weights["simplenet"]["2"] = tg.suspect_env_model
    return model_weights

def early_stop_triggered(loss1, loss2, threshold):
    return ((loss2 - loss1) / loss1 > threshold):
