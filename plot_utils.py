import inspect 
import matplotlib.pyplot as plt

from models import *
from utils import *


def plot_train_val_acc(save_name, ret_dict, orig_hp, new_hp, save, gru=False):

    fig, ax = plt.subplots(1, 1)
    xx = np.linspace(1, orig_hp.N_EPOCHS, orig_hp.N_EPOCHS)

    ax.plot(xx, ret_dict['train_accs'], label='Train')
    ax.plot(xx, ret_dict['valid_accs'], 
            label='Validation; max accuracy=%.4f' % (np.max(ret_dict['valid_accs'])))
    ax.set_ylim([0.45, 1.05])

    diff_param = diff_hparams(orig_hp=orig_hp, new_hp=new_hp)
    
    lstm_or_gru = {False: 'LSTM', True: 'GRU'}
    
    if len(diff_param) != 0:
        title_ = lstm_or_gru[gru] + ';' + diff_param
    else:
        title_ = lstm_or_gru[gru]
    title_ += ';test_acc:%.4f' % ret_dict['test_acc']
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title(title_)
    fig.tight_layout()
    if save == 'y':
        if len(diff_param) != 0:
            plt.savefig('%s_%s.pdf' % (save_name, diff_param.strip()), dpi=500, bbox_inches='tight')
        else:
            plt.savefig('%s_default_hp.pdf' % (save_name), dpi=500, bbox_inches='tight')
    return 