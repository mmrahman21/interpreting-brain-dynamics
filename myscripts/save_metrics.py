import os
import pandas as pd
import numpy as np

def save_metrics(exp_type, accuracy, loss, auc, req_epoch, run_dir):
    path = os.path.join(run_dir, 'Metrics')
    filename = os.path.join(path, 'accuracy_' + exp_type + '.csv')
    df = pd.DataFrame(accuracy)
    df.to_csv(filename)
    print(df.mean(axis=1))

    filename = os.path.join(path, 'loss_' + exp_type + '.csv')
    df = pd.DataFrame(loss)
    df.to_csv(filename)
    print(df.mean(axis=1))

    filename = os.path.join(path, 'auc_' + exp_type + '.csv')
    df = pd.DataFrame(auc)
    df.to_csv(filename)
    print(df.mean(axis=1))

    filename = os.path.join(path, 'Epochs_' + exp_type + '.csv')
    df = pd.DataFrame(req_epoch)
    df.to_csv(filename)
    print(df.mean(axis=1))
    print('All metrics saved correctly..')

def save_metrics_other_algo(accuracy, auc, run_dir, name):
    path = os.path.join(run_dir, 'Metrics')
    filename = os.path.join(path, name+'_accuracy' + '.csv')
    df = pd.DataFrame(accuracy)
    df.to_csv(filename)
    print(' Saved...', filename)
    print(df.mean(axis=1))

    filename = os.path.join(path, name+'_auc'+ '.csv')
    df = pd.DataFrame(auc)
    df.to_csv(filename)
    print(' Saved...', filename)
    avg = pd.Series(df.mean(axis=1))
    print(avg)
    filename = os.path.join(path, name + '_mean_auc' + '.csv')
    avg.to_csv(filename, header=True)
    print('All metrics saved correctly..')


# Save cross-validation results
def save_metrics_cv(bestFoldAuc, avgFoldAuc, avgFoldAcc, run_dir, name):
    path = os.path.join(run_dir, 'Metrics')
    filename = os.path.join(path, name+'_bestFoldAUC' + '.csv')
    df = pd.DataFrame(bestFoldAuc)
    df.to_csv(filename)
    avg = df.mean(axis=0)
    print('{} : Mean over best AUCs - {}'.format(name, avg))

    filename = os.path.join(path, name+'_avgFoldAUC'+ '.csv')
    df = pd.DataFrame(avgFoldAuc)
    df.to_csv(filename)
    avg = df.mean(axis=0)
    print('{} : Mean over mean AUCs - {}'.format(name, avg))

    filename = os.path.join(path, name + '_avgFoldACC' + '.csv')
    df = pd.DataFrame(avgFoldAcc)
    df.to_csv(filename)
    avg = df.mean(axis=0)
    print('{} : Mean over mean ACCs - {}'.format(name, avg))
    print('All metrics saved correctly..')