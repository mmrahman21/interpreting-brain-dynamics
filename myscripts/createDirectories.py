import os
import datetime

def create_Directories(args, model):

    if model == "sequence":
        print('Creating Essential Directories for sequence based models...')
        currentDT = datetime.datetime.now()
        d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
        d1 = d1[:12] + '-' + d1[13:15]
        dir = 'run-' + d1
        dir = dir + '-' + str(args.script_ID)

        wdb = 'wandb'
        wpath = os.path.join('../', wdb)
        sbpath = os.path.join(wpath, 'Sequence_Based_Models')
        path = os.path.join(sbpath, dir)
        run_dir = path

        args.path = path
        print('This is the path:', args.path)
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Fig')
        args.fig_path = path
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'TrainAccuracy')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'TrainLoss')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'ValAccuracy')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'ValLoss')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Metrics')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Predictions')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Saliency')
        if not os.path.exists(path):
            os.mkdir(path)

    elif model == 'window':
        print('Creating Essential Directories for window based models...')
        currentDT = datetime.datetime.now()
        d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
        d1 = d1[:12] + '-' + d1[13:15]
        dir = 'run-' + d1
        dir = dir + '-' + str(args.script_ID)

        wdb = 'wandb'
        wpath = os.path.join('../', wdb)
        wbpath = os.path.join(wpath, 'Window_Based_Models')
        path = os.path.join(wbpath, dir)
        run_dir = path

        args.path = path
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Fig')
        args.fig_path = path
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'TrainAccuracy')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'TrainLoss')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'ValAccuracy')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'ValLoss')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Metrics')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Predictions')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Saliency')
        if not os.path.exists(path):
            os.mkdir(path)

    else:
        print('Creating Essential Directories for standard ML models...')
        currentDT = datetime.datetime.now()
        d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
        d1 = d1[:12] + '-' + d1[13:15]
        dir = 'run-' + d1
        dir = dir + '-' + str(args.script_ID)

        wdb = 'wandb'
        wpath = os.path.join('../', wdb)
        wbpath = os.path.join(wpath, 'Classic_ML_Models')
        path = os.path.join(wbpath, dir)
        run_dir = path

        args.path = path
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Metrics')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(run_dir, 'Predictions')
        if not os.path.exists(path):
            os.mkdir(path)

    return run_dir



