import os
from multiprocessing import cpu_count
from tf_wrapper.utils import multiple_process


def task(share_queue, locker, data, parameters):

    print('Child process %s with pid %s' % (parameters[0], os.getpid()))

    print(data)

    for record in data:
        os.system("python {5} --training_samples {1} --fake_samples {1} "
                  "--client_batch_size {3} "
                  "--attack_epoch 20000 --attack_local_epoch 1 --client_local_epoch {2} "
                  "--server_lr 0.1 --server_plot_epochs 500 "
                  "--dataset mnist --fname {0}-I{1}-E{2}-{4}".format(*record))

    locker.acquire()
    share_queue.put(None)
    locker.release()


if __name__ == "__main__":

    # number of process
    n_job = 8  # cpu_count()

    #######################################################################################
    # Varying # of images, DLG, FedSGD, LeNet
    mode = 'DLG-LeNet-FedSGD'
    scripts = 'GradientAttack_DLG_LeNet.py'
    num_images = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_epochs = [1]
    batch_size = max(num_images)
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )

    #######################################################################################
    # Varying # of images, DLG, FedAvg, LeNet
    mode = 'DLG-LeNet-FedAvg'
    scripts = 'GradientAttack_DLG_LeNet.py'
    num_images = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_epochs = [20]
    batch_size = 1
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )

    #######################################################################################
    # Varying # of epochs, DLG, FedAvg, LeNet
    mode = 'DLG-LeNet-FedAvg'
    scripts = 'GradientAttack_DLG_LeNet.py'
    num_images = [2, 6]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_size = 1
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )

    #######################################################################################
    # Varying # of images, FedSGD, MLP
    mode = 'DLG-MLP-FedSGD'
    scripts = 'GradientAttack_DLG_MLP.py'
    num_images = [1, 5, 10, 15, 0, 25, 30]
    num_epochs = [1]
    batch_size = max(num_images)
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )

    #######################################################################################
    # Varying # of images, FedAvg, MLP
    mode = 'DLG-MLP-FedAvg'
    scripts = 'GradientAttack_DLG_MLP.py'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [20]
    batch_size = 1
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )

    #######################################################################################
    # Varying # of epochs, FedAvg, MLP
    mode = 'DLG-MLP-FedAvg'
    scripts = 'GradientAttack_DLG_MLP.py'
    num_images = [10, 20, 30]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_size = 1
    repeat_times = 50

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, batch_size, r, scripts])

    multiple_process(
        distribute_list=data,
        partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[]
    )
