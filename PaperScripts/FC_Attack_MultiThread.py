import os
from tf_wrapper.utils import multiple_process

instruction = 'python GradientAttack_FC_MLP.py --training_samples {} --client_batch_size {} ' \
              '--client_local_epoch {} --fname {}'


def task(share_queue, locker, data, parameters):

    print('Child process %s with pid %s' % (parameters[0], os.getpid()))
    print(data)
    for record in data:
        os.system(instruction.format(record[1], record[4], record[2],
                                     '{}-I{}-E{}-{}'.format(record[0], record[1], record[2], record[3])))
    locker.acquire()
    share_queue.put(None)
    locker.release()


if __name__ == "__main__":

    # number of process
    n_job = 10  # cpu_count()

    ############################################################################
    # Varying # of images, FC, FedSGD, MLP
    mode = 'FC-MLP-FedSGD'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [1]
    repeat_times = 10

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, r, ni])

    multiple_process(distribute_list=data,
                     partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
                     task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[])

    ############################################################################
    # Varying # of images, FC, FedAvg, MLP
    mode = 'FC-MLP-FedAvg'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [20]
    repeat_times = 10

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, r, ni])

    multiple_process(distribute_list=data,
                     partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
                     task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[])

    ############################################################################
    # Varying # of epochs, FC, FedAvg, MLP
    mode = 'FC-MLP-FedAvg'
    num_images = [10, 20, 30]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    repeat_times = 10

    data = []
    for ni in num_images:
        for ne in num_epochs:
            for r in range(repeat_times):
                data.append([mode, ni, ne, r, ni])

    multiple_process(distribute_list=data,
                     partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
                     task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x, parameters=[])
