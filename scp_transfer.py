import os
import argparse
import paramiko
from getpass import getpass


def local_recursive_ls(path):
    files = os.listdir(path)
    results = []
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            results += local_recursive_ls(os.path.join(path, file))
        else:
            results.append(os.path.join(path, file))
    return results


def remote_recursive_mkdir(sftp, remote_path):
    p = remote_path.split('/')
    path = ''
    for i in p[1:-1]:
        path = path + '/'
        dirs = sftp.listdir(path)
        if i in dirs:
            path = path + i
        else:
            path = path + i
            sftp.mkdir(path)


def upload_check(file_name, ignore_list):
    for ft in ignore_list:
        if file_name.endswith(ft):
            return False
    return True


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--upload', '-u', type=str)
    args_parser.add_argument('--remote_dir', '-r', type=str)
    args_parser.add_argument('--ignore', '-i', type=str)
    
    args = args_parser.parse_args()
    local_dir = args.upload

    files = local_recursive_ls(local_dir)
    remote_path = args.remote_dir

    if args.ignore is not None:
        ignore = args.ignore.split(',')
        files = [e for e in files if upload_check(e, ignore)]

    host = '210.3.29.222'
    port = 30041
    user_name = 'chaidi'
    password = getpass()

    trans = paramiko.Transport((host, port))
    trans.connect(username=user_name, password=password)
    sftp = paramiko.SFTPClient.from_transport(trans)

    for file in files:
        try:
            sftp.put(localpath=file, remotepath=remote_path + '/' + file)
        except FileNotFoundError:
            remote_recursive_mkdir(sftp, remote_path + '/' + file)
            sftp.put(localpath=file, remotepath=remote_path + '/' + file)
        print('Uploaded', file, 'to', user_name + '@' + host + ':' + remote_path)
    trans.close()
