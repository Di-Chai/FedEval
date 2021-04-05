import os

os.chdir('./')
os.system('scp -i D:\\id_rsa -r configs trials ubuntu@10.173.1.22:/ldisk/chaidi/FedEval')

os.chdir('FedEval')
os.system('scp -i D:\\id_rsa -r attack dataset model role strategy utils *.py '
          'ubuntu@10.173.1.22:/ldisk/chaidi/FedEval/FedEval')

"""
shake SGD
[1.6441494226455688, 0.5213959217071533]
"""