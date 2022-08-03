import os

with open('Appendix_1.rst', 'r') as f:
    appendix = f.readlines()

with open('FC_Attack.html', 'r') as f:
    fc_attack = ['    ' + e for e in f.readlines()]

with open('Appendix.rst', 'w') as f:
    f.writelines(appendix + fc_attack + ['\n----------------------\n'])

os.chdir('../')

os.system('sphinx-build sphinx docs')