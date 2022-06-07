import os

log_dir = 'log'
log_files = [os.path.join(log_dir, e) for e in os.listdir(log_dir) if 'c1' in e]

for log_file in log_files:
    num_samples, bandwidth, latency = log_file.split('_')[:3]
    
    with open(log_file, 'r') as f:
        data = f.readlines()
    
    samples = [e for e in data if e.startswith('Synthetic')][0].split()
    cost = [e for e in data if e.startswith('Training Cost')][0].split()[3]

    print(', '.join([str(e) for e in [samples[3], samples[1], bandwidth[:-4], latency[:-2], cost]]))
