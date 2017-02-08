import os
import numpy as np
import matplotlib.pyplot as plt
from parse import *
import progressbar
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pickle
import os.path
import scipy
import scipy.signal

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("root_path", help="path to log")
parser.add_argument("output_prefix", help="output prefix. output images will be <output prefix>_disc_loss.png, <output prefix>_real_loss.png, <output prefix>_fake_loss.png, <output prefix>_gen_loss.png")
#parser.add_argument("log_filenames", nargs='+', 
#                    help="log filenames. multiple filenames are available. if it is the case, all the logs will be drawed in each corresponding plot (disc, real, fake, gen)")
parser.add_argument("-d", "--data", nargs=2, action='append', 
                    help="<label> <log_filename> pairs. multiple data are available. if it is the case, all the logs will be drawed in each corresponding plot (disc, real, fake, gen)")
parser.add_argument("-a", "--avg", help="the logs will be smoothed at every <avg> iterations",
                    type=int,
                    default=1)
#parser.add_argument("-v", "--verbose", help="increase output verbosity",
#                    action="store_true")
args = parser.parse_args()
#if args.verbose:
#    print "verbosity turned on"

def parse_logs(gan_log_path, avg_iter=1):
  #gan_log_path = os.path.join(root_path, gan_log_filename)
  
  ###################################### gan 
  # Open gan_log_path 
  with open(gan_log_path, 'rt') as f:
    lines = f.readlines()
  num_data = len(lines)
  
  # Init necessary variables 
  gan_daxis = np.zeros(num_data)
  gan_gaxis = np.zeros(num_data)
  gan_real_loss = np.zeros(num_data)
  gan_fake_loss = np.zeros(num_data)
  gan_disc_loss = np.zeros(num_data)
  gan_gen_loss  = np.zeros(num_data)
  
  #gan_axis_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  #gan_real_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  #gan_fake_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  #gan_disc_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  #gan_gen_loss_avg  = np.zeros(int(math.floor(num_data/avg_iter)))
  
  # Init bar and do parsing
  print "progress: " 
  bar = progressbar.ProgressBar(maxval=num_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  for i in xrange(num_data):
    tokens = lines[i].split()
    #print tokens 

    gan_disc_loss[i] = float(tokens[9]) 
    gan_real_loss[i] = float(tokens[11])
    gan_fake_loss[i] = float(tokens[13])
    gan_gen_loss[i]  = float(tokens[7])
    #print gan_disc_loss[i]
    #print gan_real_loss[i]
    #print gan_fake_loss[i]
    #print gan_gen_loss[i]
   
    buffers = parse("[{}][{}/{}][{}]", tokens[1])
    epoch = int(buffers[0])+1 
    cur_diter = int(buffers[1])
    max_diter = int(buffers[2])
    giter = int(buffers[3])
    #print buffers
    #print epoch
    #print cur_diter
    #print max_diter

    gan_daxis[i] = (float(epoch)-1) + float(cur_diter)/float(max_diter)
    gan_gaxis[i] = giter 
    #if (i+1) % avg_iter == 0:
    #  j = int(math.floor((i+1) / avg_iter))-1
    #  gan_axis_avg[j] = gan_axis[i]
    #  gan_real_loss_avg[j] = 0
    #  gan_fake_loss_avg[j] = 0
    #  gan_disc_loss_avg[j] = 0
    #  gan_gen_loss_avg[j]  = 0
    #  for ii in xrange(avg_iter):
    #    gan_real_loss_avg[j] = gan_real_loss_avg[j] + gan_real_loss[i-ii] / avg_iter
    #    gan_fake_loss_avg[j] = gan_fake_loss_avg[j] + gan_fake_loss[i-ii] / avg_iter
    #    gan_disc_loss_avg[j] = gan_disc_loss_avg[j] + gan_disc_loss[i-ii] / avg_iter
    #    gan_gen_loss_avg[j]  = gan_gen_loss_avg[j]  + gan_gen_loss[i-ii]  / avg_iter
       
    bar.update(i+1)
  bar.finish()
  return {'daxis':gan_daxis, 'gaxis':gan_gaxis, 'real':gan_real_loss, 'fake':gan_fake_loss , 'disc':gan_disc_loss, 'gen':gan_gen_loss }

def smoothing_logs(logs, avg_iter=1):
  num_data = len(logs['daxis'])

  gan_daxis_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  gan_gaxis_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  gan_real_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  gan_fake_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  gan_disc_loss_avg = np.zeros(int(math.floor(num_data/avg_iter)))
  gan_gen_loss_avg  = np.zeros(int(math.floor(num_data/avg_iter)))

  print "progress: "
  bar = progressbar.ProgressBar(maxval=num_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  for i in xrange(num_data):
    if (i+1) % avg_iter == 0:
      j = int(math.floor((i+1) / avg_iter))-1
      gan_daxis_avg[j] = logs['daxis'][i]
      gan_gaxis_avg[j] = logs['gaxis'][i]
      gan_real_loss_avg[j] = 0
      gan_fake_loss_avg[j] = 0
      gan_disc_loss_avg[j] = 0
      gan_gen_loss_avg[j]  = 0
      for ii in xrange(avg_iter):
        gan_real_loss_avg[j] = gan_real_loss_avg[j] + logs['real'][i-ii] / avg_iter
        gan_fake_loss_avg[j] = gan_fake_loss_avg[j] + logs['fake'][i-ii] / avg_iter
        gan_disc_loss_avg[j] = gan_disc_loss_avg[j] + logs['disc'][i-ii] / avg_iter
        gan_gen_loss_avg[j]  = gan_gen_loss_avg[j]  + logs['gen'][i-ii]  / avg_iter
    bar.update(i+1)
  bar.finish()
  return {'daxis':gan_daxis_avg, 'gaxis':gan_gaxis_avg, 'real':gan_real_loss_avg, 'fake':gan_fake_loss_avg , 'disc':gan_disc_loss_avg, 'gen':gan_gen_loss_avg }



###################################### process data
#print(args)
#print(args.avg)
#print(args.root_path)
#print(args.output_prefix)
#print(args.data)

# init input arguments
num_files = len(args.data)
logs = []
avg_logs = []
avg_iter = args.avg 
root_path = args.root_path
output_prefix = args.output_prefix

# load logs
for i in range(0, num_files):
  gan_log_filename = args.data[i][1] #log_filenames[i]
  gan_log_path = os.path.join(root_path, gan_log_filename)
  gan_log_cache_path = '{}.{}'.format(gan_log_path, 'pkl')
  gan_log_avg_cache_path = '{}.{}.{}'.format(gan_log_path, avg_iter, 'pkl')
  
  if not os.path.exists(gan_log_cache_path):
    print 'parse log (label: {})'.format(args.data[i][0])
    logs.append(parse_logs(gan_log_path))
    pickle.dump(logs[i], open(gan_log_cache_path , "wb"))
  else:
    logs.append(pickle.load(open(gan_log_cache_path, "rb")))
  
  if not os.path.exists(gan_log_avg_cache_path):
    print 'smoothing (label: {})'.format(args.data[i][0])
    avg_logs.append(smoothing_logs(logs[i], avg_iter))
    pickle.dump(avg_logs[i], open(gan_log_avg_cache_path , "wb"))
  else:
    avg_logs.append(pickle.load(open(gan_log_avg_cache_path, "rb")))

###################################### plot real loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  #plt.plot(avg_logs[i]['daxis'], avg_logs[i]['real'], label=args.data[i][0])
  plt.plot(avg_logs[i]['gaxis'], avg_logs[i]['real'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('real loss', fontsize=14, color='black')
plt.title('Real Loss (avg_iter: {})'.format(avg_iter))
plt.savefig('{}_real_loss'.format(output_prefix))
#plt.show()

###################################### plot fake loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  #plt.plot(avg_logs[i]['daxis'], avg_logs[i]['fake'], label=args.data[i][0])
  plt.plot(avg_logs[i]['gaxis'], avg_logs[i]['fake'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('fake loss', fontsize=14, color='black')
plt.title('Fake Loss (avg_iter: {})'.format(avg_iter))
plt.savefig('{}_fake_loss'.format(output_prefix))
#plt.show()

###################################### plot disc loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  #plt.plot(avg_logs[i]['daxis'], avg_logs[i]['disc'], label=args.data[i][0])
  plt.plot(avg_logs[i]['gaxis'], avg_logs[i]['disc'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('disc loss', fontsize=14, color='black')
plt.title('Discriminator Loss (real + fake) (avg_iter: {})'.format(avg_iter))
plt.savefig('{}_disc_loss'.format(output_prefix))
#plt.show()

###################################### plot gen loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  #plt.plot(avg_logs[i]['daxis'], avg_logs[i]['gen'], label=args.data[i][0])
  plt.plot(avg_logs[i]['gaxis'], avg_logs[i]['gen'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('gen loss', fontsize=14, color='black')
plt.title('Generator Loss (avg_iter: {})'.format(avg_iter))
plt.savefig('{}_gen_loss'.format(output_prefix))
#plt.show()

###################################### plot -disc(medfilt) loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  med_filtered_loss = scipy.signal.medfilt(-logs[i]['disc'], 101)
  #plt.plot(logs[i]['daxis'], med_filtered_loss, label=args.data[i][0])
  plt.plot(logs[i]['gaxis'], med_filtered_loss, label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('-disc loss', fontsize=14, color='black')
plt.title('Negative Discriminator Loss (median filtered, size: {})'.format(101))
plt.savefig('{}_mdisc_medfilt_loss'.format(output_prefix))
#plt.show()

###################################### plot -disc loss
fig, ax = plt.subplots()
#plt.xlim(0, max(gan_axis[-1], egan_const_axis[-1], egan_ent_vi_axis[-1]))
#plt.ylim(0, 2.5)

for i in range(0, num_files):
  plt.plot(logs[i]['gaxis'], -logs[i]['disc'], label=args.data[i][0])

plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.minorticks_on()
#plt.yticks(np.arange(0, 5, 10))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#plt.xlabel('epochs', fontsize=14, color='black')
plt.xlabel('generator iterations', fontsize=14, color='black')
plt.ylabel('-disc loss', fontsize=14, color='black')
plt.title('Negative Discriminator Loss'.format(avg_iter))
plt.savefig('{}_mdisc_loss'.format(output_prefix))
#plt.show()


print 'Done.'
