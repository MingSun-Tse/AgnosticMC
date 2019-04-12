import glob
import os
import shutil
import time
import sys
pjoin = os.path.join

class LogPrint():
  def __init__(self, file):
    self.file = file
  def __call__(self, some_str):
    print(time.strftime("[%Y/%m/%d-%H:%M] ") + str(some_str), file=self.file, flush=True)
  
def check_path(x):
  if x:
    complete_path = glob.glob(x)
    assert(len(complete_path) == 1)
    x = complete_path[0]
  return x

def get_previous_step(e2, resume):
  previous_epoch = previous_step = 0
  if e2 and resume:
    for clip in os.path.basename(e2).split("_"):
      if clip[0] == "E" and "S" in clip:
        num1 = clip.split("E")[1].split("S")[0]
        num2 = clip.split("S")[1]
        if num1.isdigit() and num2.isdigit():
          previous_epoch = int(num1)
          previous_step  = int(num2)
  return previous_epoch, previous_step
  
def set_up_dir(project_name, resume, CodeID):
  TimeID = time.strftime("%Y%m%d-%H%M")
  ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
  project_path = pjoin("../Experiments", ExpID + "_" + project_name)
  rec_img_path = pjoin(project_path, "reconstructed_images")
  weights_path = pjoin(project_path, "weights") # to save torch model
  if not os.path.exists(project_path):
    os.makedirs(project_path)
  else:
    if not resume:
      shutil.rmtree(project_path)
      os.makedirs(project_path)
  if not os.path.exists(rec_img_path):
    os.makedirs(rec_img_path)
  if not os.path.exists(weights_path):
    os.makedirs(weights_path)
  log_path = pjoin(weights_path, "log_" + ExpID + ".txt")
  log = open(log_path, "w+") if CodeID else sys.stdout # Given CodeID, it means this is a formal experiment, i.e., not debugging
  return TimeID, ExpID, rec_img_path, weights_path, log