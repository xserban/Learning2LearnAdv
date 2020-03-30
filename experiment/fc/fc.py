"""This class runs all experiments for fully connected networks
"""
from __future__ import absolute_import
from __future__ import print_function

import json
from l2l import flags, train_fc


def run_exp(data):
  """Runs one experiment"""
  try:
    flags.set_flags(data)
  except Exception as e:
    raise e
  else:
    train_fc.main(None)


def main():
  """Loads and runs all experiments"""
  try:
    with open("experiment/fc/design.json") as data:
      exp = json.load(data)
  except Exception as e:
    print('[ERROR] Could not open experimental design file. \t ', e)
  else:
    exp = exp["fc"]
    for index, value in enumerate(exp):
      try:
        print("[INFO] Running experiment {}".format(index))
        run_exp(value["flags"])
      except Exception as e:
        print("[ERROR] Could not run configuration {}".format(index), e)


if __name__ == '__main__':
  main()
