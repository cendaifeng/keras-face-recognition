#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import os
import time

print("\33[42;1mdetected face! unlocking...\033[0m")

with os.popen('python '+os.path.join(os.path.split(os.path.realpath('__file__'))[0], '。。/unlock.py')) as f:
    print(f.read())
time.sleep(20)  # 20s 内不再次开锁
