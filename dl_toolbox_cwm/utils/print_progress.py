"""
Print as progress bar
Programmer: Weiming Chen
Date: 2021.1
"""
import time


class ProgressBar:
    def __init__(self, total):
        self.total = total
        self.width = 50  # the width of progress bar
        self.current = 0
        self.start = time.time()
        self.end = time.time()

    def update_step(self, stride):
        self.current += stride
        self.end = time.time()
        elapsed = self.end - self.start
        stride_per_second = stride / elapsed
        remain_task = self.total - self.current
        if stride_per_second != 0:
            remain_time = remain_task / stride_per_second
        else:
            remain_time = 'null'
        percent = int(100 * self.current / self.total)
        if percent > 100:
            percent = 100
        show_str = ('[%%-%ds]' % self.width) % (int(self.width * percent / 100) * ">")
        print('\rProcess: {} {}/{}, FPS: {:.1f}, elapsed: {:.2f} s, remain: {:.2} s'.
              format(show_str, self.current, self.total, stride_per_second, elapsed, remain_time), end='')

    def start_timing(self):
        self.start = time.time()

    def __del__(self):
        print('')
