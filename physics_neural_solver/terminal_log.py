"""
terminal_log.py  -  readable training progress in the terminal (and in train.log when run with nohup)

Every message is flushed at once, so `tail -f train.log` shows it immediately.
Training has no epochs (there is no dataset: every step uses fresh states from the training pool), so progress is
counted in steps, with % done, the time per step, the time so far and the time left.

    [14:02:11] step    500   1.2%  calm  | total 1.83e-03 | mom 2.1e-04  surf 3.0e-05 ... | lr 5.0e-04 | 0.031 s | 0:00:16 | 0:20:30 left
"""

import time                                                         # clock
from datetime import timedelta                                      # readable durations

COLUMNS = [                                                         # (loss name in the log, short column title)
    ("momentum (eq. 2)", "mom(2)"),
    ("level set (eq. 5)", "surf(5)"),
    ("eikonal (eq. 6)", "eik(6)"),
    ("volume (eq. 7)", "vol(7)"),
    ("no shear (eq. 9)", "shear(9)"),
    ("no penetration (eq. 10)", "pen(10)"),
    ("wall law (eq. 11)", "wall(11)"),
    ("stone boundary (eq. 13)", "stone(13)"),
]
HEADER_EVERY = 20                                                   # repeat the column titles every 20 rows


def duration(seconds: float) -> str:
    return str(timedelta(seconds=int(max(seconds, 0))))             # e.g. 1:23:45


class TerminalLog:
    def __init__(self, total_steps: int, start_step: int = 0, gpu: bool = False):
        self.total_steps = total_steps                              # steps in the whole run
        self.start_step = start_step                                # 0, or the step we resumed from
        self.gpu = gpu                                              # show GPU memory?
        self.started = time.time()                                  # when this session started
        self.rows = 0                                               # table rows printed so far

    def say(self, text: str = ""):
        stamp = time.strftime("%H:%M:%S")                           # wall-clock time
        print(f"[{stamp}] {text}", flush=True)                      # flush: appears in train.log at once

    def settings_block(self, lines: dict):
        self.say("=" * 78)
        for key, value in lines.items():                            # one setting per line
            self.say(f"{key:<22} {value}")
        self.say("=" * 78)

    def _header(self):
        titles = "  ".join(f"{t:>9}" for _, t in COLUMNS)           # the 8 equation columns
        gpu = "  GPU GB" if self.gpu else ""
        self.say(f"{'step':>8} {'done':>6} {'stage':<5} | {'total':>9} | {titles} | {'lr':>7} | {'s/step':>6} | "
                 f"{'elapsed':>8} | {'left':>8} | resets{gpu}")

    def row(self, step: int, stage: str, total: float, terms: dict, learning_rate: float,
            seconds_per_step: float, resets: int, gpu_memory_gb: float = None):
        if self.rows % HEADER_EVERY == 0:                           # column titles now and then
            self._header()
        self.rows += 1
        done = 100.0 * step / self.total_steps                      # % of the run
        elapsed = time.time() - self.started                        # time in this session
        steps_done = max(step - self.start_step, 1)                 # steps in this session
        left = (self.total_steps - step) * elapsed / steps_done     # remaining time at the average speed
        losses = "  ".join(f"{float(terms[name]):>9.2e}" for name, _ in COLUMNS)
        gpu = f"  {gpu_memory_gb:6.2f}" if self.gpu and gpu_memory_gb is not None else ""
        self.say(f"{step:>8} {done:5.1f}% {stage:<5} | {total:>9.2e} | {losses} | {learning_rate:7.1e} | "
                 f"{seconds_per_step:6.3f} | {duration(elapsed):>8} | {duration(left):>8} | {resets:>6}{gpu}")
