from math import log, exp, cos, pi

from typing import Union, List

from .util import Schedule, ConstSched


class CatSched(Schedule):
    def __init__(
        self,
        sched_l: Union[Schedule, float],
        sched_r: Union[Schedule, float],
        where: Union[float, int],
    ):
        super().__init__()
        self.sched_l = sched_l if isinstance(sched_l, Schedule) else ConstSched(sched_l)
        self.sched_r = sched_r if isinstance(sched_r, Schedule) else ConstSched(sched_r)
        self.where = where

    def prep(self, n_steps: int, n_epochs: int, steps_per_epoch: int):
        """Materialize both sub-schedules with correct split."""
        super().prep(n_steps, n_epochs, steps_per_epoch)

        if isinstance(self.where, float):
            # interpret 'where' as fraction of total steps
            n_steps_l = int(self.where * n_steps)
            n_steps_r = n_steps - n_steps_l
            n_epochs_l, n_epochs_r = -1, -1
        elif isinstance(self.where, int):
            # interpret 'where' as epoch index
            n_epochs_l = self.where
            n_epochs_r = n_epochs - n_epochs_l
            n_steps_l, n_steps_r = -1, -1
        else:
            raise ValueError("Unknown type for 'where' (expected float or int).")

        # prepare sub-schedules with their own sub-lengths
        self.sched_l.prep(n_steps_l, n_epochs_l, steps_per_epoch)
        self.sched_r.prep(n_steps_r, n_epochs_r, steps_per_epoch)

        return self

    def __call__(self, it: int, epoch_offset: int = 0):
        """Dynamic call, dispatch to left or right schedule."""
        step = it + epoch_offset * self.steps_per_epoch

        if isinstance(self.where, float):
            cutoff = int(self.where * self.n_steps)
        elif isinstance(self.where, int):
            cutoff = int(self.where * self.steps_per_epoch)
        else:
            raise ValueError("Unknown type for 'where' (expected float or int).")

        if step < cutoff:
            # left side schedule
            return self.sched_l(step, 0)
        else:
            # right side schedule (shift step index)
            rel_step = step - cutoff
            return self.sched_r(rel_step, 0)

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.sched_l, self.sched_r, self.where])


class LinSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch
        x = step / max(1, self.n_steps - 1)
        return self.y_start + (self.y_end - self.y_start) * x

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class CosSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch
        x = -pi * (1 - step / max(1, self.n_steps - 1))
        cos_val = 0.5 + 0.5 * cos(x)
        return self.y_start + (self.y_end - self.y_start) * cos_val

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class ExpSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch
        x = step / max(1, self.n_steps - 1)
        log_start, log_end = log(self.y_start), log(self.y_end)
        val = log_start + (log_end - log_start) * x
        return exp(val)

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class LinWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(LinSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class ExpWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(ExpSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class CosWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(CosSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class MultiStep(Schedule):
    def __init__(self, start: float, gamma: float, *steps: List[Union[int, float]]):
        super().__init__()
        self.start = float(start)
        self.gamma = float(gamma)
        self.steps = steps  # ratio of steps if float, epochs if int

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch
        value = self.start
        for s in self.steps:
            cutoff = s * self.steps_per_epoch if isinstance(s, int) else int(s * self.n_steps)
            if step < cutoff:
                return value
            value *= self.gamma
        return value

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.start, self.gamma] + list(self.steps))


class StepSched(Schedule):
    def __init__(self, start: float, gamma: float, step_every: int, warmup_epochs: int):
        super().__init__()
        self.start = float(start)
        self.gamma = float(gamma)
        self.step_every = step_every
        self.warmup_epochs = warmup_epochs

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch
        milestones = [
            i * self.steps_per_epoch
            for i in range(self.step_every - self.warmup_epochs, self.n_epochs, self.step_every)
        ]

        value = self.start
        for m in milestones:
            if step < m:
                return value
            value *= self.gamma
        return value

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(
            self, [self.start, self.gamma, self.step_every, self.warmup_epochs]
        )


class StepCycleSched(Schedule):
    """
    Creates a schedule where we rise from min_v to max_v in cos schedule,
    then decay to min_v in a cos schedule. the rise and falls happen in length of cycle_length.

    Then we decay the max_v using the decay factor. and repeat until another cycle won't fit in the schedule.
    """

    def __init__(self, min_v: float, max_v: float, decay: float, cycle_length: int):
        super().__init__()
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.decay = float(decay)
        self.cycle_length = cycle_length

    def __call__(self, it: int, epoch_offset: int = 0):
        step = it + epoch_offset * self.steps_per_epoch

        # determine cycle length in steps
        if 0 < self.cycle_length < 1:
            cycle_len = int(self.cycle_length * self.n_steps)
        elif self.cycle_length > 1:
            cycle_len = int(self.cycle_length * self.steps_per_epoch)
        else:
            raise ValueError("cycle_length must be > 0")

        cycle_id = step // cycle_len
        pos_in_cycle = step % cycle_len
        half_cycle = cycle_len // 2

        if pos_in_cycle < half_cycle:
            # rising cosine
            x = pos_in_cycle / half_cycle
            val = self.min_v + (self.max_v - self.min_v) * (0.5 - 0.5 * cos(pi * x))
        else:
            # falling cosine
            x = (pos_in_cycle - half_cycle) / half_cycle
            val = self.max_v + (self.min_v - self.max_v) * (0.5 - 0.5 * cos(pi * x))

        return val * (self.decay**cycle_id)
