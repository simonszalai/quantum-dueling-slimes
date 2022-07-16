import time


def print_progress(epoch, epochs, step, steps, start_time):
    total_steps = epochs * steps
    steps_done = epoch * steps + step
    percent = round(100 * steps_done / total_steps, 2)
    step_time = time.time()
    time_per_step = (step_time - start_time) / (steps_done + 1)
    time_remaining_s = (total_steps - steps_done) * time_per_step
    time_remaining_str = time.strftime("%H:%M:%S", time.gmtime(time_remaining_s))

    metrics_to_print = [
        f"Epoch: {str(epoch).rjust(4)};",
        f"Round: {str(step).rjust(4)};",
        f"Step: {str(steps_done).rjust(6)};",
        f"Percent: {str(percent).rjust(4)}%;",
        f"Time per step: {str(round(1000 * time_per_step)).rjust(4)} ms;",
        f"Time remaining: {time_remaining_str.rjust(9)};",
    ]

    print("\r", " ".join(metrics_to_print), end="")
