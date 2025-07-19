import os
import torch
import functools
import multiprocessing


def timeout_decorator(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target(queue, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    queue.put((True, result))  # Success case
                except Exception as e:
                    queue.put((False, e))  # Capture exception
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=target,
                args=(queue, *args),
                kwargs=kwargs
            )
            print("Starting an external process for model saving.")
            # Terminate the process when the main program exits
            process.daemon = True
            process.start()
            print("Starting to wait for the process to end, with a time-out.")
            process.join(seconds)
            if process.is_alive():
                print("Time-out occurred! Attempting to terminate the process.")
                process.kill()  # Ask: Should we use ".terminate()" instead?
                print("Waiting for the process to end.")
                # Consider: Remove this line and just leave the process be
                #     Potential resource leak?
                process.join()
                raise TimeoutError(f"Function '{func.__name__}' timed out.")
            else:
                print("Time-out did not occur.")
                # Edge case: If the process exits without throwing an exception
                #     For example seg fault?
                # Workaround: Make sure something is in the queue
                if queue.empty():
                    raise Exception("Unexpected error: Subprocess exited without posting status")
                success, payload = queue.get()
                if success:
                    return payload
                else:
                    raise payload  # Re-raise the captured exception
        return wrapper
    return decorator


# Issue: It sometimes hangs on some servers, when saving to /scratch
# Workaround: Use multiprocessing to timeout after 300
# Note: multiprocessing requires that checkpoint is on CPU
@timeout_decorator(seconds=300)
def save_checkpoint(checkpoint, idx_iter, CKPT_PATH, save_best):
    print("\n\nStart save_checkpoint")
    # ----- #
    # Define MAX_CKPTS
    # ----- #
    # Background: Preemption may happen during model saving.
    # Workaround: Saving a set amount of copies from previous runs.
    MAX_CKPTS = 2
    # ----- #
    
    # ----- #
    # Save LAST Checkpoint
    # ----- #
    checkpoint_path_last = os.path.join(
        CKPT_PATH, f"checkpoint_last_{idx_iter}.pt"
    )
    torch.save(checkpoint, checkpoint_path_last)
    print("Saved checkpoint_path_last")
    # Remove extra LAST checkpoints if they exceed the limit
    last_ckpts = [
        f
        for f in os.listdir(CKPT_PATH)
        if f.startswith("checkpoint_last")
    ]
    if len(last_ckpts) > MAX_CKPTS:
        print("COND: MAX LOWEST CKPT EXCEED")
        # Extract iteration number from filename and sort
        last_ckpts_sorted = sorted(
            last_ckpts,
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        # Remove the oldest ones
        num_to_remove = len(last_ckpts_sorted) - MAX_CKPTS
        for f in last_ckpts_sorted[:num_to_remove]:
            os.remove(os.path.join(CKPT_PATH, f))
        print("Delete lowest LAST checkpoint")
    # ----- #
    
    # ----- #
    # Save BEST Checkpoint
    # ----- #
    if save_best:
        print("Saving checkpoint_path_best")
        checkpoint_path_best = os.path.join(
            CKPT_PATH, f"checkpoint_best_{idx_iter}.pt"
        )
        torch.save(checkpoint, checkpoint_path_best)
        print("Saved checkpoint_path_best")
        # Remove extra BEST checkpoints if they exceed the limit
        best_ckpts = [
            f
            for f in os.listdir(CKPT_PATH)
            if f.startswith("checkpoint_best")
        ]
        if len(best_ckpts) > MAX_CKPTS:
            print("COND: MAX BEST CKPT EXCEED")
            best_ckpts_sorted = sorted(
                best_ckpts,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            num_to_remove = len(best_ckpts_sorted) - MAX_CKPTS
            for f in best_ckpts_sorted[:num_to_remove]:
                os.remove(os.path.join(CKPT_PATH, f))
            print("Delete lowest BEST checkpoint")
    # ----- #
    print("save_checkpoint completed\n\n")
