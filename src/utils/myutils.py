import pickle
import time
import os

def pickle_save(filename, data, try_multiple_save=100, verbose=False):
    if not filename.endswith('.pkl'): filename += '.pkl'
    if try_multiple_save <= 0: try_multiple_save = 1

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for _ in range(try_multiple_save):
        try:
            with open(filename, "wb") as fp:
                pickle.dump(data, fp)
            return
        except Exception as e:
            if verbose: print(f"Save attempt failed with error: {e}. Retrying...")
            time.sleep(0.5)
            continue
    raise ValueError("Save failed after multiple attempts. Check file directory and permissions.")


def pickle_load(file, verbose=False):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    if verbose:
        print(f"Pickle file {file} loaded; datatype {str(type(file))}")
    return data