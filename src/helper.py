from pathlib import Path
import shutil
from src import biased_updating
from src import unbiased_updating
from src import change_point_updating
from src import omniscient_updating

"""
Remove all files and subdirectories inside a directory,
without deleting the directory itself.

Parameters
----------
path : Path
    Directory whose contents will be deleted.
"""
def clear_dir(path: Path) -> None:
    path = path.resolve()
    assert "outputs" in path.parts, f"Refusing to clear {path}"

    if not path.exists():
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

MODEL = {
    "Biased": biased_updating.biased_optimized,
    "Unbiased": unbiased_updating.unbiased_optimized,
    "Changepoint": change_point_updating.change_point_optimized,
    "Omniscient": omniscient_updating.omniscient
}

"""
Obtains the respective function

Parameters
----------
model_name : String

Returns
-------
MODEL[model_name] : ?
    uses the string as a key to the dictionary above
"""
def get_update_function(model_name: str):
    try:
        return MODEL[model_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid models: {list(MODEL)}"
        ) from e

"""
Filters the model_names to remove duplicates and bad names

Parameters
----------
model_names : list

Returns
-------
functions : list
    list of functions with no duplicate functions
"""
def model_functions(model_names):
    seen = set()
    functions = []

    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)

        update_fn = get_update_function(model_name)
        functions.append(update_fn)

    return functions


    



