import sys
import gc
import inspect

def get_size(obj):
    """Recursively calculate size of objects, including referenced objects."""
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_size(k) for k in obj.keys())
        size += sum(get_size(v) for v in obj.values())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_size(i) for i in obj)
    return size

def get_variable_names(obj):
    """Get all variable names that refer to the same object."""
    names = []
    for frame in inspect.stack():
        for name, value in frame.frame.f_locals.items():
            if value is obj:
                names.append(name)
    return names

def rank_objects_by_usage(show_count = 10, unit = "GB"):
    '''
    Fetches all objects currently in memory and ranks them by size; useful for debugging memory issues.
    Args:
        show_count (int):   number of biggest objects to show memory usage for
        unit (str):         unit of memory to use, i.e. bytes, KB, MB, or GB
    Returns:
        object_info (list): list of object info for all objects in memory
    '''
    # Define the memory size divisor for the desired units
    if unit.upper() == "GB":
        divisor = 1024 ** 3
    elif unit.upper() == "MB":
        divisor = 1024 ** 2
    elif unit.upper() == "KB":
        divisor = 1024
    elif unit.upper() == "B" or unit == "bytes":
        divisor = 1
    else:
        print(f"Unrecognized unit for memory ranking was given: {unit}")
        unit = "MB"
        divisor = 1024 ** 2
    # Collect all objects
    all_objects = gc.get_objects()
    # Create a list of (object, size, names) tuples
    object_info = []
    for obj in all_objects:
        try:
            size = get_size(obj)
            names = get_variable_names(obj)
            if names:  # Only consider objects with at least one variable name
                object_info.append((obj, size, names))
        except Exception as e:
            # Handle exceptions for objects that can't be sized or inspected
            continue
    # Sort by size
    object_info.sort(key=lambda x: x[1], reverse=True)
    # Print ranked objects by memory usage
    for i, (obj, size, names) in enumerate(object_info[:show_count], 1):
        print(f"{i}. Size: {size/divisor:.2f} {unit} - Names: {', '.join(names)}")
    return object_info
