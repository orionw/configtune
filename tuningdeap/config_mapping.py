from functools import reduce
import operator
import typing
import collections

def get_by_path(root: dict, path_list: typing.List[str]):
    """
    Access a nested object in root by item sequence.
    :param dict_object: the dictionary to access
    :param path_list: the list of strings containing the path
    :return the value of the path
    """
    return reduce(operator.getitem, path_list, root)

def set_by_path(dict_object: dict, path_list: typing.List[str], value, is_bool: bool = False):
    """
    Set a value in a nested object in root by item sequence.
    :param dict_object: the dictionary to access
    :param path_list: the list of strings containing the path
    :param value: the value to set that path to
    """
    assert type(dict_object) == dict, "was given wrong item type for dict_object: expected dict got {}".format(type(dict_object))
    if is_bool:
        get_by_path(dict_object, path_list[:-1])[path_list[-1]] = bool(value)
    else:
        get_by_path(dict_object, path_list[:-1])[path_list[-1]] = value

def get_paths(source: dict) -> typing.List[typing.List[str]]:
    """
    Returns all paths from a dictionary object to all different keys
    :param source: the dict object to find paths in
    :return a list of paths, where each path is a list of keys
    """
    paths = []
    if isinstance(source, collections.MutableMapping):  # found a dict-like structure...
        for k, v in source.items():  # iterate over it; Python 2.x: source.iteritems()
            paths.append([k])  # add the current child path
            paths += [[k] + x for x in get_paths(v)]  # get sub-paths, extend with the current
    # else, check if a list-like structure, remove if you don't want list paths included
    elif isinstance(source, collections.Sequence) and not isinstance(source, str):
        #                          Python 2.x: use basestring instead of str ^
        for i, v in enumerate(source):
            paths.append([i])
            paths += [[i] + x for x in get_paths(v)]  # get sub-paths, extend with the current
    return paths