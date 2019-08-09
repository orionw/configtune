from functools import reduce
import operator
import typing

def get_by_path(root: dict, path_list: typing.List[str]):
    """
    Access a nested object in root by item sequence.
    :param dict_object: the dictionary to access
    :param path_list: the list of strings containing the path
    :return the value of the path
    """
    print(root, path_list)
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