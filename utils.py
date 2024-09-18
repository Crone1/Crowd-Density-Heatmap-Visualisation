
import os


def _is_file_with_valid_extension(path, extension):
    return os.path.isfile(path) and path.endswith(".{}".format(extension))


def _add_extension(name, extension):
    """
    Function Goal : add a file extension to the end of the video name if it isn't already there

    name : string - name of the output file
    extension: string - the extension to add to the end of the file name

    return : string - name of the output video ending in the extension
    """
    if name[-4:] != "." + extension:
        return name + "." + extension
    else:
        return name


def _exit_if_empty(value, error, criteria):
    if not value:
        print(f"\nInputError: {error}\nPlease re-run this program ensuring {criteria}")
        exit(0)


def _exit_if_try_fails(function, args, exception, error, criteria):
    try:
        function(*args)
    except exception:
        print(f"\nInputError: {error}\nPlease re-run this program ensuring {criteria}")
        exit(0)