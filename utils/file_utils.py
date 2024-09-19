
import os


def is_file_with_valid_extension(path, extension):
    return os.path.isfile(path) and path.endswith(".{}".format(extension))


def get_filename_no_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def add_extension(name, extension):
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
