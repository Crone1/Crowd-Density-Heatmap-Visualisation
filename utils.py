

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