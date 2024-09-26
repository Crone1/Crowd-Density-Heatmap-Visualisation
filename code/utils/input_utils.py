def exit_if_false(condition, error, criteria):
    if not condition:
        print(f"\nInputError: {error}\nPlease re-run this program ensuring {criteria}")
        exit(0)


def exit_if_try_fails(function, args, exception, error, criteria):
    try:
        function(*args)
    except exception:
        print(f"\nInputError: {error}\nPlease re-run this program ensuring {criteria}")
        exit(0)
