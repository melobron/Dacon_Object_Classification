import configparser


class Argments(object):
    @staticmethod
    def _convert(val):
        try:
            res = eval(val)
        except:
            res = val

        return res

    def update(self, key, val):
        self.__dict__[key] = Argments._convert(val)


def print_user_args(arg, line_limit=80):
    print("-" * 80)

    for key, val in sorted(arg.__dict__.items()):
        val = str(val)
        log_string = key
        log_string += "." * (line_limit - len(key) - len(val))
        log_string += val
        print(log_string)


def parse_args(file_path):

    config = configparser.ConfigParser()
    config.read("script/" + file_path + ".ini")

    args = Argments()

    for key, val in config['default'].items():
        args.update(key, val)

    print_user_args(args)

    return args


# unit test
if __name__ == '__main__':
    arg = parse_args('example')


