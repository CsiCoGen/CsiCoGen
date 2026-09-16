from datetime import datetime
import os
import sys
import traceback

INFO = 0
WARNING = 2

log_level = INFO
_log_file = None


def _format(level, messages):
    timestr = datetime.strftime(datetime.now(), '%m.%d/%H:%M')
    father = traceback.extract_stack()[-4]
    func_info = f'{father[0].split("/")[-1]}:{str(father[1]).ljust(4, " ")}'
    m = ' '.join(map(str, messages))
    return f'{level} {timestr} {func_info}] {m}'


def set_file(path):
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _log_file = open(path, 'w')


def _write_file(msg):
    if _log_file is None:
        return
    try:
        _log_file.write(msg + '\n')
        _log_file.flush()
    except Exception:
        pass


def info(*messages):
    if log_level > INFO:
        return
    msg = _format('I', messages)
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()
    _write_file(msg)


def warning(*messages):
    if log_level > WARNING:
        return
    msg = _format('W', messages)
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()
    _write_file(msg)
