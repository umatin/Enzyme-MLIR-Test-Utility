import json


def parse_results(stdout: str):
    """
    Parse the stdout from MLIR's printMemrefF64/printF64 functions.

    This assumes:
    1. Floats (printF64) are followed by a newline.
    2. The printMemrefF64 implementation always has a newline after "data =".
    """

    def is_number(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    return [
        json.loads(line)
        for line in stdout.splitlines()
        if is_number(line) or line.startswith("[")
    ]
