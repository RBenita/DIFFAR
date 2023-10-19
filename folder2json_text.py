import json
import sys
from audio import find_TextGrid_files

if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_TextGrid_files(path)
    json.dump(meta, sys.stdout, indent=4)
