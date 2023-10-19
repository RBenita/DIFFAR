import json
import sys
import os
from pathlib import Path

def find_npy_files(path, exts=[".npy"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(file_path),...]
    """
    
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
               
           
    meta = []
    for idx, file in enumerate(audio_files):
        meta.append((file))
        if progress:
            # print("helo PROGRESS")
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta
    # return []

if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_npy_files(path)
    json.dump(meta, sys.stdout, indent=4)
