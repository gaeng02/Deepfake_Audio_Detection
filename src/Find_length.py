import os
#import torchaudio

def find_length () : 
    M, m = 0, 1237870
    path = ["./real", "./fake"]
    
    for p in path : 
        for file_name in os.listdir(p) : 
            if file_name.endswith(".ogg") : 
                file_path = os.path.join(p, file_name)
                wf, sr = torchaudio.load(file_path)

                M = max(M, wf.shape[1])
                m = min(m, wf.shape[1])
                
    return [M, m]

def find_object (m) :
    path = ["./real", "./fake"]
    for p in path : 
        for file_name in os.listdir(p) :
            if file_name.endswith(".ogg") : 
                    file_path = os.path.join(p, file_name)
                    wf, sr = torchaudio.load(file_path)

                    if wf.shape[1] == m : print(file_path); return

    return None

if (__name__ == "__main__") :
    length = find_length() # [Max, Min]
    
    find_object(length[0])
    find_object(length[1])

    
