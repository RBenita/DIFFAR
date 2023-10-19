import os
import textgrid
import csv
from g2p_en import G2p

def get_filenames_from_csv(csv_path):
    with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            name_list = []
            for row in reader:
                name_list.append(row[0])
    return name_list[1:]

def making_textgrid_file_start_with_None(durations, phonemes, target_path_textGrid_file, Is_G2P=False):
    
    if not(Is_G2P):
        ##adding 0.25 sec of None in the start of the textgrid
        total_length = 0
        total_length += round(float(0.25),2)

        # regular behaviour
        for phoneme_time in durations[0]:
            total_length += round(float(phoneme_time),2)
        # end

        total_start_time = 0
        total_end_time = total_length
        tg = textgrid.TextGrid()
        words = textgrid.IntervalTier('words', total_start_time, total_end_time)
        phones = textgrid.IntervalTier('phones', total_start_time, total_end_time)
        running_time=0
        cur_mark=''
        cur_dur=round(float(0.25),2)
        phones.add(running_time, running_time + cur_dur,  cur_mark)
        running_time += cur_dur

        for i in range(len(phonemes)):
            if i==0 and phonemes[i]=='':
                total_length=total_length-round(float(durations[0][i]),2)
                continue
            cur_mark = phonemes[i]
            cur_dur = round(float(durations[0][i]),2)
            phones.add(running_time, running_time + cur_dur,  cur_mark)
            running_time += cur_dur
        
        assert round(float(running_time),2)==round(float(total_length),2), f"they supposed to be the same. {running_time}, {total_length}"
        
        ## the next line will change the max time of phonemes and then wont be 2 "" at the end ##
        phones.maxTime = running_time
        tg.append(words)
        tg.append(phones)
        tg.write(target_path_textGrid_file)

    else:
        ##adding 0.25 sec of None in the start of the textgrid
        shift= 0 
        min_len = 0.04
        total_length = 0
        total_length += round(float(0.25),2)

        ## start make sure every length is more then min_len
        for phoneme_time in durations[0]:
            if phoneme_time>=min_len:
                total_length += round(float(phoneme_time),2)
            else:
                # print(f"this is too small: {phoneme_time}")
                total_length += round(float(min_len),2)
        ## end make sure every length is more then min_len

        # ### start change length so max 0.01 or third
        # for phoneme_time in durations[0]: 
        #     original_pohoneme_length = phoneme_time
        #     phoneme_time = round(float(phoneme_time),2)
        #     if phoneme_time>=min_len:
        #         total_length += round(float(phoneme_time),2)
        #     else:
        #         print(f"i need to cahnge duration {original_pohoneme_length}")
        #         added_duration =round(float(max(phoneme_time/3,0.01)),2) 
        #         print(f"I add:    {added_duration}")
        #         print(f"the new length is:   {phoneme_time + added_duration }")

        #         total_length += phoneme_time + added_duration
        #         # total_length += round(float(phoneme_time),2) + added_duration      

        ### end change length so max 0.01 or third

        # total_length = int(torch.sum(durations[0]))
        total_start_time = 0
        total_end_time = total_length
        tg = textgrid.TextGrid()
        words = textgrid.IntervalTier('words', total_start_time, total_end_time)
        phones = textgrid.IntervalTier('phones', total_start_time, total_end_time)
        running_time=0
        cur_mark=''
        cur_dur=round(float(0.25),2)
        phones.add(running_time, running_time + cur_dur,  cur_mark)
        running_time += cur_dur
        for i in range(len(phonemes)):
            if i==0 and phonemes[i]=='':
                total_length=total_length-round(float(durations[0][i]),2)
                # print( f"Delete {durations[0][i]} ms")
                continue
            cur_mark = phonemes[i]
            cur_dur = round(float(durations[0][i]),2)

            ###start: make sure dur>= min_length
            if cur_dur<min_len:
                cur_dur=min_len
                # print("cahnged duration to min duration")
            ###end: make sure dur>= min_length

            # ###start: max 0.01 or third
            # if cur_dur<min_len:
            #     print(f"i cahnged duration {cur_dur}")
            #     added_duration = round(float(max(cur_dur/3,0.01)),2)
            #     cur_dur=cur_dur+added_duration
            # ###end: max 0.01 or third

            phones.add(running_time, running_time + cur_dur,  cur_mark)
            running_time += cur_dur
        
        assert round(float(running_time),2)==round(float(total_length),2), f"they supposed to be the same. {running_time}, {total_length}"
        ## the next line will change the max time of phonemes and then wont be 2 "" at the end ##
        phones.maxTime = running_time
        ###
        tg.append(words)
        tg.append(phones)
        tg.write(target_path_textGrid_file)

def grapheme_to_phoneme_func(text_list):
    g2p = G2p()
    for text in text_list:
        out = g2p(text)
        new_phonemes = []
    i=0
    for i in range(len(out)):
        if  out[i]==' ':
            if out[i+1]==',':
                i+=2
                continue
            if out[i+1]=='.':
                i+=2
                continue
            if out[i+1]==';':
                i+=2
                continue

            if out[i-1]==',':
                new_phonemes.append('')
                i+=1
                continue
            if out[i-1]=='.':
                new_phonemes.append('')
                i+=1
                continue
            if out[i-1]==';':
                new_phonemes.append('')
                i+=1
                continue
            continue

        elif  out[i]=='.':
            i+=1
            continue

        elif  out[i]==',':      
            i+=1
            continue
        elif  out[i]==';':      
            i+=1
            continue
        elif  out[i]=="'":     
            i+=1
            continue
        
        new_phonemes.append(out[i])
        i=i+1
    # print("Adding None at the end")
    new_phonemes.append('')
    return new_phonemes
    
def cleaning_text_from_Punctuation(origin_text_path, target_text_path):
    ext = ('.txt')
    punctuation_list=[',','.','?','!','-',';','"']
    for files in os.listdir(origin_text_path):
        if files.endswith(ext):
            file_name = os.path.basename(files)
            file_name = os.path.splitext(file_name)[0]
            with open(f"{origin_text_path}/{files}", 'r') as file:
                data = file.read().rstrip()
                new_file = open(f"{target_text_path}/{files}", "w")
                for char in punctuation_list:
                    if char == "-":
                        data = data.replace(char, " ")
                    else:
                        data = data.replace(char, "")
                new_file.write(data)
                new_file.close()

if __name__ == "__main__":
    print("playground.py")
