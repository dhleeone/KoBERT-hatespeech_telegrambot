hate_dict = {"한국 남자":["한남충", "한남"], "한국 여자":["한녀충", "한녀"], "중국인":["짱꼴라","짱깨","짱께", "짱개", "장꺠"],  "일본인":["쪽바리","쪽빠리", "쪽본인"], "연장자":["틀딱충", "틀딱"]}

keys = list(hate_dict.keys())
values = list(hate_dict.values())

def purifier(string):
    key_li = []
    val_li = []
    for i in range(len(keys)):
        cnt=0
        for j in range(len(values[i])):
            if values[i][j] in string:
                key_li.append(keys[i])
                val_li.append(values[i][j])
                cnt+=1
            if cnt>=1:
                break;
                
    if len(key_li)== 1:            
        purified_mes = f"'{val_li[0]}'(을)를 '{key_li[0]}'(으)로 바꾸어 표현해주세요!"
        return purified_mes
    elif len(key_li)>1:  
        purified_mes = f"{val_li[:]}(을)를 {key_li[:]}(으)로 바꾸어 표현해주세요!"
        return purified_mes        
    else:
        return ""
