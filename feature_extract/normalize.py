# Normalizing text input for tokenizing more clearly
def useNormalize(texts):
    print('Normalizing Texts')

    for m in range(len(texts)):
        tokens = texts[m].strip().split()
        temp = []
        for i in range(len(tokens)):
            pre_token = ''
            j = 0
            while(j < len(tokens[i])):
                if pre_token == tokens[i][j]:
                    tokens[i] = tokens[i][0 : j] + tokens[i][min(len(tokens[i]), j + 1):]
                    j-=1
                else:
                    pre_token = tokens[i][j]
                j += 1

            temp.append(tokens[i])
        texts[m] = ' '.join(temp)
    return texts
