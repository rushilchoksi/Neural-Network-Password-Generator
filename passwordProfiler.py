import string
from datetime import datetime
from itertools import combinations
from random import randint, choice, random

def return_leet_words(inputWord, mappingDict, specialCharsList, outputLength):
    """
        uses weighted random technique to return a leet version of the inputWord
        will replace random characters with leet characters
    """
    leet_words_list = []
    for _ in range(outputLength):
        changable_chars_indices = []
        for char_index, char_value in enumerate(inputWord):
            if char_value.lower() in mappingDict.keys():
                changable_chars_indices.append(char_index)

        probaility_of_leet = 0.125
        offset_to_add = 0.35
        leet_word = inputWord
        for _ in changable_chars_indices:
            rand = random()
            if rand > probaility_of_leet:
                random_index = choice(changable_chars_indices)
                changable_chars_indices.remove(random_index)
                leet_word = leet_word[:random_index] + mappingDict[leet_word[random_index].lower()] + leet_word[random_index + 1:]
                probaility_of_leet += offset_to_add

        # print(leet_word)
        leet_words_list.append(leet_word)
    return leet_words_list


def returnLeetWords(inputWord, mappingDict, specialCharsList):
    leetCharsList, leetWordsList = [], [inputWord]
    for charValue in inputWord:
        if charValue.lower() in mappingDict.keys():
            leetCharsList.append(charValue)
    for tempWord in leetCharsList:
        leetWordsList.append(inputWord.replace(tempWord, mappingDict.get(tempWord.lower())))

    return list(set(leetWordsList))

def genPasswordSet(inputWordsList, dobSuffixData, specialCharsList, parsedDateList):
    profiledData = []
    for tempName in inputWordsList:
        for specialChar in specialCharsList:
            for dobSuffixValue in list(dobSuffixData):
                profiledData.append(f'{tempName}{specialChar}{dobSuffixValue}')

        for dateData in parsedDateList:
            profiledData.append(f'{tempName}{dateData}')
            profiledData.append(f'{tempName}{dateData}'.lower())
            profiledData.append(f'{tempName}{dateData}'.upper())
            profiledData.append(f'{tempName}{dateData}'.swapcase())
            
    return list(set(profiledData))

def generateWordList(firstName, lastName, dateOfBirth):
    outputFileName = f'{firstName}.{lastName}.txt'
    with open(outputFileName, 'w') as wordlistFile:
        mappingDict = {'a': '@', 'e': '3', 'f': 'ƒ', 'i': '!', 'o': '0', 's': '$', 'y': '¥'}
        specialCharsList = ['!', '@', '#', '$','%', '^', '&', '*', '(', ')']

        parsedDate = datetime.strptime(dateOfBirth, '%d%m%Y')
        parsedDateList = [str(parsedDate.day).zfill(2), str(parsedDate.month).zfill(2), parsedDate.year]
        finalInputWordList, lastNameWithLeet = return_leet_words(firstName, mappingDict, specialCharsList, 5000), return_leet_words(lastName, mappingDict, specialCharsList, 5000)
        finalInputWordList.extend(lastNameWithLeet)

        dobSuffixData = set()
        for dobSuffixLength in range(1, 5):
            for tempSuffixData in combinations(dateOfBirth, dobSuffixLength):
                dobSuffixData.add(''.join(tempSuffixData))

        profiledData = genPasswordSet(finalInputWordList, dobSuffixData, specialCharsList, parsedDateList)
        profiledData.append(f'{firstName[0]}{lastName[0]}{dateOfBirth}')
        profiledData.append(f'{firstName[0]}{lastName[0]}{dateOfBirth}'.lower())

        for tempPassword in profiledData:
            wordlistFile.write(f'{tempPassword}\n')

    return outputFileName

if __name__ == '__main__':
    firstName, lastName, dateOfBirth = 'Dev', 'Patel', '06092001'
    mappingDict = {'a': '@', 'e': '3', 'f': 'ƒ', 'i': '!', 'o': '0', 's': '$', 'y': '¥'}
    specialCharsList = ['!', '@', '#', '$','%', '^', '&', '*', '(', ')']
    print(f'Created file {generateWordList(firstName, lastName, dateOfBirth)}.')
    # keyword="DevPatelisawsome"
    # print(generateWordList(firstName, lastName, dateOfBirth))
    # print(return_leet_words(keyword, mappingDict, specialCharsList,4))
