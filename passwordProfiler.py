import string
from datetime import datetime
from itertools import combinations
from random import randint, choice

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

        for _ in range(randint(1, 20)):
            tempRandPasswordDataOne = f'{tempName}'
            for tempNum in range(randint(1, 5)):
                tempRandPasswordDataOne += choice(specialCharsList)
            profiledData.append(tempRandPasswordDataOne)

        for _ in range(randint(1, 20)):
            tempRandPasswordDataTwo = f'{tempName}'
            for tempNum in range(randint(1, 5)):
                tempRandPasswordDataTwo += choice(list(string.digits))
            profiledData.append(tempRandPasswordDataTwo)

    return list(set(profiledData))

def generateWordList(firstName, lastName, dateOfBirth):
    outputFileName = f'{firstName}.{lastName}.txt'
    with open(outputFileName, 'w') as wordlistFile:
        mappingDict = {'a': '@', 'e': '3', 'f': 'ƒ', 'i': '!', 'o': '0', 's': '$', 'y': '¥'}
        specialCharsList = ['!', '@', '#', '$','%', '^', '&', '*', '(', ')']

        parsedDate = datetime.strptime(dateOfBirth, '%d%m%Y')
        parsedDateList = [str(parsedDate.day).zfill(2), str(parsedDate.month).zfill(2), parsedDate.year]
        finalInputWordList, lastNameWithLeet = returnLeetWords(firstName, mappingDict, specialCharsList), returnLeetWords(lastName, mappingDict, specialCharsList)
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
