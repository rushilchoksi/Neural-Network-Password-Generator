import pandas as pd
import random

def replace(name,SpecialALpha, SpecialChar,Char):
    Index = SpecialAlpha.index(Char[i])
    Location = name.index(Char[i])
    name = name.replace(name[Location], SpecialChar[Index])
    return name

def appendNumber(String):
    readNumber = pd.read_csv("D:\\college work\\Sem - 7\\SIEM\\number.csv")
    readNumber = readNumber.to_numpy()
    number = random.choice(readNumber)
    String += str(number[0])
    if len(String) < 10:
        appendNumber(String)
    return String

SpecialAlpha = ['a','s','i','o']
SpecialChar = ['@','$','!','0']

readName = pd.read_csv("D:\\college work\\Sem - 7\\SIEM\\name.csv")
readName = readName.to_numpy()
#name = random.choice(readName)
for name in readName:
    name = name[0]
    #location = []
    #locationFlag = []
    Char = []
    for i in range(len(name)):
        if name[i] in SpecialAlpha:
            Char.append(name[i])

    if len(Char) >= 3:
        change = random.randint(2,len(Char))
        for i in range(change):
            StrongPasswordName = replace(name,SpecialAlpha,SpecialChar, Char)
    elif len(Char) != 0:
        for i in range(len(Char)):
            StrongPasswordName = replace(name,SpecialAlpha,SpecialChar, Char)
    if len(StrongPasswordName) <= 10:
        StrongPasswordName = appendNumber(StrongPasswordName)
    with open(f'StrongPasswordName.txt', 'a') as f:
        f.write(StrongPasswordName+"\n")