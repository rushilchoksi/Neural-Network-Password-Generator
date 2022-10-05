import sys
import hashlib
import requests
from getpass import getpass

breachPasswordFlag, userPassword = False, getpass('Enter password: ')
hashVal = hashlib.sha1(userPassword.encode()).hexdigest().upper()

requestData = requests.get(f'https://api.pwnedpasswords.com/range/{hashVal[:5]}')
for tempPasswordHash in requestData.text.split('\r\n'):
    tempPasswordVal = tempPasswordHash.split(":")
    if hashVal == f'{hashVal[:5]}{tempPasswordVal[0]}':
        breachPasswordFlag = True
        print(f'\nThe password provided as input has been compromised in {tempPasswordVal[1]} data breaches.')

if breachPasswordFlag == False:
    print('\nYour password is secure as of now and now detected in any of the data breaches! 🤙🏻')
else:
    print('Your password has been compromised in one or more data breaches, use our software to stay safe! 😂')
