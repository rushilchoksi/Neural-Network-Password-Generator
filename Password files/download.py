import urllib.request
import requests



url = "https://raw.githubusercontent.com/josuamarcelc/common-password-list/main/rockyou.txt/rockyou_1.txt"
r = requests.get(url, allow_redirects=True)
open('./rock.txt', 'wb').write(r.content)