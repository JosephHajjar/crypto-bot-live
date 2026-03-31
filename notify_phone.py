import os, time
import urllib.request as r

target_file = 'models/holy_grail_edge_found.pth'

while True:
    if os.path.exists(target_file):
        req = r.Request(
            'https://ntfy.sh/TradeBot5234', 
            data=b'HOLY GRAIL SECURED! The machine learning bot has found a hyperparameter configuration that beats the market on entirely unseen test data! Check your PC.', 
            headers={'Title':'ML Trading Bot Alert', 'Tags':'rotating_light,moneybag', 'Priority':'high'}
        )
        try:
            r.urlopen(req)
        except:
            pass
        break
    time.sleep(15)
