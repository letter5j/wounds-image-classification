import urllib.request
import sys
from datetime import datetime

# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))
ip = sys.argv[1]
print('ip:', str(ip))
if __name__ == "__main__":
    code = 404

    while(True):
        print(datetime.now().strftime("%H:%M:%S.%f"))
        try:

            code = urllib.request.urlopen("http://%s" %(ip), timeout=0.5).getcode()
            time = datetime.now().strftime("%H:%M:%S.%f")
        except urllib.error.URLError as e:
            if hasattr(e,'code'):
                print (e.code)
            if hasattr(e,'reason'):             
                print (e.reason)
        except urllib.error.HTTPError as e:
            if hasattr(e,'code'):
                print(e.code)
            if hasattr(e,'reason'):
                print(e.reason)
            print('HTTPError!!!')

        if(code == 200):
            print('Get it at %s !' %(time))
            break
        # time.sleep(60)   # Delay for 1 minute (60 seconds).
    
    