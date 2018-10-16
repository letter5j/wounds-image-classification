import requests

HEADERS = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5)'
                          'AppleWebKit/537.36 (KHTML, like Gecko)'
                          'Chrome/45.0.2454.101 Safari/537.36'),
                          'referer': 'http://stats.nba.com/scores/',
                          'Connection':'close'}

for a in range(97, 123):
    for b in range(97, 123):
        for c in range(97, 123):

            domain = chr(a) + chr(b) + chr(c)
            payload = ('{"domainNames":["%s.io"]}' %(domain))

            r = requests.post('https://api.name.com/v4/domains:checkAvailability', auth=('letter5j','462c013cf6c21266858cab493aa0854a433075ba'), headers=HEADERS, data=payload)
            result = r.json()
            result = result['results'][0]
            if( 'purchasable' in result.keys() and result['purchasable']== True and 'premium' not in result.keys()):
                print(result['domainName'])


# payload = ('{"domainNames":["aae.io"]}')

# r = requests.post('https://api.name.com/v4/domains:checkAvailability', auth=('letter5j','462c013cf6c21266858cab493aa0854a433075ba'), headers=HEADERS, data=payload)
# result = r.json()
# result = result['results'][0]
# print(result)
# if( 'purchasable' in result.keys() and result['purchasable']== True and 'premium' not in result.keys()):
#     print(result['domainName'])

