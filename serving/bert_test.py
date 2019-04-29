import requests
import json
url = 'http://172.17.0.2:7002/align_bert'
dic = {'data':[{'origin':'What do you mean? - You go to haiti, and ... I take this job.', 'translate':'你什么意思？—你去海.地…我接受我的工作。'}]}
fdic = {'data':[{'origin':'What do you mean? - You go to haiti, and ... I take this job.', 'translate':'你什么意思？—我不去巴黎…这不是我的工作。'}]}
r = requests.post(url, data=json.dumps(dic))
print(r.text)
