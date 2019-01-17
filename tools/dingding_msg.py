import requests


class DingdingMsg():
    def __init__(self):
        self.url = 'https://oapi.dingtalk.com/robot/send?access_token=fcaa954223a86f7871303b457d8a6058d6a39a45ff5d79d970b2719e40ee2f25'

    def send_msg(self, msg):
        msg_data = {
            'msgtype': 'text',
            'text': {'content': msg}
        }
        requests.post(url=self.url, json=msg_data, headers={'Content-Type': 'application/json'})
