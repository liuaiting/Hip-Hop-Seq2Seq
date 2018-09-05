from lxml import etree
import requests
import random
import json
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36',
}

filter_set = set()
detail_list = []

def get_response(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        parse_html(response.text)
    else:
        print(url, 'failed!')

def parse_html(html):
    doc = etree.HTML(html)
    for li in doc.xpath('//ul[@class="f-hide"]//li'):
        try:
            title = li.xpath('./a/text()')[0]
            song_id = li.xpath('./a/@href')[0].split('id=')[1]
            song_data = {
                'title': title,
                'song_id': song_id,
            }
            if song_id not in filter_set:
                detail_list.append(song_data)
                filter_set.add(song_id)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    with open('sheet_list.json', 'r', encoding='utf-8') as f:
        sheet_list = json.load(f)
    for sheet_data in sheet_list:
        sheet_id = sheet_data['res_id']
        url = 'https://music.163.com/playlist?id={}'.format(sheet_id)
        get_response(url)
        sleep_time = random.choice(range(6, 11)) * 0.2
        time.sleep(sleep_time)
    with open('detail_list.json', 'w', encoding='utf-8') as f:
        json.dump(detail_list, f)

