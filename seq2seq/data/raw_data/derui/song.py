from lxml import etree
import requests
import random
import json
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36',
}

def get_song_data(item):
    song_id = item['song_id']
    song_title = item['title']
    url = 'http://music.163.com/api/song/lyric?lv=-1&id={}'.format(song_id)

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        doc = json.loads(response.text)
    else:
        print(url, 'failed!')

    # 歌词
    lrc = doc.get('lrc', None)
    if lrc:
        lyric = lrc.get('lyric', None)
    else:
        print('no lrc')

    song_data = {
        'title': song_title,
        'lrc': lyric
    }

    print(song_data)

if __name__ == "__main__":
    with open('detail_list.json', 'r', encoding='utf-8') as f:
        detail_list = json.load(f)
    count = 0
    for detail_data in detail_list[:1]:
        count += 1
        get_song_data(detail_data)
        if count % 500 == 0:
            print(count, 'succeed!')