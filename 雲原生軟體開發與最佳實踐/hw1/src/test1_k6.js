// Creator: k6 Browser Recorder 0.6.2

import { sleep, group } from 'k6'
import http from 'k6/http'

export const options = {
  //vus: 100, 
  //duration: '30s', 
  thresholds: {
      http_req_duration: ['p(95)<500'], // threshold for response time
      http_req_failed: ['rate<0.1'], // threshold for request failure rate
  },
  stages: [
      { duration: '10s', target: 10 },
      { duration: '10s', target: 50 },
      { duration: '10s', target: 100 },
  ],
};

export default function main() {
  let response
  let url = 'http://192.168.198.1:5173/'
  group('page_1 - ' + url, function () {
    response = http.post(
      url+'api/v1/todos',
      '{"id":"","name":"do homework","description":"test","status":false}',
      {
        headers: {
          'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
          accept: 'application/json, text/plain, */*',
          'content-type': 'application/json',
          'sec-ch-ua-mobile': '?0',
          'sec-ch-ua-platform': '"Windows"',
          origin: url,
          'sec-fetch-site': 'same-origin',
          'sec-fetch-mode': 'cors',
          'sec-fetch-dest': 'empty',
          'accept-encoding': 'gzip, deflate, br, zstd',
          'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        },
      }
    )

    response = http.get( url + '/api/v1/todos', {
      headers: {
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        accept: 'application/json, text/plain, */*',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
      },
    })
  })

  // Automatically added sleep
  sleep(1)
}