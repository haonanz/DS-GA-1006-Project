import scrapy
import re


class StreeteasySpider(scrapy.Spider):
    name = "streeteasy"

    def start_requests(self):
        urls = ('http://streeteasy.com/sale/{}'.format(i) for i in xrange(1233614, 1300000))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = 'data/sale-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
