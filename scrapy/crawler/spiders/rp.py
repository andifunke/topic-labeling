# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from os import makedirs
from os.path import join

from crawler.items import CrawlerItem

class RPOnlineSpider(CrawlSpider):
    """Spider for 'RP Online'"""
    name = 'rp'
    clazz = RPOnlineSpider
    path = "../../data/feeds/" + name
    makedirs(path, exist_ok=True)

    custom_settings = {
        'LOG_FILE': join(path, name + '.log'),
        'LOG_ENABLED': True,
        'LOG_LEVEL': "DEBUG",
        'LOG_STDOUT': False,
    }

    def __init__(self, categories='politik|wirtschaft', *args, **kwargs):
        super(clazz, self).__init__(*args, **kwargs)

        self.rotate_user_agent = True
        self.allowed_domains = ['rp-online.de']
        self.start_urls = ['http://www.rp-online.de', 'http://www.rp-online.de/thema/']

        clazz.rules = (
            Rule(
                LinkExtractor(
                    allow=('(' + categories + ')\/.*\/$',),
                ),
                follow=True
            ),
            Rule(
                LinkExtractor(
                    allow=('(' + categories + ')\/.+\.\d+$',),
                ),
                callback='parse_page',
            ),
        )
        super(clazz, self)._compile_rules()

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url
        item['visited'] = datetime.datetime.now().isoformat()
        item['published'] = response.selector.xpath('//meta[@property="vr:published_time"]/@content').extract_first()
        item['title'] = response.selector.xpath('//meta[@property="og:title"]/@content').extract_first()
        item['description'] = response.selector.xpath('//meta[@property="og:description"]/@content').extract_first().strip()
        item['text'] = response.selector.xpath('//div[@class="main-text "]/p/text()').extract()
        item['author'] = response.selector.xpath('//meta[@name="author"]/@content').extract()
        item['keywords'] = response.selector.xpath('//meta[@name="keywords"]/@content').extract()
        return item
