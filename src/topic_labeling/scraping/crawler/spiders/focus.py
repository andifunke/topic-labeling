# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from topic_labeling.scraping.crawler.items import CrawlerItem


class FocusSpider(CrawlSpider):
    """Spider for 'Focus Online'"""
    name = 'focus'

    custom_settings = {
        'LOG_FILE': f"../../logs/{name}.log",
        'LOG_ENABLED': True,
        'LOG_LEVEL': "INFO",
        'LOG_STDOUT': False,
    }

    def __init__(self, categories='politik|finanzen', *args, **kwargs):
        super(FocusSpider, self).__init__(*args, **kwargs)

        self.rotate_user_agent = True
        self.allowed_domains = ['focus.de']
        self.start_urls = ['http://www.focus.de']

        FocusSpider.rules = (
            Rule(
                LinkExtractor(
                    allow=('(' + categories + ')',),
                    deny='\.html'
                ),
                follow=True
            ),
            Rule(
                LinkExtractor(
                    allow=('(' + categories + ')(\/\w+)*.*\.html'),
                ),
                callback='parse_page',
            ),
        )
        super(FocusSpider, self)._compile_rules()

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url
        item['visited'] = datetime.datetime.now().isoformat()
        item['text'] = response.css('.articleContent').xpath('.//div[@class="textBlock"]/p').extract()
        item['keywords'] = response.xpath('//meta[@name="keywords"]/@content').extract()
        item['published'] = response.xpath('//meta[@name="date"]/@content').extract_first()
        item['title'] = response.xpath('//meta[@property="og:title"]/@content').extract_first()
        item['description'] = response.xpath('//meta[@name="description"]/@content').extract_first()
        item['author'] = response.css('created').extract()
        return item
