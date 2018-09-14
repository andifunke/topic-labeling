# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request
from os import makedirs
from os.path import join

from crawler.items import CrawlerItem

class HandelsblattSpider(CrawlSpider):
    """Spider for 'Handelsblatt'"""
    name = 'hb'
    path = "../../data/feeds/" + name
    makedirs(path, exist_ok=True)

    custom_settings = {
        'LOG_FILE': join(path, name + '.log'),
        'LOG_ENABLED': True,
        'LOG_LEVEL': "INFO",
        'LOG_STDOUT': False,
    }

    rotate_user_agent = True
    allowed_domains = ['www.handelsblatt.com']
    start_urls = [
        'http://www.handelsblatt.com',
        'http://www.handelsblatt.com/themen',
    ]
    rules = (
        Rule(
            LinkExtractor(
                allow=('(politik|finanzen|panorama|themen).*\/$'),
                deny=('\.html')
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=('(politik|finanzen|panorama).*\/\d+\.html$'),
                deny=('video')
            ),
            callback='parse_page',
        ),
    )

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url
        item['visited'] = datetime.datetime.now().isoformat()
        item['published'] = response.selector.xpath('//div[@class="vhb-article-author-cell"]/@content').extract_first()
        item['title'] = response.selector.xpath('//meta[@property="og:title"]/@content').extract_first()
        item['description'] = response.selector.xpath('//meta[@name="description"]/@content').extract_first()
        item['text'] = [s.strip() for s in response.selector.css('.vhb-article-content p').xpath('.//text()').extract()]
        item['author'] = [s for s in response.selector.xpath('.//a[@rel="author"]/span[@itemprop="name"]/text()').extract()]
        item['keywords'] = [s for s in response.selector.xpath('//meta[@name="keywords"]/@content').extract()]
        return item
