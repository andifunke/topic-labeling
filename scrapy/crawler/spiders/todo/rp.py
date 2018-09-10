# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request

from crawler.items import CrawlerItem
from crawler.utils import get_first

class RPOnlineSpider(CrawlSpider):
    """Spider for 'RP Online'"""
    name = 'rp'
    rotate_user_agent = True
    allowed_domains = ['www.rp-online.de']
    start_urls = [
            'http://www.rp-online.de',
            'http://www.rp-online.de/thema/',
    ]
    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    '(politik|wirtschaft|panorama|thema)\/.*\/$',
                ),
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=(
                    '(politik|wirtschaft|panorama|thema)\/.+\.\d+$',
                ),
            ),
            callback='parse_page',
        ),
    )

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url.encode('utf-8')
        item['visited'] = datetime.datetime.now().isoformat().encode('utf-8')
        item['published'] = get_first(response.selector.xpath('//meta[@property="vr:published_time"]/@content').extract())
        item['title'] = get_first(response.selector.xpath('//meta[@property="og:title"]/@content').extract())
        item['description'] = get_first(response.selector.xpath('//meta[@property="og:description"]/@content').extract()).strip()
        item['text'] = "".join([s.strip().encode('utf-8') for s in response.selector.xpath('//div[@class="main-text "]/p/text()').extract()])
        item['author'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="author"]/@content').extract()]
        item['keywords'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="keywords"]/@content').extract()]
        return item
