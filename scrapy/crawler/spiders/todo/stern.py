# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request

from crawler.items import CrawlerItem
from crawler.utils import get_first

class SternSpider(CrawlSpider):
    """Spider for 'Stern.de'"""
    name = 'stern'
    rotate_user_agent = True
    allowed_domains = ['www.stern.de']
    start_urls = ['http://www.stern.de']
    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    '(politik|wirtschaft|panorama|themen)\/$',
                    'themen\/.+\.html$',
                ),
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=('(politik|wirtschaft|panorama)(\/\w+)*.*\.html$'),
                deny=('themen')
            ),
            callback='parse_page',
        ),
    )

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url.encode('utf-8')
        item['visited'] = datetime.datetime.now().isoformat().encode('utf-8')
        item['published'] = get_first(response.selector.xpath('//time/@datetime').extract())
        item['title'] = get_first(response.selector.xpath('//meta[@property="og:title"]/@content').extract())
        item['description'] = get_first(response.selector.xpath('//meta[@name="description"]/@content').extract())
        item['text'] = "".join([s.strip().encode('utf-8') for s in response.selector.css('.article-content>.rtf-content-wrapper>P').xpath('.//text()').extract()])
        item['author'] = [s.encode('utf-8') for s in response.selector.xpath('//div[@class="name"]/a[@rel="author"]/text()').extract()]
        item['keywords'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="news_keywords"]/@content').extract()]
        return item
