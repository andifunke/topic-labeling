# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request

from crawler.items import CrawlerItem
from crawler.utils import get_first

class HandelsblattSpider(CrawlSpider):
    """Spider for 'Handelsblatt'"""
    name = 'hb'
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
        item['url'] = response.url.encode('utf-8')
        item['visited'] = datetime.datetime.now().isoformat().encode('utf-8')
        item['published'] = get_first(response.selector.xpath('//div[@class="vhb-article-author-cell"]/@content').extract())
        item['title'] = get_first(response.selector.xpath('//meta[@property="og:title"]/@content').extract())
        item['description'] = get_first(response.selector.xpath('//meta[@name="description"]/@content').extract())
        item['text'] = "".join([s.strip().encode('utf-8') for s in response.selector.css('.vhb-article-content p').xpath('.//text()').extract()])
        item['author'] = [s.encode('utf-8') for s in response.selector.xpath('.//a[@rel="author"]/span[@itemprop="name"]/text()').extract()]
        item['keywords'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="keywords"]/@content').extract()]
        return item
