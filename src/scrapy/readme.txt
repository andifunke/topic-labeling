To fetch news-articles from one of the following sources, run:

>>> scrapy crawl faz

>>> scrapy crawl focus

By default these spiders crawl for articles from the categories "politik" and "wirtschaft".
You can specify different news-sections by supplying the following argument, e.g.:

>>> scrapy crawl faz -a categories="politik|finanzen|feuilleton"

Available categories are:

FAZ:
finanzen|feuilleton|gesellschaft|technik|politik|wirtschaft|reise|rhein-main|technik-motor|wissen|reise|beruf-chance

Focus:
kultur|panorama|digital|reisen|auto|immobilien|regional|politik|finanzen|wissen|gesundheit
