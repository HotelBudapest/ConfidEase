import feedparser
import requests
from bs4 import BeautifulSoup
import datetime
import re
import html
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Updated industry options with more reliable RSS feeds and fallbacks
INDUSTRIES = {
    "technology": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "https://www.wired.com/feed/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        "https://news.google.com/rss/search?q=technology+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "finance": [
        "https://www.ft.com/rss/home/uk",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.wsj.com/xml/rss/3_7031.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://finance.yahoo.com/news/rssindex",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://news.google.com/rss/search?q=finance+business+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "healthcare": [
        "https://www.medpagetoday.com/rss/headlines.xml",
        "https://www.healthline.com/health-news/rss.xml",
        "https://www.medicalnewstoday.com/newsfeeds/medical.xml",
        "https://rss.medicalnewstoday.com/feednews.xml",
        "https://www.statnews.com/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
        "https://news.google.com/rss/search?q=healthcare+medical+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "education": [
        "https://www.chronicle.com/rss/all",
        "https://feeds.feedburner.com/InsideHigherEd",
        "https://hechingerreport.org/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/Education.xml",
        "https://www.kqed.org/mindshift/feed",
        "https://www.eschoolnews.com/feed/",
        "https://news.google.com/rss/search?q=education+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "manufacturing": [
        "https://www.assemblymag.com/rss",
        "https://feeds.feedburner.com/SupplyChainDigest",
        "https://www.themanufacturer.com/feed/",
        "https://www.automationworld.com/rss.xml",
        "https://www.mmsonline.com/rss/all-content.rss",
        "https://www.reuters.com/pf/api/v3/external/ArticleRss?channelId=2074",
        "https://news.google.com/rss/search?q=manufacturing+industry+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "environment": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml",
        "https://www.ecowatch.com/feeds/news.rss",
        "https://feeds.feedburner.com/greenbiz/news-articles",
        "https://www.theguardian.com/environment/rss",
        "https://e360.yale.edu/feed.xml",
        "https://www.treehugger.com/feeds/latest/",
        "https://news.google.com/rss/search?q=environmental+news&hl=en-US&gl=US&ceid=US:en"
    ],
    "sports": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
        "https://www.espn.com/espn/rss/news",
        "https://api.foxsports.com/v1/rss?partnerKey=zBaFxRyGKCfxBagJG9b8pqLyndmvo7UU",
        "https://www.cbssports.com/rss/headlines/",
        "https://feeds.feedburner.com/bleacherreport-nfl",
        "https://www.si.com/rss/si_topstories.rss",
        "https://news.google.com/rss/search?q=sports+news&hl=en-US&gl=US&ceid=US:en"
    ]
}

# Cache setup - store news for 30 minutes to avoid hitting APIs too frequently
NEWS_CACHE = {}
CACHE_DURATION = 30 * 60  # 30 minutes in seconds

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def get_cached_news(industry):
    """Get news from cache if available and not expired"""
    if industry in NEWS_CACHE:
        timestamp, news = NEWS_CACHE[industry]
        if datetime.datetime.now().timestamp() - timestamp < CACHE_DURATION:
            return news
    return None

def cache_news(industry, news):
    """Store news in cache with current timestamp"""
    NEWS_CACHE[industry] = (datetime.datetime.now().timestamp(), news)

def clean_html(text):
    """Clean HTML tags and entities from text"""
    if not text:
        return ""
    # First unescape HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_image(entry):
    """Extract image URL from feed entry using multiple methods"""
    # Try multiple approaches to find an image
    
    # Method 1: Look in media_content
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if 'url' in media:
                return media['url']
    
    # Method 2: Look in media_thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        for media in entry.media_thumbnail:
            if 'url' in media:
                return media['url']
    
    # Method 3: Look in enclosures
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enclosure in entry.enclosures:
            if 'href' in enclosure and enclosure.get('type', '').startswith('image/'):
                return enclosure['href']
            if 'url' in enclosure and enclosure.get('type', '').startswith('image/'):
                return enclosure['url']
    
    # Method 4: Look in content
    if hasattr(entry, 'content') and entry.content:
        for content in entry.content:
            if 'value' in content:
                soup = BeautifulSoup(content['value'], 'html.parser')
                img = soup.find('img')
                if img and img.get('src'):
                    return img['src']
    
    # Method 5: Look in summary or description
    for field in ['summary', 'description']:
        if hasattr(entry, field) and getattr(entry, field):
            soup = BeautifulSoup(getattr(entry, field), 'html.parser')
            img = soup.find('img')
            if img and img.get('src'):
                return img['src']
    
    # Method 6: Look for image in links
    if hasattr(entry, 'links'):
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                return link.get('href')
    
    # No image found
    return None

def get_entry_date(entry):
    """Extract and standardize date from entry"""
    for date_field in ['published', 'pubDate', 'updated', 'created', 'date']:
        if hasattr(entry, date_field) and getattr(entry, date_field):
            return getattr(entry, date_field)
    return 'N/A'

def fetch_feed_with_timeout(feed_url, timeout_seconds=10):
    """Fetch RSS feed with timeout and rotating user agents"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml, text/html',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    try:
        # Use requests with timeout for the initial fetch
        response = requests.get(feed_url, timeout=timeout_seconds, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the feed content
        feed = feedparser.parse(response.content)
        if feed and hasattr(feed, 'entries') and feed.entries:
            logger.info(f"Successfully fetched {len(feed.entries)} entries from {feed_url}")
            return feed
        else:
            logger.warning(f"Feed successfully fetched but no entries found in {feed_url}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {feed_url}: {e}")
    
    # Fallback to direct parsing if request fails
    try:
        logger.info(f"Trying direct feedparser on {feed_url}")
        feed = feedparser.parse(feed_url)
        if feed and hasattr(feed, 'entries') and feed.entries:
            logger.info(f"Direct parsing successful: {len(feed.entries)} entries from {feed_url}")
            return feed
        else:
            logger.warning(f"Direct parsing: no entries found in {feed_url}")
    except Exception as e2:
        logger.warning(f"Direct parsing error for {feed_url}: {e2}")
    
    return None

def fetch_news(industry):
    """Fetch news for a specific industry from RSS feeds with improved parsing"""
    # Check cache first
    cached_news = get_cached_news(industry)
    if cached_news:
        logger.info(f"Returning {len(cached_news)} cached news items for {industry}")
        return cached_news
    
    if industry not in INDUSTRIES:
        logger.warning(f"Unknown industry: {industry}")
        return []
    
    logger.info(f"Fetching fresh news for {industry}")
    news_items = []
    successful_feeds = 0
    
    # Try to get at least 10 news items from all available sources
    for feed_url in INDUSTRIES[industry]:
        # If we already have enough news items, break
        if len(news_items) >= 15 and successful_feeds >= 2:
            logger.info(f"Already have {len(news_items)} items from {successful_feeds} feeds, stopping fetch")
            break
            
        try:
            # Fetch feed with timeout
            feed = fetch_feed_with_timeout(feed_url)
            if not feed:
                continue
            
            # Get source name
            source_name = feed.feed.get('title', feed_url.split('/')[2])
            if not source_name or source_name.strip() == '':
                source_name = feed_url.split('/')[2].replace('www.', '')
            
            # Counter for items from this feed
            items_from_feed = 0
            
            for entry in feed.entries[:5]:  # Get top 5 news from each source
                # Extract image
                image_url = extract_image(entry)
                
                # Get summary
                summary = ''
                for field in ['summary', 'description', 'content']:
                    if hasattr(entry, field) and getattr(entry, field):
                        content_value = getattr(entry, field)
                        if isinstance(content_value, list) and len(content_value) > 0 and 'value' in content_value[0]:
                            summary = content_value[0]['value']
                        else:
                            summary = content_value
                        break
                
                # Clean summary
                clean_summary = clean_html(summary)
                if clean_summary:
                    # Limit summary length
                    clean_summary = clean_summary[:200] + ('...' if len(clean_summary) > 200 else '')
                
                # Get title
                title = clean_html(entry.get('title', 'No Title'))
                if not title or title.strip() == '':
                    continue
                
                # Get link
                link = entry.get('link', '#')
                if not link or link == '#':
                    continue
                
                # Get publication date
                published = get_entry_date(entry)
                
                # Check for duplicates (based on title similarity)
                duplicate = False
                for existing in news_items:
                    if existing['title'].lower() == title.lower():
                        duplicate = True
                        break
                
                if not duplicate:
                    news_items.append({
                        'title': title,
                        'link': link,
                        'published': published,
                        'summary': clean_summary,
                        'image': image_url,
                        'source': source_name
                    })
                    items_from_feed += 1
            
            if items_from_feed > 0:
                successful_feeds += 1
                logger.info(f"Added {items_from_feed} items from {source_name}")
                
        except Exception as e:
            logger.error(f"Error processing feed from {feed_url}: {e}")
    
    logger.info(f"Total news items fetched for {industry}: {len(news_items)}")
    
    # If we have news items, cache them
    if news_items:
        cache_news(industry, news_items)
        return news_items
    
    # If we have no news items but cache exists, return the expired cache as a fallback
    if industry in NEWS_CACHE:
        logger.warning(f"No fresh news for {industry}, returning expired cache")
        return NEWS_CACHE[industry][1]
    
    # Last resort - try to get general news
    return fetch_general_news()

def fetch_general_news():
    """Fetch general news as a fallback when industry-specific news fails"""
    logger.info("Fetching general news as fallback")
    general_feeds = [
        "https://news.google.com/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.huffpost.com/section/front-page/feed",
        "https://www.yahoo.com/news/rss"
    ]
    
    news_items = []
    for feed_url in general_feeds:
        try:
            feed = fetch_feed_with_timeout(feed_url)
            if not feed or not hasattr(feed, 'entries') or not feed.entries:
                continue
                
            source_name = feed.feed.get('title', feed_url.split('/')[2])
            
            for entry in feed.entries[:3]:  # Get top 3 news from each general source
                # Process entry similar to fetch_news function
                image_url = extract_image(entry)
                
                title = clean_html(entry.get('title', 'No Title'))
                link = entry.get('link', '#')
                published = get_entry_date(entry)
                
                # Get summary
                summary = ''
                for field in ['summary', 'description', 'content']:
                    if hasattr(entry, field) and getattr(entry, field):
                        content_value = getattr(entry, field)
                        if isinstance(content_value, list) and len(content_value) > 0 and 'value' in content_value[0]:
                            summary = content_value[0]['value']
                        else:
                            summary = content_value
                        break
                
                clean_summary = clean_html(summary)
                if clean_summary:
                    clean_summary = clean_summary[:200] + ('...' if len(clean_summary) > 200 else '')
                
                news_items.append({
                    'title': title,
                    'link': link,
                    'published': published,
                    'summary': clean_summary,
                    'image': image_url,
                    'source': source_name
                })
        except Exception as e:
            logger.error(f"Error fetching general news from {feed_url}: {e}")
    
    logger.info(f"Fetched {len(news_items)} general news items as fallback")
    return news_items

def get_news_for_industry(industry='technology'):
    """Main function to get news for a specific industry"""
    return fetch_news(industry)

def get_available_industries():
    """Return list of available industries"""
    return sorted(INDUSTRIES.keys())
