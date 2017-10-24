# Web Crawling Application Testing

### Web crawling based testing

We show that web crawling based testing is application specific. Testers have to manually configure the webcrawler for each web application
and then they can perform test.

**web-crawler.py** is a sample of a crawler program that extracts features from a web form

**web-crawler-config.yaml** is the configuration file for web-crawler config. We can see that this config is application specific.

### Project Dependencies
All packages are installed using pip, which is a python package manager.

BeautifulSoup4 (version 4.6)
- pip install beautifulsoup4

numpy (version 1.13.3)
- pip install

scipy (version 1.0.0rc2)
** If using windows: install 'wheel' for scipy package
    wheel (version 0.30.0)
    - pip install wheel

    then download scipy from 'http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy'
    install scipy using wheel
    - pip install C:\user\Downloads\scipy-1.0.0rc2-cp36-cp36m-win32.whl
** Else, install from pip
    - pip install scipy

Note: If on Windows, numpy and scipy can also be installed using Anaconda, which is a python data
science platform. Anaconda comes with the necessary packages for data science projects right out
of the box.
visit: 'anaconda.com' for more details

scikit-learn (0.19.1)
- pip install -U scikit-learn

*How do we enhance web crawling based technique to adapt multiple web applications with lower labour-intensive task?*