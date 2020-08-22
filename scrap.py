from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException, ElementNotInteractableException
from selenium.webdriver.support.ui import Select
import time
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
import pprint
import re
import datetime
import locale
from selenium.webdriver.chrome.options import Options 
from stopit import SignalTimeout as Timeout
from stopit import TimeoutException


# necessary to get french date
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

# headless mode : try to set this one
chrome_options = Options()  
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--start-maximized")
# chrome_options.add_argument("--headless")  

def make_data_persistent(backup_link, cols, new_data):
    """
    save data to csv file

    Parameters :
    backup_link : link to save the data to
    cols : columns of the dataframe we want to create - correspond to the review elements we want to get
    new_data : the data we want to save

    Output : None
    We save the data into an existing CSV file or we create a new one.
    """
    
    try:
        backup = pd.read_csv(backup_link)
    except FileNotFoundError:
        backup = pd.DataFrame([], columns=cols)
        
    # if new_data isn't a dataframe, we create a DataFrame from it
    if not isinstance(new_data, pd.core.frame.DataFrame):
        try:
            new_data = pd.DataFrame(new_data, columns=cols)
        except Exception as e:
            print(e)
            raise
    
    new_backup = pd.concat([backup, new_data])
    new_backup.to_csv(backup_link, index=False)


def accessibility(browser):
    """
    if a cookie banner is present, we remove it
    if there is a robot detection pop up, we return False so the main program can load the page again

    Parameters :
    browser : Selenium web element

    Output :
    Returns True if the webpage is accessible otherwise False
    """
    
    # if the 'are you human' popup appears, remove it
    try :
        browser.find_element_by_id('botdetect_abu_nip__overlay')
        return False
    except NoSuchElementException:
        pass
    
    # remove cookie banner because it takes almost 50% height of the page
    try:
        cookie_banner = browser.find_element_by_css_selector('#cookie_warning button, #onetrust-button-group button')
        cookie_banner.click()
        
    except NoSuchElementException:
        pass
    
    return True


def reload_page(url):
    """
    Access a URL, renew attempt if there is an issue with the network

    Parameters :
    url : the url we want to visit

    Output :
    browser : Selenium Webdriver
    """
    
    while True:
        try:
            browser = webdriver.Chrome(options=chrome_options)
            browser.get(url)
            time.sleep(3) # let's the DOM load
            
            browser.find_element_by_css_selector('#logo_no_globe_new_logo') # if it can't get the logo, it means the page isn't loaded
            accessible = accessibility(browser)
    
            if accessible :
                break
            else :
                browser.close()
        except:
            browser.close() # reload the page
    
    return browser


def get_hotels_links_by_page(browser, query):
    """
    get all hotel links inside a webpage. the hotel must have reviews otherwise we skip it.

    Parameters:
    browser : Selenium webelement
    query : city where we get the hotel links from

    Output :
    hotel_links_array : the hotel links from 1 result page
    """
    
    # we can compare it to a previous backup if we have it
    try:
        backup_links = pd.read_csv(f'datasets/backup_hotel_links_{query}.csv')
    except:
        backup_links = None
    
    hotel_links_array = []

    # get link for each hotel in the page
    for hotel in browser.find_elements_by_css_selector('#hotellist_inner .sr_item'):
        
        try:
            # we only want hotel with reviews
            has_rating = hotel.find_elements_by_css_selector('.bui-review-score__badge')
            
            if len(has_rating) > 0 :
                link = hotel.find_element_by_css_selector('h3 .hotel_name_link').get_attribute('href')
                
                existing_link = []
                
                if backup_links is not None:
                    existing_link = backup_links.loc[backup_links['link'] == link]
                
                if len(existing_link) == 0: # if not in database or if the backup file doesn't exist
                    # we want the short version of the link to save space in the csv
                    pattern = re.compile(r'(.+)\?')
                    result = pattern.match(link)
                    short_link = result.group(1)
                    hotel_links_array.append([short_link, 0])
                    
        except StaleElementReferenceException as e:
            print(e, "stale element in get_hotels_links_by_page function")
       
    # persistent data
    make_data_persistent(f'datasets/backup_hotel_links_{query}.csv', ['link', 'has_been_scrapped'], hotel_links_array)
    
    return hotel_links_array


def get_all_hotel_links(browser, query):
    """
    loop through all results pages to get hotel links to scrap

    Parameters:
    browser : Selenium webelement
    query : city where we get the hotel links from

    Output :
    all_links : the hotel links from all the result pages
    """
    
    all_links = []
    
    while True:
        time.sleep(2) # to prevent stale elements
        try :
            all_links.append(get_hotels_links_by_page(browser, query))
            next_btn = browser.find_element_by_css_selector('.bui-pagination__next-arrow:not(.bui-pagination__item--disabled) .bui-pagination__link')
            next_btn.click()
        except NoSuchElementException:
            break # no more results
        except StaleElementReferenceException as e:
            print(e, "stale element in fetching the hotel links")
            
    # flatten the list of all links
    all_links = [link[0] for links_by_page in all_links for link in links_by_page]
    return all_links


def get_value_for_comment_item(review, col):
    """
    collect the values of reviews elements (title, name, rating, etc..)

    Parameters:
    review : Selenium webelement with the review
    col : review element to collect

    Output : the review element value
    """
    
    items = {
        'column': ['nom', 'pays', 'favorite', 'titre', 'bons_points', 'mauvais_points', 'note'],
        'css_selector' : ['.bui-avatar-block__title', '.bui-avatar-block__subtitle', '.c-review-block__badge', '.c-review-block__title', '.c-review__prefix--color-green + .c-review__body--original', '.c-review__prefix:not(.c-review__prefix--color-green) + .c-review__body--original', '.bui-review-score__badge'],
    }
    
    try:
        # we get the index of the column we are dealing with, to get the related css
        col_idx = items['column'].index(col)
        css = items['css_selector'][col_idx]
        
        # we get the value related to the column
        item = review.find_element_by_css_selector(css).text
                
        # in case we are dealing with the Choix de l'utilisateur / favorite column
        # we don't need the text, if favorite element present in the block => 1 otherwise 0
        if col == 'favorite' and item:
            item = 1
                
        return item

    except NoSuchElementException: # in case the item can't be fetched
        if col == 'favorite':
            return 0
        else :
            return 'None' # other exceptions will lead to the closing of the current browser, so we are bubbling them up


def get_comment_data(review, etablissement, elements):
    """
    get comment elements such as the note, comment title, good points, bad points, is_favorite, client country, accomodation

    Parameters :
    review : Selenium Web element containing the review
    etablissement : accomodation from which we get the comment from
    cols : names of the elements we want to get from the review

    Output : comment split into an array of elements - title, good points, bad pointsm is_favorite, etc
    """
    
    split_comment = []
    
    # we can get the first 6 cells from the review block itself
    for el in elements[:7]:
        split_comment.append(get_value_for_comment_item(review, el))
    
    # we collect also the data about the accomodation
    split_comment.append(etablissement['type'])
    split_comment.append(etablissement['lieu'])
    split_comment.append(etablissement['note'])
    
    return split_comment


def get_comments(browser, etablissement, query, link):
    """
    comments can be displayed on several pages. we want max 300 reviews per hotel

    Parameters :
    browser : Seleniu; webdriver
    etablissement : the hotel to get comments from
    query : the city we get the hotels fro. needed to save the comments in a CSV file with the query value
    link : hotel link, needed to indicate that this link has been scrapped already

    Output : 
    Save 300 coments in a new CSV file with the query name and indicate that the hotel link has been scrapped
    """
    
    count = 0
    
    # create new dataframe to save the comments into a backup csv file
    cols = ['nom', 'pays', 'favorite', 'titre', 'bons_points', 'mauvais_points', 'note', 'type_etablissement', 'lieu', 'note_etablissement']
    data = pd.DataFrame([], columns=cols)
    
    while True:
        time.sleep(2) # wait till booking get the following comments
        
        # we get each review and call the script to get its content
        for review in browser.find_elements_by_css_selector('.review_list .review_list_new_item_block'): 
        
            # scroll to the review
            browser.execute_script('arguments[0].scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});', review)
        
            # we add the content of the review
            data.loc[len(data.index)] = get_comment_data(review, etablissement, cols)
            
            count += 1
            
        # when we are done with a section of comments, we check if we have at least 300 comments
        if count >= 300:
            break 
            
        # otherwise, we open a new comments panel
        try :
            next_btn = browser.find_element_by_css_selector('#review_list_score_container .bui-pagination__next-arrow:not(.bui-pagination__item--disabled) a')
            next_btn.click()
        except NoSuchElementException:
            print('no more comments to load')
            break
    
    # we save the scrapped comments in a backup csv file
    make_data_persistent(f'datasets/backup_booking_{query}.csv', cols, data)
    
    # update the hotel_links list of the query, so when we have to do scrap again we can resume at the right spot
    try:
        backup_links = pd.read_csv(f'datasets/backup_hotel_links_{query}.csv')
        mask = backup_links['link'] == link
        backup_links.loc[mask, 'has_been_scrapped'] = 1
        backup_links.to_csv(f'datasets/backup_hotel_links_{query}.csv', index=False)
    except FileNotFoundError as e:
        print(e)


def open_comments_panel(browser):
    """
    get hotel page, open French comments section and return the hotel and browser elements to the main caller
    """
    
    # open reviews panel
    time.sleep(2)
    try:
        btn_cmt = browser.find_element_by_id('show_reviews_tab')
        btn_cmt.click()
    except NoSuchElementException as e:
        print(e, 'there is no review')
        return False
    except ElementClickInterceptedException as e:
        print(e, 'already open') # if the webpage has already been visited by us
        
    #get only french reviews
    time.sleep(2) # wait a little bit till the checkbox is available
    try:
        btn_french = browser.find_element_by_css_selector('.language_filter_checkbox[value="fr"] + span')
        btn_french.click()
    except NoSuchElementException as e:
        print(e, 'there is no french review')
        return False
    
    # get only bad comments - uncomment if you want only bad comments
    # try:
        # select = Select(browser.find_element_by_id('review_score_filter'))
        # select.select_by_index(4)
    # except NoSuchElementException as e:
        # print('no bad comment')

    # it has to take into account the language change
    time.sleep(2)
    
    # get info about accomodation
    try:
        etablissement = {
            'nom': browser.find_element_by_css_selector('.hp__hotel-name').text,
            'type' : browser.find_element_by_css_selector('.hp__hotel-name span').text,
            'note': browser.find_element_by_css_selector('.reviewFloater .bui-review-score__badge').get_attribute('innerHTML'), # sometimes it is hidden
            'lieu' : browser.find_element_by_css_selector('.sb-destination__input').get_attribute("value")
        }
    except NoSuchElementException as e:
        etablissement = {
            'nom': 'None',
            'type' : 'None',
            'note': 'None', # sometimes it is hidden
            'lieu' : 'None'
        }
        
    return etablissement


def scrap_comments(query, nb_comments=6000):
    """
    global function to scrap Booking comments. We want to gather the hotel links first then iterate over them 
    to get at least 300 comments per hotel.

    Parameters :
    query : the city to get the comments from
    nb_comments : the number we want to get eventually

    Output :
    the comments are saved in the CSV file backup_booking_{query}.csv
    """
    
    # we reload the page until we can access it
    browser = reload_page("https://booking.com")
    
    # send query value
    search_input = browser.find_element_by_id('ss')
    search_input.send_keys(query)
    
    # btn submit
    btn_submit = browser.find_element_by_class_name('sb-searchbox__button')
    btn_submit.click()
    
    # if it is not the first time we scrap, we resume the process with the corresponding csv file
    try :
        hotel_links = pd.read_csv(f'datasets/backup_hotel_links_{query}.csv')
        mask = hotel_links['has_been_scrapped'] == 0
        all_links = hotel_links.loc[mask, 'link']
    except Exception as e: 
        print(e, 'the backup file doesnt exist')
        all_links = get_all_hotel_links(browser, query)
        
    # close current window - we are done with getting the hotel links
    browser.close()
    print('got all hotel links')
    

    #get comments for each hotel
    for link in all_links:

        # we reload the page until we can access it
        new_browser = reload_page(link)
            
        try: # 4 min to get all comments, otherwise we go to the next link
            with Timeout(240.0) as timeout_ctx:
                etablissement = open_comments_panel(new_browser)
                if etablissement: # we fetch comment only if we can open the comments panel
                    get_comments(new_browser, etablissement, query, link)
        except TimeoutException as e:
            print(e, 'timeout')
        except Exception as e: # all exceptions not catched in subprocesses will be dealt here
            print(e, 'unexpected error')
        finally:
            new_browser.close()

        # stop the process when we get the defined number of comments per city
        scrapped_comments = pd.read_csv(f'datasets/backup_booking_{query}.csv')
        if scrapped_comments.shape[0] >= nb_comments :
            break
    
    print(f'got the {nb_comments} comments')


# merge datasets
def merge_datasets(cities):
    """
    Since the global function only del with one city at a time, 
    when we have several csv files with comments we can merge them

    Parameters : 
    cities : array of cities for which we get hotel comments

    Output :
    Create a CSV file with all comments
    """
    list_datasets = []

    for city in cities :
        dataset = pd.read_csv(f'datasets/backup_booking_{city}.csv')
        list_datasets.append(dataset)

    comments = pd.concat(list_datasets, axis=0)
    comments.to_csv('datasets/booking_comments_new.csv', index=False)


if __name__ == "__main__" :
    cities = ['Paris', 'Marseille', 'Nice']

    with multiprocessing.Pool() as pool:
        pool.map(scrap_comments, cities)

    merge_datasets(cities)
