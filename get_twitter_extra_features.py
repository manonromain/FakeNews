
# coding: utf-8


import tweepy
import time
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

#### GETTING CREDENTIALS

with open("credentials.txt") as f:
    consumer_key = f.readline()[:-1] # [:-1] to remove the '\n'
    consumer_secret = f.readline()[:-1]
    oauth_token = f.readline()[:-1]
    oauth_token_secret = f.readline()[:-1]

#### GET ALL USER/POST IDS OF TWITTER15/16 

files_tree = []
paths_folder = ['rumor_detection_acl2017/twitter15/tree', 'rumor_detection_acl2017/twitter16/tree']

for path_folder in paths_folder:
    files_tree += [join(path_folder, f) for f in listdir(path_folder) if isfile(join(path_folder, f))]
    
list_user_post = [] #will contain all tuples (user, post)

for file_name in files_tree:
    list_user_post_file = []
    with open(file_name) as f:
        for line in f:
            line_split = line.split("'")
            list_user_post_file.append((line_split[1], line_split[3]))
            list_user_post_file.append((line_split[7], line_split[9]))
            
    list_user_post.extend(list(set(list_user_post_file)))


all_users = list(set([elt[0] for elt in list_user_post]))
all_posts = list(set([elt[1] for elt in list_user_post]))
all_users.sort()
all_posts.sort()
if all_users[-1]=='ROOT':
    del(all_users[-1])
if all_posts[-1]=='ROOT':
    del(all_posts[-1])  

print(f"Nb users: {len(all_users)}")
print(f"Nb posts: {len(all_posts)}")


#### GET POSTS' CONTENT

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(oauth_token, oauth_token_secret)

api = tweepy.API(auth)


name_post_file = "rumor_detection_acl2017/tweet_features.txt"

attributes = ['id', 'text', 'created_at']

with open(name_post_file, 'w') as f:
    f.write(";".join(attributes)+"\n")

def get_posts_content_batch(start_id = 0, end_id = len(all_posts)):
    step = 100
    for i in tqdm(range(start_id, end_id, step)):
        current_time = time.time()
        
        post_ids = [all_posts[i+k] for k in range(min(step, end_id-i))]
        post_objs = api.statuses_lookup(post_ids)
        
        output = []
        for post_obj in post_objs:
            post_attributes = [str(post_obj.__getattribute__(attr)).replace('\n', '   ').replace(';', ',') 
                               for attr in attributes]
            output.append(";".join(post_attributes)+"\n")
        
        
        output.sort()
        
        with open(name_post_file, 'a') as f:
            f.writelines(output)
            
        if time.time()-current_time < 5:
            time.sleep(5.05 - (time.time()-current_time))          



get_posts_content_batch()

#### GET USERS' CONTENT

name_user_file = "rumor_detection_acl2017/user_features.txt"

attributes = ['id', 'created_at', 'description', 'favourites_count', 'followers_count', 'friends_count', 
'geo_enabled', 'listed_count', 'location', 'name', 'screen_name', 'statuses_count', 'verified']

with open(name_user_file, 'w') as f:
    f.write(";".join(attributes)+"\n")

def get_users_content_batch(start_id = 0, end_id = len(all_users)):
    step = 100
    
    for i in tqdm(range(start_id, end_id, step)):
        current_time = time.time()
        
        user_ids = [all_users[i+k] for k in range(min(step, end_id-i))]
        user_objs = api.lookup_users(user_ids)
                
        output = []
        for user_obj in user_objs:
            user_attributes = [str(user_obj.__getattribute__(attr)).replace('\n', '   ').replace(';', ',') 
                               for attr in attributes]
            output.append(";".join(user_attributes)+"\n")
        
        output.sort()
        
        with open(name_user_file, 'a') as f:
            f.writelines(output)
            
        if time.time()-current_time < 5:
            time.sleep(5.05 - (time.time()-current_time))            


get_users_content_batch()

