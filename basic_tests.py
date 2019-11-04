### SCRIPT USED FOR SOME VERY BASIC TESTS TO CHECK THAT OUR CODE WORKS.
### THE FUNCTIONS PRINT THE CURRENT STATE OF THE ALGORITHM, SOME CROSS VALIDATION
### WITH TWITTER API IS OFTEN NECESSARY TO CHECK IF THE OUTPUTS ARE CORRECT.

### WE DON'T USE PYTEST AND/OR ASSERTIONS AS THE CODE/FEATURES ORDER MAY
### STILL CHANGE A LOT

import random


def test_user_preprocessed_features(preprocessed_user_fts):

    print("test_user_preprocessed_features output")
    for _ in range(10):
        key = random.choice(list(preprocessed_user_fts))
        print(key, preprocessed_user_fts[key])

    print(2211151549, preprocessed_user_fts[2211151549])
    # expecting something like, with non normalized features:
    # favourites_count 2.8600e+03
    # followers_count 3.5400e+02
    # friends_count 8.0500e+02
    # geo_enabled 1
    # has_description 1
    # len_name 13
    # len_screen_name 12
    # listed_count 27
    # statuses_count 137+
    # verified 0
    # (data extracted from twitter api)
    # May have changed a little bit with time.

    input("Results just above, press anything to continue running.")


