import requests

url  = 'http://localhost:5000/predict_api'

s = requests.post(url,json={'release_year': 2020,
'acousticness': 0.1,
'speechiness':2,
'hist_user_behavior_n_seekfwd': 0,
'hist_user_behavior_reason_start': 6,
'hist_user_behavior_reason_end': 5,
'duration':190,
'hour_of_day': 15,
'session_position':6,
'session_length': 16,
'context_type': 3,
'long_pause_before_play':0,
'short_pause_before_play':0,
'no_pause_before_play':1,
'not_skipped': 0})

print(s.json())