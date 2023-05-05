import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

session_data = pd.read_csv('D:\\INTERNSHIPS\\Technocolabs AI Developer Intern\\log_mini.csv')
tf0 = pd.read_csv('D:\\INTERNSHIPS\\Technocolabs AI Developer Intern\\tf_000000000000.csv')
tf1 = pd.read_csv('D:\\INTERNSHIPS\\Technocolabs AI Developer Intern\\tf_000000000001.csv')

track_data = tf0.append(tf1, ignore_index = True)

session_data.rename(columns = {'track_id_clean':'track_id'},inplace = True)
session_track_data = pd.merge(session_data,track_data, on = 'track_id',how = 'left')

session_track_data.drop('track_id',axis = 1, inplace = True)
session_track_data.set_index('session_id')

session_track_data['skipped'] = session_track_data['skip_1'] + session_track_data['skip_2'] + session_track_data['skip_3']
session_track_data_copy = session_track_data.copy()

session_track_data_copy.drop(['skip_1','skip_2','skip_3'],axis = 1,inplace=True)

s = (session_track_data_copy.dtypes == 'object')
object_cols = list(s[s].index)

print("Object variables in the dataset:", object_cols)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in object_cols:
    session_track_data_copy[i]= session_track_data_copy[[i]].apply(LE.fit_transform)

bool_col = (session_track_data_copy.dtypes == 'bool')
cols = list(bool_col[bool_col].index)

print("Bool variables in the dataset:", cols)

for i in cols:
    session_track_data_copy[i] = session_track_data_copy[i].astype(int)


X = session_track_data_copy[['release_year','acousticness','speechiness','hist_user_behavior_n_seekfwd','duration','long_pause_before_play','context_type','no_pause_before_play','session_length','hour_of_day','session_position','short_pause_before_play','hist_user_behavior_reason_start','hist_user_behavior_reason_end','not_skipped']]
Y = session_track_data_copy[['skipped']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
clf=RandomForestClassifier()
clf.fit(x_train, y_train)

#preds = clf.predict(x_test)

pickle.dump(clf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

