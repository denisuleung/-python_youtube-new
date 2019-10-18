from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ExcelIO:
    def __init__(self, folder_path=None, file_lst=None, df=None):
        self.folder_path = folder_path
        self.file_lst = file_lst
        self.df = df  # It is better to create indicator to prevent importation repeatedly.

    def get_csv_path_array(self):
        self.folder_path = os.getcwd()
        self.file_lst = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f)) and f[-4:] == '.csv']

    def import_n_concat_to_1_df(self):
        self.df, tmp_data = pd.read_csv(self.file_lst[0]), []
        self.df.insert(loc=0, column="Region", value=self.file_lst[0][0:2])
        print(self.file_lst[0] + ' has been imported')
        for i in range(1, len(self.file_lst)):
            tmp_data = pd.read_csv(self.file_lst[i], encoding="ISO-8859-1")
            tmp_data.insert(loc=0, column="Region", value=self.file_lst[i][0:2])
            self.df = pd.concat([tmp_data, self.df])
            print(self.file_lst[i] + ' has been imported')
            # maybe we can try to add region data in data frame

    # def export_csv(self, df):
    #     df.to_csv("export.csv")

    def main(self):
        self.get_csv_path_array()
        self.import_n_concat_to_1_df()


excel = ExcelIO()
excel.main()


class DataMiner:
    def __init__(self, df):
        self.df = df

    def get_trending_date_time(self):
        self.df['trending_year'] = self.df['trending_date'].apply(lambda x: '20'+x[0:2]).astype(int)
        self.df['trending_month'] = self.df['trending_date'].apply(lambda x: x[6:]).astype(int)
        self.df['trending_day'] = self.df['trending_date'].apply(lambda x: x[3:5]).astype(int)

    def get_title_content(self):
        self.df['title_length'] = self.df['title'].apply(lambda x: len(x))
        self.df['title_captain_letter_ratio'] = self.df['title'].apply(lambda x: sum(1 for c in x if c.isupper())/len(x.replace(" ", "")))
        self.df['title_contain_mark'] = self.df['title'].apply(lambda x: 'True' if re.compile('[?!]').search(x) else 'False')

    def get_publish_date_time(self):
        self.df['publish_year'] = self.df['publish_time'].apply(lambda x: x[0:4]).astype(int)
        self.df['publish_month'] = self.df['publish_time'].apply(lambda x: x[5:7]).astype(int)
        self.df['publish_day'] = self.df['publish_time'].apply(lambda x: x[8:10]).astype(int)
        self.df['publish_hour'] = self.df['publish_time'].apply(lambda x: x[11:13]).astype(int)
        self.df['publish_minute'] = self.df['publish_time'].apply(lambda x: x[14:16]).astype(int)
        self.df['publish_second'] = self.df['publish_time'].apply(lambda x: x[17:19]).astype(int)

    def get_video_online_days(self, ty, tm, td, puy, pum, pud):
        # To prevent publish day and trending days are same, add 0.5 on it.
        return (date(ty, tm, td) - date(puy, pum, pud)).days + 0.5

    def get_tag_details(self):
        self.df['no_of_tag'] = self.df['tags'].apply(lambda x: len(x.split("|")))

    def get_description_length(self):
        self.df['length_of_description'] = self.df['description'].apply(lambda x: len(str(x)))

    def get_video_online_day(self):
        self.df['video_online_days'] = np.vectorize(self.get_video_online_days)\
            (self.df['trending_year'], self.df['trending_month'], self.df['trending_day'],
             self.df['publish_year'], self.df['publish_month'], self.df['publish_day'])

    def get_views_per_day(self):
        self.df['views_per_day'] = self.df['views'] / self.df['video_online_days']

    def remove_duplicate_video_id(self):
        self.df = self.df.sort_values(['video_id', 'video_online_days'], ascending=[0, 0]).groupby('video_id').head(1)

    def main(self):
        self.get_trending_date_time()
        self.get_title_content()
        self.get_publish_date_time()
        self.get_tag_details()
        self.get_description_length()
        self.get_video_online_day()
        self.get_views_per_day()
        self.remove_duplicate_video_id()


data_mined = DataMiner(excel.df)
data_mined.main()

# Draft:
# 1) Value come from views mainly (OK)
# 2) Views may be from
#   a) 1st time view: title (OK), channel_title, region (OK)
#   b) 2nd time view: tag details (OK), amount of comments (OK), region(OK), trending_date


class DescriptiveChart:
    def __init__(self, df, graph_title_length=None, graph_region=None, graph_no_of_tag=None, graph_comment_count=None):
        self.df = df
        self.graph_title_length = graph_title_length
        self.graph_region = graph_region
        self.graph_no_of_tag = graph_no_of_tag
        self.graph_comment_count = graph_comment_count

    def get_chart_title_length_to_views(self):
        plt.figure(figsize=(8, 8))
        self.graph_title_length = sns.scatterplot(x="title_length", y="views", s=10, data=self.df, color='green')
        self.graph_title_length.set_xlabel(xlabel="Length of Title")
        self.graph_title_length.set_ylabel(ylabel="Views Count")
        self.graph_title_length.set_title(label="Title Characters Against Views", fontdict={'size': 20, 'color': 'darkred', 'weight': 'bold'})
        self.graph_title_length.figure.text(0.6, 0.8, "To get the higher views count (i.e. More than 1000000),\nTitle should be within 100 characters", ha='left', fontsize=8)
        self.graph_title_length.figure.get_axes()[0].set_yscale('log')
        # plt.savefig("title_length_to_views.png", dpi=400)

    def get_chart_region_to_views(self):
        plt.figure(figsize=(8, 8))
        self.graph_region = sns.boxplot(x='Region', y='views', data=self.df)
        self.graph_region.set_xlabel(xlabel="Region")
        self.graph_region.set_ylabel(ylabel="Views Count")
        self.graph_region.set_title(label="Region Against Views", fontdict={'size': 20, 'color': 'darkred', 'weight': 'bold'})
        self.graph_region.figure.text(0.6, 0.85, "US and GB video have higher views count,\nIt may relate to English is the most using language in the world", ha='left', fontsize=8)
        self.graph_region.figure.get_axes()[0].set_yscale('log')
        # plt.savefig("region_to_views.png", dpi=400)

    def get_no_of_tag_to_views(self):
        plt.figure(figsize=(8, 8))
        self.graph_no_of_tag = sns.scatterplot(x="no_of_tag", y="views", s=10, data=self.df, color='red')
        self.graph_no_of_tag.set_xlabel(xlabel="Number of Tags")
        self.graph_no_of_tag.set_ylabel(ylabel="Views Count")
        self.graph_no_of_tag.set_title(label="Number of Tags Against Views", fontdict={'size': 20, 'color': 'darkred', 'weight': 'bold'})
        self.graph_no_of_tag.figure.text(0.6, 0.8, "To get the higher views count (i.e. More than 1000000),\nNumber of Tags should not greater than 60", ha='left', fontsize=8)
        self.graph_no_of_tag.figure.get_axes()[0].set_yscale('log')
        # plt.savefig("no_of_tag_to_views.png", dpi=400)

    def get_comment_count_to_views(self):
        plt.figure(figsize=(8, 8))
        self.graph_comment_count = sns.scatterplot(x="comment_count", y="views", s=10, data=self.df, color='blue')
        self.graph_comment_count.set_xlabel(xlabel="Comment Count")
        self.graph_comment_count.set_ylabel(ylabel="Views Count")
        self.graph_comment_count.set(ylim=(100000, 20000000))
        self.graph_comment_count.set(xlim=(0, 75000))
        self.graph_comment_count.set_title(label="Comment Count Against Views \n[100K < Views < 20M, Comment < 75K]", fontdict={'size': 16, 'color': 'darkred', 'weight': 'bold'})
        self.graph_comment_count.figure.text(0.6, 0.2, "Comment Count and Views are slightly positive correlated", ha='left', fontsize=8)
        self.graph_comment_count.figure.get_axes()[0].set_yscale('log')
        # plt.savefig("no_of_tag_to_views.png", dpi=400)

    def main(self):
        # pass
        self.get_chart_title_length_to_views()
        self.get_chart_region_to_views()
        self.get_no_of_tag_to_views()
        self.get_comment_count_to_views()


chart = DescriptiveChart(data_mined.df)
chart.main()


class Model:
    def __init__(self, df, y=None, x=None,
                 train_x=None, val_x=None, train_y=None, val_y=None, model=None, val_predict=None):
        self.df = df
        self.y, self.x = y, x
        self.train_x, self.val_x = train_x, val_x
        self.train_y, self.val_y = train_y, val_y
        self.model, self.val_predict = model, val_predict

    def filter_useful_col(self):
        self.df = self.df.drop(['video_id', 'trending_date', 'title', 'channel_title', 'publish_time', 'tags', 'thumbnail_link', 'description'], axis=1)

    def do_one_hot_encoding(self):
        self.df = pd.get_dummies(self.df)

    def define_model_parameter(self):
        self.y = self.df['views']
        self.x = self.df.drop(['views'], axis=1)

    def split_train_val(self):
        self.train_x, self.val_x, self.train_y, self.val_y = \
            train_test_split(self.x, self.y, train_size=0.8, test_size=0.2, random_state=0)

    def do_regression(self):
        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        self.model.fit(self.train_x, self.train_y, early_stopping_rounds=5, eval_set=[(self.val_x, self.val_y)], verbose=False)

    def get_mae(self):
        self.val_predict = self.model.predict(self.val_x)
        print("MAE", mean_absolute_error(self.val_y, self.val_predict))

    def get_expected_views(self):
        self.df['Expected_Views'] = self.model.predict(self.x)

    def main(self):
        self.filter_useful_col()
        self.do_one_hot_encoding()
        self.define_model_parameter()
        self.split_train_val()
        self.do_regression()
        self.get_mae()
        self.get_expected_views()


modeled = Model(data_mined.df)
modeled.main()

# excel.export_csv(modeled.df)

# ================================== #

# test = data_mined.df.sort_values(['video_id', 'video_online_days'], ascending=[0, 0])
# test1 = data_mined.df.sort_values(['video_id', 'video_online_days'], ascending=[0, 0]).groupby('video_id').head(1)