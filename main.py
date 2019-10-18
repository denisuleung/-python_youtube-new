import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import seaborn as sns


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

    def main(self):
        self.get_csv_path_array()
        self.import_n_concat_to_1_df()


excel = ExcelIO()
excel.main()


class DataMiner:
    def __init__(self, df):
        self.df = df

    def get_trending_date_time(self):
        self.df['trending_year'] = self.df['trending_date'].apply(lambda x: x[0:2])
        self.df['trending_month'] = self.df['trending_date'].apply(lambda x: x[6:])
        self.df['trending_day'] = self.df['trending_date'].apply(lambda x: x[3:5])

    def get_title_content(self):
        self.df['title_length'] = self.df['title'].apply(lambda x: len(x))
        self.df['title_captain_letter_ratio'] = self.df['title'].apply(lambda x: sum(1 for c in x if c.isupper())/len(x.replace(" ", "")))
        self.df['title_contain_mark'] = self.df['title'].apply(lambda x: 'True' if re.compile('[?!]').search(x) else 'False')

    def get_publish_date_time(self):
        self.df['publish_year'] = self.df['publish_time'].apply(lambda x: x[0:4])
        self.df['publish_month'] = self.df['publish_time'].apply(lambda x: x[5:7])
        self.df['publish_day'] = self.df['publish_time'].apply(lambda x: x[8:10])
        self.df['publish_hour'] = self.df['publish_time'].apply(lambda x: x[11:13])
        self.df['publish_minute'] = self.df['publish_time'].apply(lambda x: x[14:16])
        self.df['publish_second'] = self.df['publish_time'].apply(lambda x: x[17:19])

    def get_tag_details(self):
        self.df['no_of_tag'] = self.df['tags'].apply(lambda x: len(x.split("|")))

    def get_description_length(self):
        self.df['length_of_description'] = self.df['description'].apply(lambda x: len(str(x)))

    def main(self):
        self.get_trending_date_time()
        self.get_title_content()
        self.get_publish_date_time()
        self.get_tag_details()
        self.get_description_length()


data_mined = DataMiner(excel.df)
data_mined.main()

# Draft:
# 1) Value come from views mainly (OK)
# 2) Views may be from
#   a) 1st time view: title (OK), channel_title, region (OK)
#   b) 2nd time view: tag details (OK), amount of comments, region, trending_date


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
        # self.get_chart_title_length_to_views()
        # self.get_chart_region_to_views()
        # self.get_no_of_tag_to_views()
        self.get_comment_count_to_views()


chart = DescriptiveChart(data_mined.df)
chart.main()
