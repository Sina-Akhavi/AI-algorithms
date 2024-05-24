import pandas as pd

df = pd.read_csv("facebook/Live.csv")

# print(df.describe())
# print("///////////////////////////")
# print(f'columns: ', df.columns)

# valuse of columns

# values = df['status_id']
# print(f'values of status IDs: ', values.unique())

# values = df['status_type']
# print(f'values of status type: ', values.unique())
#
# values = df['status_published']
# print(f'values of status_published: ', values.unique())
#
# values = df['num_reactions']
# print(f'values of num_reactions: ', values.unique())
#
# values = df['num_comments']
# print(f'values of num_comments: ', values.unique())
#
# values = df['num_comments']
# print(f'values of num_comments: ', values.unique())
#
# values = df['num_shares']
# print(f'values of num_shares: ', values.unique())
#
# values = df['num_likes']
# print(f'values of num_likes: ', values.unique())
#
# values = df['Column1']
# print(f'values Column1: ', values.unique())
#
# values = df['Column2']
# print(f'values Column2: ', values.unique())
#
# values = df['Column3']
# print(f'values Column3: ', values.unique())


df = df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1)
print(f'columns: ', df.columns)

df = pd.get_dummies(df, columns=['status_type'])
print("After using dummies: \n", df.head())


