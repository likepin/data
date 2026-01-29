import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

ETTh1 = pd.read_csv('ETTh1.csv')
weather = pd.read_csv('weather.csv')
# fig1=px.line(ETTh1.sort_values(["date"]), x='date', y='OT',title = "ETTh1" )
# fig1.show()




# selected_ETTh1 = ETTh1.iloc[:, 1:8]
# correlation_matrix = selected_ETTh1.corr()
# print(correlation_matrix)
# plt.rcParams['axes.unicode_minus'] = False
# # 绘制热力图
# sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
# # 设置标题
# plt.title("Heatmap of the correlation matrix.")
# # 显示图形
# plt.show()

# a = ETTh1.copy()
# a['date'] = pd.to_datetime(a.date)
# a["year"] = a.date.dt.year
# a["month"] = a.date.dt.month
# fig2 = px.box(a, x="year", y="OT" , color = "month", title = "ETT OT changes")
# fig2.show()

# a = weather.copy()  
# a['OT'] = a['OT'].apply(lambda x: a['OT'].mean() if x <-300 else x)
# fig3=px.line(a.sort_values(["date"]), x='date', y='OT',title = "weather" ) 
# fig3.show()

# a = weather.copy() 
# a['OT'] = a['OT'].apply(lambda x: a['OT'].mean() if x <-300 else x)
# a['date'] = pd.to_datetime(a['date'])
# a.set_index('date', inplace=True)
# daily_mean = a.resample('D').mean()
# daily_mean = daily_mean.reset_index()
# figure4=px.line(daily_mean, x='date', y='OT',title = "weather" )
# figure4.show()


# a = weather.copy()    
# a['OT'] = a['OT'].apply(lambda x: a['OT'].mean() if x <-300 else x)
# a['date'] = pd.to_datetime(a.date)
# a["day"] = a.date.dt.day
# a["hour"] = a.date.dt.hour
# fig5=px.box(a, x="hour", y="OT" , color = "hour", title = "weather OT changes")
# fig5.show()

weather_corr = weather.copy()
weather_corr['OT'] = weather_corr['OT'].mask(weather_corr['OT'] < -300, weather_corr['OT'].mean())
numeric_cols = weather_corr.select_dtypes(include='number').columns
correlation_matrix = weather_corr[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Weather Feature Correlation Heatmap')
plt.tight_layout()
plt.show()