import pandas as pd

file_path = "/Users/czu/Desktop/VSC/trysentimentscore.csv"
data = pd.read_csv(file_path)
shop = pd.read_csv('/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/shop.csv')

data['品項分'] = data['品項'] * data['sentiment_score']
data['口味分'] = data['口味'] * data['sentiment_score']
data['服務態度分'] = data['服務態度'] * data['sentiment_score']

data.to_csv('dummy_score.csv')


shop_scores = data.groupby('shop_id').apply(lambda x: pd.Series({
    '品項_score': x['品項分'].sum() / x['品項'].sum() if x['品項'].sum() != 0 else 0,
    '口味_score': x['口味分'].sum() / x['口味'].sum() if x['口味'].sum() != 0 else 0,
    '服務態度_score': x['服務態度分'].sum() / x['服務態度'].sum() if x['服務態度'].sum() != 0 else 0
})).reset_index()
print(shop_scores)


shop = shop.merge(shop_scores, on='shop_id', how='left')
shop['avg_sentiment'] = (shop['品項_score'] + shop['口味_score'] + shop['服務態度_score']) / 3
# print(shop)
shop.to_csv('shop_rating.csv')


