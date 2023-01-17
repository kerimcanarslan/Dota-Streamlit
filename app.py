import pandas as pd
import joblib
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_extras.no_default_selectbox import selectbox



from sklearn.base import BaseEstimator, TransformerMixin
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]




col1, col2, col3, col4, col5 = st.columns([0.6, 1, 0.6, 1, 0.6], gap="small")

with col2:
    hero=pd.read_csv("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/hero_list.csv")
    radiant_1 = st.selectbox("Radiant Takımı 1.Hero Seçimi", hero)
    radiant_2 = st.selectbox("Radiant Takımı 2.Hero Seçimi", hero)
    radiant_3 = st.selectbox("Radiant Takımı 3.Hero Seçimi", hero)
    radiant_4 = st.selectbox("Radiant Takımı 4.Hero Seçimi", hero)
    radiant_5 = st.selectbox("Radiant Takımı 5.Hero Seçimi", hero)
    

with col1:
    image_r_1 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{radiant_1}.png')
    st.image(image_r_1, width=75, use_column_width=True, channels='RGB')
    
    image_r_2 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{radiant_2}.png')
    st.image(image_r_2, width=75, use_column_width=True, channels='RGB')
    
    image_r_3 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{radiant_3}.png')
    st.image(image_r_3, width=75, use_column_width=True, channels='RGB')
    
    image_r_4 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{radiant_4}.png')
    st.image(image_r_4, width=75, use_column_width=True, channels='RGB')
    
    image_r_5 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{radiant_5}.png')
    st.image(image_r_5, width=75, use_column_width=True, channels='RGB')

with col4:
    dire_1 = st.selectbox("Dire Takımı 1.Hero Seçimi", hero)
    dire_2 = st.selectbox("Dire Takımı 2.Hero Seçimi", hero)
    dire_3 = st.selectbox("Dire Takımı 3.Hero Seçimi", hero)
    dire_4 = st.selectbox("Dire Takımı 4.Hero Seçimi", hero)
    dire_5 = st.selectbox("Dire Takımı 5.Hero Seçimi", hero)


with col5:
    image_d_1 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{dire_1}.png')
    st.image(image_d_1, width=75, use_column_width=True, channels='RGB')
    
    image_d_2 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{dire_2}.png')
    st.image(image_d_2, width=75, use_column_width=True, channels='RGB')
    
    image_d_3 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{dire_3}.png')
    st.image(image_d_3, width=75, use_column_width=True, channels='RGB')
    
    image_d_4 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{dire_4}.png')
    st.image(image_d_4, width=75, use_column_width=True, channels='RGB')
    
    image_d_5 = Image.open(f'/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_2_hero_resim/{dire_5}.png')
    st.image(image_d_5, width=75, use_column_width=True, channels='RGB')



with col3:

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe
    

    data_hero_roles = pd.read_csv("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/data_hero_roles.csv")

    feature_to_use_ = pd.read_csv("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/Oyun_Oncesi_Feature_Names.csv")

    df = pd.read_csv("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/df.csv")

    model = joblib.load("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/dota_oyun_oncesi.pkl")

    player_heroes_oyun_oncesi = ['Abaddon', 'Alchemist', 'Ancient Apparition', 'Anti-Mage', 'Axe',
        'Bane', 'Batrider', 'Beastmaster', 'Bloodseeker', 'Bounty Hunter',
        'Brewmaster', 'Bristleback', 'Broodmother', 'Centaur Warrunner',
        'Chaos Knight', 'Chen', 'Clinkz', 'Clockwerk', 'Crystal Maiden',
        'Dark Seer', 'Dazzle', 'Death Prophet', 'Disruptor', 'Doom',
        'Dragon Knight', 'Drow Ranger', 'Earth Spirit', 'Earthshaker',
        'Elder Titan', 'Ember Spirit', 'Enchantress', 'Enigma',
        'Faceless Void', 'Gyrocopter', 'Huskar', 'Invoker', 'Io', 'Jakiro',
        'Juggernaut', 'Keeper of the Light', 'Kunkka', 'Legion Commander',
        'Leshrac', 'Lich', 'Lifestealer', 'Lina', 'Lion', 'Lone Druid',
        'Luna', 'Lycan', 'Magnus', 'Medusa', 'Meepo', 'Mirana',
        'Morphling', 'Naga Siren', "Nature's Prophet", 'Necrophos',
        'Night Stalker', 'Nyx Assassin', 'Ogre Magi', 'Omniknight',
        'Oracle', 'Outworld Devourer', 'Phantom Assassin',
        'Phantom Lancer', 'Phoenix', 'Puck', 'Pudge', 'Pugna',
        'Queen of Pain', 'Razor', 'Riki', 'Rubick', 'Sand King',
        'Shadow Demon', 'Shadow Fiend', 'Shadow Shaman', 'Silencer',
        'Skywrath Mage', 'Slardar', 'Slark', 'Sniper', 'Spectre',
        'Spirit Breaker', 'Storm Spirit', 'Sven', 'Techies',
        'Templar Assassin', 'Terrorblade', 'Tidehunter', 'Timbersaw',
        'Tinker', 'Tiny', 'Treant Protector', 'Troll Warlord', 'Tusk',
        'Undying', 'Ursa', 'Vengeful Spirit', 'Venomancer', 'Viper',
        'Visage', 'Warlock', 'Weaver', 'Windranger', 'Winter Wyvern',
        'Witch Doctor', 'Wraith King', 'Zeus']
   
    h_roles_oyun_oncesi = ['Carries', 'Supports', 'Nukers', 'Disablers', 'Junglers', 'Durable', 'Escape', 'Pushers','Initiators']
    h_specs_oyun_oncesi = ['Strength', 'Agility', 'Intelligence', 'Health_avg', 'Health_reg_avg', 'Mana_avg', 'Mana_reg_avg', 'Armor_avg', 'Att/sec_avg', 'Damage_avg', 'Magic resistance', 'Movement speed', 'Attack speed', 'Turn rate', 'Attack range', 'Damage block']

    radiant_cols_oyun_oncesi = ['radiant_Abaddon',
     'radiant_Alchemist',
     'radiant_Ancient Apparition',
     'radiant_Anti-Mage',
     'radiant_Axe',
     'radiant_Bane',
     'radiant_Batrider',
     'radiant_Beastmaster',
     'radiant_Bloodseeker',
     'radiant_Bounty Hunter',
     'radiant_Brewmaster',
     'radiant_Bristleback',
     'radiant_Broodmother',
     'radiant_Centaur Warrunner',
     'radiant_Chaos Knight',
     'radiant_Chen',
     'radiant_Clinkz',
     'radiant_Clockwerk',
     'radiant_Crystal Maiden',
     'radiant_Dark Seer',
     'radiant_Dazzle',
     'radiant_Death Prophet',
     'radiant_Disruptor',
     'radiant_Doom',
     'radiant_Dragon Knight',
     'radiant_Drow Ranger',
     'radiant_Earth Spirit',
     'radiant_Earthshaker',
     'radiant_Elder Titan',
     'radiant_Ember Spirit',
     'radiant_Enchantress',
     'radiant_Enigma',
     'radiant_Faceless Void',
     'radiant_Gyrocopter',
     'radiant_Huskar',
     'radiant_Invoker',
     'radiant_Io',
     'radiant_Jakiro',
     'radiant_Juggernaut',
     'radiant_Keeper of the Light',
     'radiant_Kunkka',
     'radiant_Legion Commander',
     'radiant_Leshrac',
     'radiant_Lich',
     'radiant_Lifestealer',
     'radiant_Lina',
     'radiant_Lion',
     'radiant_Lone Druid',
     'radiant_Luna',
     'radiant_Lycan',
     'radiant_Magnus',
     'radiant_Medusa',
     'radiant_Meepo',
     'radiant_Mirana',
     'radiant_Morphling',
     'radiant_Naga Siren',
     "radiant_Nature's Prophet",
     'radiant_Necrophos',
     'radiant_Night Stalker',
     'radiant_Nyx Assassin',
     'radiant_Ogre Magi',
     'radiant_Omniknight',
     'radiant_Oracle',
     'radiant_Outworld Devourer',
     'radiant_Phantom Assassin',
     'radiant_Phantom Lancer',
     'radiant_Phoenix',
     'radiant_Puck',
     'radiant_Pudge',
     'radiant_Pugna',
     'radiant_Queen of Pain',
     'radiant_Razor',
     'radiant_Riki',
     'radiant_Rubick',
     'radiant_Sand King',
     'radiant_Shadow Demon',
     'radiant_Shadow Fiend',
     'radiant_Shadow Shaman',
     'radiant_Silencer',
     'radiant_Skywrath Mage',
     'radiant_Slardar',
     'radiant_Slark',
     'radiant_Sniper',
     'radiant_Spectre',
     'radiant_Spirit Breaker',
     'radiant_Storm Spirit',
     'radiant_Sven',
     'radiant_Techies',
     'radiant_Templar Assassin',
     'radiant_Terrorblade',
     'radiant_Tidehunter',
     'radiant_Timbersaw',
     'radiant_Tinker',
     'radiant_Tiny',
     'radiant_Treant Protector',
     'radiant_Troll Warlord',
     'radiant_Tusk',
     'radiant_Undying',
     'radiant_Ursa',
     'radiant_Vengeful Spirit',
     'radiant_Venomancer',
     'radiant_Viper',
     'radiant_Visage',
     'radiant_Warlock',
     'radiant_Weaver',
     'radiant_Windranger',
     'radiant_Winter Wyvern',
     'radiant_Witch Doctor',
     'radiant_Wraith King',
     'radiant_Zeus']

    dire_cols_oyun_oncesi = ['dire_Abaddon',
     'dire_Alchemist',
     'dire_Ancient Apparition',
     'dire_Anti-Mage',
     'dire_Axe',
     'dire_Bane',
     'dire_Batrider',
     'dire_Beastmaster',
     'dire_Bloodseeker',
     'dire_Bounty Hunter',
     'dire_Brewmaster',
     'dire_Bristleback',
     'dire_Broodmother',
     'dire_Centaur Warrunner',
     'dire_Chaos Knight',
     'dire_Chen',
     'dire_Clinkz',
     'dire_Clockwerk',
     'dire_Crystal Maiden',
     'dire_Dark Seer',
     'dire_Dazzle',
     'dire_Death Prophet',
     'dire_Disruptor',
     'dire_Doom',
     'dire_Dragon Knight',
     'dire_Drow Ranger',
     'dire_Earth Spirit',
     'dire_Earthshaker',
     'dire_Elder Titan',
     'dire_Ember Spirit',
     'dire_Enchantress',
     'dire_Enigma',
     'dire_Faceless Void',
     'dire_Gyrocopter',
     'dire_Huskar',
     'dire_Invoker',
     'dire_Io',
     'dire_Jakiro',
     'dire_Juggernaut',
     'dire_Keeper of the Light',
     'dire_Kunkka',
     'dire_Legion Commander',
     'dire_Leshrac',
     'dire_Lich',
     'dire_Lifestealer',
     'dire_Lina',
     'dire_Lion',
     'dire_Lone Druid',
     'dire_Luna',
     'dire_Lycan',
     'dire_Magnus',
     'dire_Medusa',
     'dire_Meepo',
     'dire_Mirana',
     'dire_Morphling',
     'dire_Naga Siren',
     "dire_Nature's Prophet",
     'dire_Necrophos',
     'dire_Night Stalker',
     'dire_Nyx Assassin',
     'dire_Ogre Magi',
     'dire_Omniknight',
     'dire_Oracle',
     'dire_Outworld Devourer',
     'dire_Phantom Assassin',
     'dire_Phantom Lancer',
     'dire_Phoenix',
     'dire_Puck',
     'dire_Pudge',
     'dire_Pugna',
     'dire_Queen of Pain',
     'dire_Razor',
     'dire_Riki',
     'dire_Rubick',
     'dire_Sand King',
     'dire_Shadow Demon',
     'dire_Shadow Fiend',
     'dire_Shadow Shaman',
     'dire_Silencer',
     'dire_Skywrath Mage',
     'dire_Slardar',
     'dire_Slark',
     'dire_Sniper',
     'dire_Spectre',
     'dire_Spirit Breaker',
     'dire_Storm Spirit',
     'dire_Sven',
     'dire_Techies',
     'dire_Templar Assassin',
     'dire_Terrorblade',
     'dire_Tidehunter',
     'dire_Timbersaw',
     'dire_Tinker',
     'dire_Tiny',
     'dire_Treant Protector',
     'dire_Troll Warlord',
     'dire_Tusk',
     'dire_Undying',
     'dire_Ursa',
     'dire_Vengeful Spirit',
     'dire_Venomancer',
     'dire_Viper',
     'dire_Visage',
     'dire_Warlock',
     'dire_Weaver',
     'dire_Windranger',
     'dire_Winter Wyvern',
     'dire_Witch Doctor',
     'dire_Wraith King',
     'dire_Zeus']
     
    radiant_roles_oyun_oncesi = ['radiant_Carries',
     'radiant_Supports',
     'radiant_Nukers',
     'radiant_Disablers',
     'radiant_Junglers',
     'radiant_Durable',
     'radiant_Escape',
     'radiant_Pushers',
     'radiant_Initiators']

    dire_roles_oyun_oncesi = ['dire_Carries',
     'dire_Supports',
     'dire_Nukers',
     'dire_Disablers',
     'dire_Junglers',
     'dire_Durable',
     'dire_Escape',
     'dire_Pushers',
     'dire_Initiators']

    radiant_h_specs_oyun_oncesi = ['radiant_Strength',
     'radiant_Agility',
     'radiant_Intelligence',
     'radiant_Health_avg',
     'radiant_Health_reg_avg',
     'radiant_Mana_avg',
     'radiant_Mana_reg_avg',
     'radiant_Armor_avg',
     'radiant_Att/sec_avg',
     'radiant_Damage_avg',
     'radiant_Magic resistance',
     'radiant_Movement speed',
     'radiant_Attack speed',
     'radiant_Turn rate',
     'radiant_Attack range',
     'radiant_Damage block']

    dire_h_specs_oyun_oncesi = ['dire_Strength',
     'dire_Agility',
     'dire_Intelligence',
     'dire_Health_avg',
     'dire_Health_reg_avg',
     'dire_Mana_avg',
     'dire_Mana_reg_avg',
     'dire_Armor_avg',
     'dire_Att/sec_avg',
     'dire_Damage_avg',
     'dire_Magic resistance',
     'dire_Movement speed',
     'dire_Attack speed',
     'dire_Turn rate',
     'dire_Attack range',
     'dire_Damage block']

    def prepare_dataframe(heroes_radiant, heroes_dire, feature_names):
        # Heroes, roles, specs
        X = None

        radiant_heroes = []
        dire_heroes = []
        radiant_roles_list = []
        dire_roles_list = []
        radiant_specs = []
        dire_specs = []
        all_heroes= []

        all_heroes.extend(heroes_radiant)
        all_heroes.extend(heroes_dire)

        new_hero_df = pd.DataFrame(all_heroes, columns = ["hero"])
        new_hero_df = pd.merge(new_hero_df, df, on ='hero', how ='left')

        for col in data_hero_roles.columns:
            new_hero_df[col] = new_hero_df['hero'].apply(lambda _h_name: 1 if _h_name in data_hero_roles[col].values else 0)
            
        new_hero_df = one_hot_encoder(new_hero_df, ["hero"], drop_first=False)
        new_hero_df.columns = [col.replace("hero_", "") for col in new_hero_df.columns]

        radi_hero = pd.DataFrame([0]*110, index=player_heroes_oyun_oncesi).T
        
        for col in heroes_radiant:
            radi_hero.loc[:,col] = 1

        dire_hero = pd.DataFrame([0]*110, index=player_heroes_oyun_oncesi).T
        
        for col in heroes_dire:
            dire_hero.loc[:,col] = 1


        radiant_heroes.append(radi_hero.values)
        dire_heroes.append(dire_hero.values)
        radiant_roles_list.append(new_hero_df[h_roles_oyun_oncesi][:5].sum().values)
        dire_roles_list.append(new_hero_df[h_roles_oyun_oncesi][5:].sum().values)
        radiant_specs.append(new_hero_df[h_specs_oyun_oncesi][:5].sum().values)
        dire_specs.append(new_hero_df[h_specs_oyun_oncesi][5:].sum().values)

        radiant_heroes = pd.DataFrame(np.array(radiant_heroes).reshape(1,-1), columns=radiant_cols_oyun_oncesi).applymap(lambda x: 1 if x > 0 else 0)
        dire_heroes = pd.DataFrame(np.array(dire_heroes).reshape(1,-1), columns=dire_cols_oyun_oncesi).applymap(lambda x: 1 if x > 0 else 0)
        radiant_roles_list = pd.DataFrame(radiant_roles_list, columns=radiant_roles_oyun_oncesi)
        dire_roles_list = pd.DataFrame(dire_roles_list, columns=dire_roles_oyun_oncesi)
        radiant_specs = pd.DataFrame(radiant_specs, columns=radiant_h_specs_oyun_oncesi)
        dire_specs = pd.DataFrame(dire_specs, columns=dire_h_specs_oyun_oncesi)

        X = pd.concat([radiant_heroes, dire_heroes, radiant_roles_list, dire_roles_list, radiant_specs, dire_specs], axis=1)
        return X[feature_names["Feature_Name"].values]
    
    heroes_radiant=[radiant_1, radiant_2, radiant_3, radiant_4, radiant_5]
    heroes_dire=[dire_1, dire_2, dire_3, dire_4, dire_5]

    sample_ = prepare_dataframe(heroes_radiant, heroes_dire, feature_to_use_)
    win_probility = model.predict_proba(sample_)[:, 1] * 100
    win_ = model.predict(sample_)
    win_probility = win_probility.round(2)
    
    if win_==1:
        st.markdown(f'<h1 <p><font face="tahoma" size="1.9" color="white"><b>Radiant Win: {win_}</b></font></p> </h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 <p><font face="tahoma" size="4.5" color="lime"><b>Probability of Win: %{win_probility}</b></font></p> </h1>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h1 <p><font face="tahoma" size="1.9" color="white"><b>Radiant Win: {win_}</b></font></p> </h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 <p><font face="tahoma" size="4.5" color="maroon"><b>Probability of Win: %{win_probility}</b></font></p> </h1>', unsafe_allow_html=True)
    
wlb_df = pd.read_csv("/Users/kerimcanarslan/PycharmProjects/pythonProject/PycharmProjects/Bitirme Projesi/wlb_hero_final.csv")

wlb_df = wlb_df.set_index('index')
wlb_df = wlb_df.iloc[:,1:]
def recommend_dota_heroes(win_loss_rates_wlb=[], hero_to_counter=[], enemy_team=[], our_team=[],
                          number_of_recommendations=5):
    if hero_to_counter not in wlb_df.columns:
        print("You entered wrong hero name")
    recommend = win_loss_rates_wlb.loc[hero_to_counter, :].dropna()
    all_hero = enemy_team
    all_hero.extend(our_team)
    return recommend[set(recommend.index).difference(set(all_hero))].sort_values()[:number_of_recommendations]


abc = st.selectbox("Hero Seçiniz", hero)
hero_recommender = recommend_dota_heroes(wlb_df, abc, heroes_dire, heroes_radiant, 10)
st.dataframe(hero_recommender)
        
    








