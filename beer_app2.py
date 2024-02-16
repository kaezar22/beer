# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:32:47 2024

@author: ASUS
"""

import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
#D://DATA_analysis//Projects//BEER//app_beer//
df = pd.read_csv('beer_city.csv')

st.image('header.png', use_column_width=True)

def main():
    st.title("Navigation")
    page = st.radio("Select Page", ["Where to drink", "Recommender"])

    if page == "Where to drink":
        where_to_drink()
    elif page == "Recommender":
        recommender()

def where_to_drink():
    # Convert 2-digit country codes to full country names
    def get_country_name(code):
        try:
            return pycountry.countries.get(alpha_2=str(code)).name if code and code == code else "Unknown Country"
        except LookupError:
            print(f"Error looking up country for code: {code}")
            return "Unknown Country"
    
    df['country_name'] = df['country'].apply(get_country_name)
    
    # Section 1: Where are you
    st.title("🍺 Where are you DRINKING?? 🍺")
    
    # First row with two columns
    col1, col2 = st.columns(2)
    
    # Filters for country and city
    with col1:
        country_filter = st.selectbox("Select Country", df['country_name'].sort_values().unique())
    
    with col2:
        city_options = df[df['country_name'] == country_filter]['city'].unique()
        city_filter = st.selectbox("Select City", city_options)
    
    # Filter DataFrame based on user selections
    filtered_df = df[(df['country_name'] == country_filter) & (df['city'] == city_filter)]
    
    # Display filtered data with brewery names and number of beers
    brewery_summary = (
        filtered_df.groupby(['Brewery', 'types'])
        .agg({'Beer Name (Full)': 'count', 'review_overall': 'mean'})
        .rename(columns={'Beer Name (Full)': 'Number of Beers', 'review_overall': 'Average Score'})
        .reset_index()
    )
    
    # Sort values by 'Average Score' in descending order
    brewery_summary = brewery_summary.sort_values(by='Average Score', ascending=False)
    brewery_summary_without_types = brewery_summary.drop(columns=['types'])
    
    # Display the top 10 breweries by default
    show_all_breweries = st.checkbox(f"Show all breweries in {city_filter}", value=False)
    if show_all_breweries:
        st.write(f"All breweries in **{city_filter}**")
        st.dataframe(brewery_summary_without_types)  # Set the desired width here
    else:
        # Display the top 10 breweries
        st.write(f"Top 10 breweries in **{city_filter}** by Average Score")
        brewery_summary_without_types = brewery_summary_without_types.rename(columns={'Brewery': 'Brewery Name'})
        st.dataframe(brewery_summary_without_types.head(10))  # Set the desired width here
    
    col3, col4 = st.columns(2)
   
    with col3:
       st.subheader(f"Style Filter in **{country_filter}**")
   
       # Filter the DataFrame based on the selected country
       styles_in_country = df[df['country_name'] == country_filter]['Style'].unique()
   
       # Let the user choose a beer style available in that country
       selected_style = st.selectbox("Select Beer Style", styles_in_country)
   
       # Filter the DataFrame based on the selected style
       style_filtered_df = df[(df['country_name'] == country_filter) & (df['Style'] == selected_style)].sort_values(by='review_overall', ascending=False).head(5)
   
       # Display the 5 best beers of the selected style
       st.write(f"**Top 5 Beers in {country_filter} of Style: {selected_style}**")
   
       # Sort the filtered DataFrame by 'review_overall' in descending order and get the top 5
       top_style_beers = style_filtered_df.sort_values(by='review_overall', ascending=False)[['Beer Name (Full)', 'review_overall', 'Brewery']]
       
       # Rename columns for better display
       top_style_beers = top_style_beers.rename(columns={'Beer Name (Full)': 'Beer Name', 'review_overall': 'Overall Rating', 'Brewery': 'Brewery Name'})
   
       # Show the top 5 beers of the selected style
       st.dataframe(top_style_beers)  # Set the desired width here
   
    with col4:
       st.subheader(f"Top 3 Most Popular Styles in {country_filter}")
   
       # Add a bar chart of the 3 most popular styles in the country
       top_styles_chart_data = df[df['country_name'] == country_filter]['Style'].value_counts().nlargest(3)
       st.bar_chart(top_styles_chart_data)
   
   # Third row with two columns
    col5, col6 = st.columns(2)
   
    with col5:
       st.subheader(f"Best Beers in **{country_filter}**")
   
       # Sort the DataFrame by 'review_overall' in descending order and get the top 5
       top_beers = filtered_df.sort_values(by='review_overall', ascending=False).head(8)[['Beer Name (Full)', 'review_overall', 'Brewery']]
       
       # Rename columns for better display
       top_beers = top_beers.rename(columns={'Beer Name (Full)': 'Beer Name', 'review_overall': 'Overall Rating', 'Brewery': 'Brewery Name'})
   
       # Show the top 5 beers in the country
       st.dataframe(top_beers)  # Set the desired width here
   
       # Create a variable for the top 5 beers
       top_beers_for_plot = top_beers[['Beer Name', 'Overall Rating']]
   
   # Sixth column: Plot a horizontal bar chart of the overall ratings
    with col6:
       # Plot a horizontal bar chart of the overall ratings
       fig = px.bar(top_beers_for_plot, x='Overall Rating', y='Beer Name', orientation='h', text='Overall Rating')
       fig.update_layout(showlegend=False)
       fig.update_traces(marker=dict(color='skyblue'), selector=dict(type='bar'))
   
       st.plotly_chart(fig)

def recommender():
    # Function to search for beers based on partial name
        df2 = pd.read_csv('beer_city.csv')
        def search_beer_partial(query):
            st.header("Beer Selector Section")
    
            # Allow the user to input the threshold
            threshold = st.slider("Select Similarity Threshold:", min_value=0, max_value=100, value=70)
    
            if not query:
                st.warning("Please provide a non-empty query.")
                return None
    
            matches = process.extract(query, df['Beer Name (Full)'], limit=None)
            filtered_matches = [match for match in matches if match[1] >= threshold]
    
            if not filtered_matches:
                st.info(f"No beers found with a similarity above {threshold}% for '{query}'.")
                return None
    
            matching_beers = df2.iloc[[match[2] for match in filtered_matches]].copy()
            matching_beers['Similarity Score'] = [match[1] for match in filtered_matches]
    
            # Display matching beers without similarity index
            #st.subheader("Matching Beers:")
            #st.table(matching_beers[['Beer Name (Full)', 'Similarity Score']])
    
            # Allow the user to select one beer
            selected_beer = st.selectbox("Select a beer:", matching_beers['Beer Name (Full)'].tolist())
            return selected_beer
    
        # Function to recommend beers based on selected beer
        def recommend_beer(selected_beer, top_n=5):
            st.header("Recommender Section")
    
            # Check if the beer name exists in the DataFrame
            if selected_beer not in df['Beer Name (Full)'].values:
                st.warning(f"Beer '{selected_beer}' not found in the dataset.")
                return None
    
            # Retrieve the index of the specified beer
            beer_index = df2[df2['Beer Name (Full)'] == selected_beer].index[0]
    
            # Compute cosine similarity
            cosine_sim = cosine_similarity(df2.select_dtypes(include=['float64']), df2.select_dtypes(include=['float64']))
    
            # Get similarity scores for the specified beer
            sim_scores = list(enumerate(cosine_sim[beer_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
            # Get the indices of the top similar beers
            top_similar_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
            # Create a DataFrame with recommended beers and their similarity index
            recommended_beers = df2.iloc[top_similar_indices][['Beer Name (Full)', 'country']]
            recommended_beers['Similarity Index'] = [round(similarity, 4) for _, similarity in sim_scores[1:top_n+1]]
    
            # Include the specified columns in the output DataFrame
            recommended_beers[['Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices']] = df2.loc[top_similar_indices, ['Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices']]
    
            # Display recommended beers with name and similarity index
            st.subheader("Recommended Beers:")
            st.table(recommended_beers[['Beer Name (Full)', 'Similarity Index']])
    
            return recommended_beers
    
        # Function to plot radar charts for recommended beers
        def radar_plot(beer_info):
            st.header("Radar Plots Section")
        
            # Display radar plot for each recommended beer
            for _, row in beer_info.iterrows():
                #st.subheader(f"Radar Plot for {row['Beer Name (Full)']}")
                radar_figure = plot_radar(row)
                st.pyplot(radar_figure)
        
        # Function to create a radar plot for a beer
        def plot_radar(beer_info):
            # Specify the columns for the radar plot
            radar_columns = ['Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices']
        
            # Set a fixed size for the plot
            fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
        
            # Filter columns based on their existence in the DataFrame
            valid_columns = [column for column in radar_columns if column in beer_info.index]
        
            if not valid_columns:
                st.warning(f"None of the specified columns exist in the beer information for {beer_info['Beer Name (Full)']}.")
                return fig
        
            # Number of variables
            num_vars = len(valid_columns)
        
            # Compute angles for radar plot with a closed loop
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
            # Ensure that values have the same length as angles
            values = beer_info[valid_columns].values.tolist()
        
            # Create radar plot
            ax.fill(angles, values, color='b', alpha=0.25)
            ax.set_title(beer_info['Beer Name (Full)'])
        
            # Add labels for each variable
            ax.set_xticks(angles)
            ax.set_xticklabels(valid_columns)
        
            # Return the plot figure
            return fig
    
        # Main Streamlit app
        st.title("Beer Recommendation App")
    
        # Beer Selector Section
        selected_beer = search_beer_partial(st.text_input("Enter partial beer name:", ""))
    
        # Recommender Section
        if selected_beer is not None:
            recommended_beers = recommend_beer(selected_beer)
    
            # Radar Plots Section
            if recommended_beers is not None:
                radar_plot(recommended_beers)
    
if __name__ == "__main__":
   main()