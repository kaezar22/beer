# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:32:47 2024

@author: ASUS
"""

import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px

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
    st.title("Beer Recommender")
    st.write("Under Construction")

if __name__ == "__main__":
    main()
