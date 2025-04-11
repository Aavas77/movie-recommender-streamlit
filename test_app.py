import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="🎬 Movie Recommendation Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("cleaned_movie_dataset.csv")
df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
df["release_decade"] = pd.cut(df["release_year"],
                              bins=[1900, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                              labels=["<1950", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])
df["primary_genre"] = df["genres"].apply(lambda x: x.split()[0] if isinstance(x, str) else "Unknown")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📊 Categorical Visuals", "📈 Relationships & Trends", "🔮 Recommendations"])


# --------------------- TAB 1: Overview ---------------------
with tab1:
    st.title("🎬 Content-Based Movie Recommendation System")
    st.markdown("""
    Welcome to the **Movie Recommendation Dashboard**!  
    This project recommends movies based on their **genres**, **keywords**, and **descriptions** using content-based filtering.

    Explore:
    - 📊 Category insights
    - 📈 Relationships and trends
    - 🔮 Movie recommendations
    """)

# --------------------- TAB 2: Categorical Visualizations ---------------------
with tab2:
    st.header("📊 Explore Categorical Features")

    cat_feat = st.selectbox("Choose a categorical feature to visualize",
                            options=["original_language", "primary_genre", "release_decade"])

    # Group minor categories into "Others"
    value_counts = df[cat_feat].value_counts()
    top_n = 10
    others_count = value_counts[top_n:].sum()

    # Create a new DataFrame for top categories + "Others"
    top_categories = value_counts[:top_n].reset_index()
    top_categories.columns = [cat_feat, "count"]
    top_categories.loc[len(top_categories.index)] = [f"Other ({len(value_counts) - top_n} categories)", others_count]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Bar Chart")
        fig_bar = px.bar(top_categories, x=cat_feat, y="count",
                         color=cat_feat, title=f"Bar Chart of {cat_feat}")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🥧 Pie Chart")
        fig_pie = px.pie(top_categories, names=cat_feat, values="count",
                         title=f"Pie Chart of {cat_feat}",
                         hole=0,
                         )

        fig_pie.update_traces(
            textinfo='percent+label',
            textposition='inside',
            insidetextorientation='radial'
        )

        st.plotly_chart(fig_pie, use_container_width=True)

# --------------------- TAB 3: Important Visualizations ---------------------
with tab3:
    st.header("📉 Key Relationships and Trends")

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    num_cols = ['popularity', 'runtime', 'vote_average', 'vote_count']
    corr_matrix = df[num_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis')
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatterplot selection
    st.subheader("🎯 Feature Relationships")
    scatter_option = st.selectbox("Choose a scatterplot", [
        "Popularity vs Vote Average", "Vote Count vs Vote Average", "Popularity vs Runtime"])

    if scatter_option == "Popularity vs Vote Average":
        fig = px.scatter(df, x="popularity", y="vote_average", color="primary_genre",
                         title=scatter_option, opacity=0.6)
    elif scatter_option == "Vote Count vs Vote Average":
        fig = px.scatter(df, x="vote_count", y="vote_average", color="primary_genre",
                         title=scatter_option, opacity=0.6)
    else:
        fig = px.scatter(df, x="popularity", y="runtime", color="primary_genre",
                         title=scatter_option, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    # Line chart: Popularity over time
    st.subheader("📈 Average Popularity Over Time")
    trend = df.groupby("release_year")["popularity"].mean().dropna().reset_index()
    fig_line = px.line(trend, x="release_year", y="popularity",
                       title="Popularity Trend", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

# --------------------- TAB 4: Recommendations ---------------------
with tab4:
    st.header("🔮 Movie Recommendation Engine")

    # Load data for recommendations
    df_movies = pd.read_csv("cleaned_movie_dataset.csv")
    df_sim_matrix = pd.read_csv("movie_similarity_matrix.csv")

    movie_list = df_movies['original_title'].dropna().unique()
    selected_movie = st.selectbox("Choose a movie to get recommendations", sorted(movie_list))

    def recommend_movies(movie_title, num_recommendations=5):
        if movie_title not in df_movies['original_title'].values:
            return []

        movie_index = df_movies[df_movies['original_title'] == movie_title].index[0]
        similarity_scores = df_sim_matrix.iloc[movie_index]
        similar_indices = similarity_scores.sort_values(ascending=False).index[1:num_recommendations + 1]
        return df_movies.iloc[similar_indices]['original_title'].values

    if st.button("🎬 Recommend"):
        recommendations = recommend_movies(selected_movie)
        st.subheader(f"🎥 Movies similar to *{selected_movie}*:")

        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

