import streamlit as st
import plotly.express as px

st.title("# Profile")
left_col,right_col = st.columns(2)

#left_col.camera_input("pic",'s1s')
#right_col.slider('Pick a value',1,100)

st.divider()
countries = px.data.gapminder()
#another way to do that
with left_col:
    st.camera_input("pic",'s1s')

with right_col:
    st.slider('Pick a value',1,100)


tab1,tab2,tab3 = st.tabs(['tab1','tab2','tab3'])
with tab1:
    st.title("tab1 💕💕")
with tab2:
    st.write(countries)
with tab3:
    fig = px.scatter(data_frame=countries, x='gdpPercap',y='lifeExp',color='continent',animation_frame='year',size='pop',hover_data='country')
    st.plotly_chart(fig)